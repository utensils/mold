use anyhow::{Context, Result};
use colored::Colorize;
use mold_core::types::{DeviceAdminState, DeviceHealth, GpuWorkerState, QueuePlan, QueueWorkItem};
use mold_core::{classify_server_error, ServerAvailability};

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
                    println!(
                        "{} · GPU {} ({}):  {:<20} {}  VRAM: {}",
                        device.id,
                        device
                            .ordinal
                            .map_or_else(|| "—".into(), |value| value.to_string()),
                        device.name,
                        model.green(),
                        state_str,
                        format_vram(device.memory.used_bytes, device.memory.total_bytes),
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

            if let Ok(queue) = ctx.client().list_queue().await {
                if let Some(plan) = queue.plan {
                    print_queue_plan(&plan);
                }
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
        Err(error) => {
            if crate::control::is_loopback_host(ctx.client().host())
                && classify_server_error(&error) == ServerAvailability::FallbackLocal
            {
                println!(
                    "  {} no mold server running — start with {}",
                    theme::prefix_hint(),
                    "mold serve".bold()
                );
                let local = crate::commands::gpu::local_device_state()?;
                if !local.devices.is_empty() {
                    println!();
                    println!("Local compute (startup preferences):");
                    for device in &local.devices {
                        println!("  {}", crate::commands::gpu::format_device_line(device));
                    }
                }
            } else {
                return Err(error).context(format!(
                    "failed to read mold server status from {}",
                    ctx.client().host()
                ));
            }
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

fn format_vram(used: Option<u64>, total: Option<u64>) -> String {
    match (used, total) {
        (Some(used), Some(total)) => format!(
            "{:.1}/{:.1} GiB",
            used as f64 / 1024_f64.powi(3),
            total as f64 / 1024_f64.powi(3)
        ),
        _ => "—".into(),
    }
}

fn format_work_item(item: &QueueWorkItem) -> String {
    let order = item.lane_order.unwrap_or_default();
    let lane = match item.planned_lane_kind.as_ref() {
        Some(mold_core::QueuePlannedLaneKind::HostUtility) => {
            format!("host-utility/{order}")
        }
        Some(mold_core::QueuePlannedLaneKind::Device) => item
            .planned_device_id
            .as_deref()
            .map(|device| format!("{device}/{order}"))
            .unwrap_or_else(|| format!("device/{order}")),
        Some(mold_core::QueuePlannedLaneKind::Unknown(_)) => format!("assigned/{order}"),
        None if item.is_host_utility_lane()
            || item.activity_phase == mold_core::QueueActivityPhase::Cpu =>
        {
            format!("host-utility/{order}")
        }
        None => item
            .planned_device_id
            .as_deref()
            .map(|device| format!("{device}/{order}"))
            .unwrap_or_else(|| "unassigned".into()),
    };
    let blocked = item
        .blocked_reason
        .as_ref()
        .map(|reason| format!(" blocked={}", reason.as_str()))
        .unwrap_or_default();
    let eta = item
        .estimated_finish_unix_ms
        .map(|finish| format!(" finish={finish}"))
        .unwrap_or_default();
    format!(
        "{} {} phase={} lane={} confidence={}{}{}",
        item.work_id,
        item.work_kind,
        item.activity_phase,
        lane,
        item.estimate_confidence,
        blocked,
        eta
    )
}

fn print_queue_plan(plan: &QueuePlan) {
    println!();
    println!(
        "Queue plan v{} (state {}, optimizer {})",
        plan.plan_version, plan.state_version, plan.optimizer_state
    );
    if let Some(next) = plan.next_replan_at_unix_ms {
        println!("Next replan: {next}");
    }
    for item in &plan.work_items {
        println!("  {}", format_work_item(item));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_vram_is_an_em_dash_instead_of_fake_zero() {
        assert_eq!(format_vram(None, Some(24 << 30)), "—");
        assert_eq!(format_vram(Some(0), None), "—");
    }

    #[test]
    fn host_utility_lane_never_formats_an_internal_device_id() {
        for planned_lane_kind in [Some(mold_core::QueuePlannedLaneKind::HostUtility), None] {
            let item = QueueWorkItem {
                work_id: "upscale".into(),
                activity_phase: mold_core::QueueActivityPhase::Cpu,
                planned_lane_kind,
                planned_device_id: Some("internal-id-must-not-render".into()),
                lane_order: Some(0),
                ..Default::default()
            };
            let formatted = format_work_item(&item);
            assert!(formatted.contains("host-utility/0"));
            assert!(!formatted.contains("internal-id-must-not-render"));
        }
    }

    #[test]
    fn legacy_queued_host_utility_lane_never_formats_its_internal_device_id() {
        let item = QueueWorkItem {
            work_id: "legacy-upscale".into(),
            activity_phase: mold_core::QueueActivityPhase::Queued,
            planned_lane_kind: None,
            planned_device_id: Some("cpu:utility:0".into()),
            lane_order: Some(1),
            ..Default::default()
        };

        let formatted = format_work_item(&item);
        assert!(formatted.contains("host-utility/1"));
        assert!(!formatted.contains("cpu:utility:0"));
    }

    #[test]
    fn typed_device_and_future_lanes_keep_public_lane_semantics() {
        let gpu = format_work_item(&QueueWorkItem {
            planned_lane_kind: Some(mold_core::QueuePlannedLaneKind::Device),
            planned_device_id: Some("cuda:stable-a".into()),
            lane_order: Some(2),
            ..Default::default()
        });
        assert!(gpu.contains("cuda:stable-a/2"));

        let future = format_work_item(&QueueWorkItem {
            planned_lane_kind: Some(mold_core::QueuePlannedLaneKind::Unknown(
                "remote_utility".into(),
            )),
            planned_device_id: Some("future-internal-id".into()),
            lane_order: Some(3),
            ..Default::default()
        });
        assert!(future.contains("assigned/3"));
        assert!(!future.contains("future-internal-id"));
    }
}
