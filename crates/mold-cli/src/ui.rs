use colored::Colorize;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use mold_core::SseProgressEvent;
use std::collections::HashMap;
use std::time::Duration;

use crate::output::{is_piped, status};

pub(crate) fn family_label(family: &str) -> &str {
    match family {
        "flux" => "FLUX.1",
        "flux2" => "FLUX.2",
        "sd15" => "SD 1.5",
        "sd3" | "sd3.5" => "SD 3.5",
        "sdxl" => "SDXL",
        "z-image" => "Z-Image",
        "qwen-image" | "qwen_image" => "Qwen-Image",
        "wuerstchen" | "wuerstchen-v2" => "Wuerstchen",
        "controlnet" => "ControlNet",
        other => other,
    }
}

pub(crate) fn format_family_padded(family: &str, width: usize) -> String {
    let padded = format!("{:<width$}", family_label(family), width = width);
    match family {
        "flux" => padded.truecolor(200, 120, 255).to_string(),
        "flux2" => padded.truecolor(255, 150, 255).to_string(),
        "sd15" => padded.green().to_string(),
        "sd3" | "sd3.5" => padded.truecolor(100, 220, 160).to_string(),
        "sdxl" => padded.yellow().to_string(),
        "z-image" => padded.cyan().to_string(),
        "qwen-image" | "qwen_image" => padded.truecolor(100, 200, 255).to_string(),
        "wuerstchen" | "wuerstchen-v2" => padded.truecolor(255, 180, 80).to_string(),
        "controlnet" => padded.bright_red().to_string(),
        _ => padded,
    }
}

pub(crate) fn format_family(family: &str) -> String {
    if matches!(
        family,
        "flux"
            | "flux2"
            | "sd15"
            | "sd3"
            | "sd3.5"
            | "sdxl"
            | "z-image"
            | "qwen-image"
            | "qwen_image"
            | "wuerstchen"
            | "wuerstchen-v2"
            | "controlnet"
    ) {
        format_family_padded(family, family_label(family).len())
            .trim_end()
            .to_string()
    } else {
        family.to_uppercase()
    }
}

pub(crate) fn print_server_unavailable(host: &str, err: &dyn std::fmt::Display) {
    eprintln!(
        "{} cannot connect to mold server at {}",
        "error:".red().bold(),
        host
    );
    eprintln!("  {} {}", "cause:".dimmed(), err);
    eprintln!(
        "  {} start the server with {}",
        "hint:".dimmed(),
        "mold serve".bold()
    );
}

pub(crate) fn print_server_fallback(host: &str, action: &str) {
    status!(
        "{} Server unavailable at {} — {}",
        "●".yellow(),
        host.bold(),
        action,
    );
}

pub(crate) fn print_server_pull_missing_model(model: &str) {
    status!(
        "{} Model '{}' not on server — pulling...",
        "●".cyan(),
        model.bold()
    );
}

pub(crate) fn print_using_local_inference() {
    status!("{} Using local GPU inference", "●".cyan());
}

/// Human-readable byte size with space before unit (e.g. "7.0 GB").
/// Used in verbose contexts like `mold rm` confirmation output.
pub(crate) fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Compact byte size without space (e.g. "7.0GB").
/// Used in dense table columns like `mold list`.
pub(crate) fn format_disk_size(bytes: u64) -> String {
    if bytes == 0 {
        "—".to_string()
    } else if bytes >= 1_073_741_824 {
        format!("{:.1}GB", bytes as f64 / 1_073_741_824.0)
    } else {
        format!("{:.0}MB", bytes as f64 / 1_048_576.0)
    }
}

pub(crate) fn col_width(
    items: impl IntoIterator<Item = usize>,
    header_len: usize,
    pad: usize,
) -> usize {
    items.into_iter().fold(header_len, usize::max) + pad
}

pub(crate) async fn render_progress(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<SseProgressEvent>,
) {
    let pb = ProgressBar::new_spinner();
    if is_piped() {
        pb.set_draw_target(indicatif::ProgressDrawTarget::stderr());
    }
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    pb.tick();

    let mut denoise_bar: Option<ProgressBar> = None;
    let mut download_multi: Option<MultiProgress> = None;
    let mut download_bars: HashMap<usize, ProgressBar> = HashMap::new();
    while let Some(event) = rx.recv().await {
        match event {
            SseProgressEvent::StageStart { name } => {
                if let Some(db) = denoise_bar.take() {
                    db.finish_and_clear();
                }
                pb.set_message(format!("{}...", name));
                pb.tick();
            }
            SseProgressEvent::StageDone { name, elapsed_ms } => {
                if let Some(db) = denoise_bar.take() {
                    db.finish_and_clear();
                }
                let secs = elapsed_ms as f64 / 1000.0;
                pb.suspend(|| {
                    status!(
                        "  {} {} {}",
                        "✓".green(),
                        name,
                        format!("[{:.1}s]", secs).dimmed(),
                    );
                });
            }
            SseProgressEvent::Info { message } => {
                pb.suspend(|| {
                    status!("  {} {}", "·".dimmed(), message.dimmed());
                });
            }
            SseProgressEvent::CacheHit { resource } => {
                pb.suspend(|| {
                    status!("  {}", format_cache_hit_badge(&resource));
                });
            }
            SseProgressEvent::DenoiseStep {
                step,
                total,
                elapsed_ms,
            } => {
                let db = denoise_bar.get_or_insert_with(|| {
                    pb.disable_steady_tick();
                    pb.set_message("");

                    let bar = ProgressBar::new(total as u64);
                    if is_piped() {
                        bar.set_draw_target(ProgressDrawTarget::stderr());
                    }
                    bar.set_style(
                        ProgressStyle::default_bar()
                            .template(
                                "  {spinner:.cyan} Denoising [{bar:30.cyan/dim}] {pos}/{len} [{elapsed_precise}, {msg}]",
                            )
                            .unwrap()
                            .progress_chars("━╸─"),
                    );
                    bar.enable_steady_tick(Duration::from_millis(100));
                    bar
                });
                let elapsed_secs = elapsed_ms as f64 / 1000.0;
                let it_s = if elapsed_secs > 0.0 {
                    1.0 / elapsed_secs
                } else {
                    0.0
                };
                db.set_message(format!("{:.2} it/s", it_s));
                db.set_position(step as u64);
            }
            SseProgressEvent::DownloadProgress {
                filename,
                file_index,
                bytes_downloaded,
                bytes_total,
                total_files,
            } => {
                let multi = download_multi.get_or_insert_with(|| {
                    pb.disable_steady_tick();
                    pb.set_message("");
                    MultiProgress::with_draw_target(ProgressDrawTarget::stderr())
                });
                let bar = download_bars.entry(file_index).or_insert_with(|| {
                    let b = multi.add(ProgressBar::new(bytes_total));
                    let msg_width = 45usize;
                    b.set_style(
                        ProgressStyle::with_template(&format!(
                            "  {{msg:<{msg_width}}} [{{bar:30.cyan/dim}}] {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, {{eta}})"
                        ))
                        .unwrap()
                        .progress_chars("━╸─"),
                    );
                    if total_files > 0 {
                        b.set_message(format!(
                            "[{}/{}] {}",
                            file_index + 1,
                            total_files,
                            truncate_name(&filename, msg_width - 8)
                        ));
                    } else {
                        b.set_message(truncate_name(&filename, msg_width));
                    }
                    b
                });
                bar.set_position(bytes_downloaded);
            }
            SseProgressEvent::DownloadDone { file_index, .. } => {
                if let Some(bar) = download_bars.get(&file_index) {
                    bar.finish_with_message("done");
                }
            }
            SseProgressEvent::PullComplete { model } => {
                for (_, bar) in download_bars.drain() {
                    bar.finish_and_clear();
                }
                if let Some(multi) = download_multi.take() {
                    multi.clear().ok();
                }
                pb.suspend(|| {
                    status!("{} Pull complete: {}", "✓".green(), model.bold());
                });
            }
        }
    }
    if let Some(db) = denoise_bar.take() {
        db.finish_and_clear();
    }
    for (_, bar) in download_bars.drain() {
        bar.finish_and_clear();
    }
    if let Some(multi) = download_multi.take() {
        multi.clear().ok();
    }
    pb.finish_and_clear();
}

fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len || max_len < 8 {
        return name.to_string();
    }
    let suffix_len = max_len - 3;
    let start = name.len() - suffix_len;
    format!("...{}", &name[start..])
}

fn format_cache_hit_badge(resource: &str) -> String {
    format!(
        "{} {} {}",
        "✓".green(),
        resource,
        "[cache hit]".bright_cyan().bold()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_family_flux_contains_label() {
        let result = format_family_padded("flux", 10);
        assert!(result.contains("FLUX.1"));
    }

    #[test]
    fn format_family_sdxl_contains_label() {
        let result = format_family_padded("sdxl", 10);
        assert!(result.contains("SDXL"));
    }

    #[test]
    fn format_family_sd15_contains_label() {
        let result = format_family_padded("sd15", 10);
        assert!(result.contains("SD 1.5"));
    }

    #[test]
    fn format_family_unknown_passthrough() {
        let result = format_family_padded("custom", 10);
        assert!(result.contains("custom"));
    }

    #[test]
    fn format_family_padded_respects_width() {
        let result = format_family_padded("sdxl", 20);
        assert!(
            result.len() >= 20,
            "result was only {} chars: {:?}",
            result.len(),
            result
        );
    }

    #[test]
    fn format_family_unknown_uppercases() {
        assert_eq!(format_family("other"), "OTHER");
    }

    #[test]
    fn format_bytes_gb() {
        assert_eq!(format_bytes(7_516_192_768), "7.0 GB");
    }

    #[test]
    fn format_bytes_mb() {
        assert_eq!(format_bytes(52_428_800), "50.0 MB");
    }

    #[test]
    fn format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    #[test]
    fn format_disk_size_zero_is_dash() {
        assert_eq!(format_disk_size(0), "—");
    }

    #[test]
    fn col_width_respects_header_and_padding() {
        assert_eq!(col_width([3usize, 9, 5], 4, 2), 11);
    }

    #[test]
    fn cache_hit_badge_includes_resource_and_marker() {
        let badge = format_cache_hit_badge("prompt conditioning");
        assert!(badge.contains("prompt conditioning"));
        assert!(badge.contains("cache hit"));
    }
}
