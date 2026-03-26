use colored::Colorize;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use mold_core::SseProgressEvent;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::output::{is_piped, status};
use crate::theme;

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
        "flux" => padded.bright_magenta().to_string(),
        "flux2" => padded.magenta().to_string(),
        "sd15" => padded.green().to_string(),
        "sd3" | "sd3.5" => padded.bright_green().to_string(),
        "sdxl" => padded.yellow().to_string(),
        "z-image" => padded.cyan().to_string(),
        "qwen-image" | "qwen_image" => padded.bright_blue().to_string(),
        "wuerstchen" | "wuerstchen-v2" => padded.bright_yellow().to_string(),
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
        theme::prefix_error(),
        host
    );
    eprintln!("  {} {}", theme::prefix_cause(), err);
    eprintln!(
        "  {} start the server with {}",
        theme::prefix_hint(),
        "mold serve".bold()
    );
}

pub(crate) fn print_server_fallback(host: &str, action: &str) {
    status!(
        "{} Server unavailable at {} — {}",
        theme::icon_warn(),
        host.bold(),
        action,
    );
}

pub(crate) fn print_server_pull_missing_model(model: &str) {
    status!(
        "{} Model '{}' not on server — pulling...",
        theme::icon_info(),
        model.bold()
    );
}

pub(crate) fn print_using_local_inference() {
    status!("{} Using local GPU inference", theme::icon_info());
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

/// Sliding-window rate estimator for smoothing download speed/ETA display.
///
/// Tracks recent `(timestamp, cumulative_bytes)` samples and computes the
/// average rate across the window. This avoids the wild oscillation that
/// occurs when indicatif's built-in estimator sees zero progress between
/// coarse SSE events.
struct SmoothedRate {
    samples: VecDeque<(Instant, u64)>,
    max_samples: usize,
}

impl SmoothedRate {
    fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples + 1),
            max_samples,
        }
    }

    fn record(&mut self, position: u64) {
        let now = Instant::now();
        self.samples.push_back((now, position));
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
    }

    /// Average bytes per second across the sample window.
    fn rate_bps(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let (t_old, b_old) = self.samples.front().unwrap();
        let (t_new, b_new) = self.samples.back().unwrap();
        let dt = t_new.duration_since(*t_old).as_secs_f64();
        if dt < 0.001 {
            return 0.0;
        }
        b_new.saturating_sub(*b_old) as f64 / dt
    }

    /// Estimated seconds remaining given total size.
    fn eta_secs(&self, total: u64) -> f64 {
        let rate = self.rate_bps();
        if rate < 1.0 {
            return f64::INFINITY;
        }
        let current = self.samples.back().map(|(_, b)| *b).unwrap_or(0);
        total.saturating_sub(current) as f64 / rate
    }
}

fn format_speed(bps: f64) -> String {
    if bps < 1.0 {
        "0 B/s".to_string()
    } else if bps >= 1_073_741_824.0 {
        format!("{:.2} GiB/s", bps / 1_073_741_824.0)
    } else if bps >= 1_048_576.0 {
        format!("{:.2} MiB/s", bps / 1_048_576.0)
    } else if bps >= 1024.0 {
        format!("{:.2} KiB/s", bps / 1024.0)
    } else {
        format!("{:.0} B/s", bps)
    }
}

fn format_eta(secs: f64) -> String {
    if secs.is_infinite() || secs.is_nan() || secs > 359_999.0 {
        "--".to_string()
    } else if secs <= 0.0 {
        "0s".to_string()
    } else {
        let s = secs as u64;
        if s >= 3600 {
            format!("{}h{:02}m{:02}s", s / 3600, (s % 3600) / 60, s % 60)
        } else if s >= 60 {
            format!("{}m{:02}s", s / 60, s % 60)
        } else {
            format!("{s}s")
        }
    }
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
            .template(&format!("{{spinner:.{}}} {{msg}}", theme::SPINNER_STYLE))
            .unwrap(),
    );
    pb.tick();

    let mut denoise_bar: Option<ProgressBar> = None;
    let mut download_multi: Option<MultiProgress> = None;
    let mut download_bars: HashMap<usize, (ProgressBar, SmoothedRate)> = HashMap::new();
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
                        theme::icon_done(),
                        name,
                        format!("[{:.1}s]", secs).dimmed(),
                    );
                });
            }
            SseProgressEvent::Info { message } => {
                pb.suspend(|| {
                    status!("  {} {}", theme::icon_bullet(), message.dimmed());
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
                            .template(&format!(
                                "  {{spinner:.{c}}} Denoising [{{bar:30.{c}/dim}}] {{pos}}/{{len}} [{{elapsed_precise}}, {{msg}}]",
                                c = theme::SPINNER_STYLE,
                            ))
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
                let (bar, rate) = download_bars.entry(file_index).or_insert_with(|| {
                    let b = multi.add(ProgressBar::new(bytes_total));
                    let msg_width = 45usize;
                    b.set_style(
                        ProgressStyle::with_template(&format!(
                            "  {{msg:<{msg_width}}} [{{bar:30.{c}/dim}}] {{bytes}}/{{total_bytes}} ({{prefix}})",
                            c = theme::SPINNER_STYLE,
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
                    b.enable_steady_tick(Duration::from_millis(100));
                    (b, SmoothedRate::new(8))
                });
                rate.record(bytes_downloaded);
                bar.set_prefix(format!(
                    "{}, {}",
                    format_speed(rate.rate_bps()),
                    format_eta(rate.eta_secs(bytes_total))
                ));
                bar.set_position(bytes_downloaded.max(bar.position()));
            }
            SseProgressEvent::DownloadDone { file_index, .. } => {
                if let Some((bar, _)) = download_bars.get(&file_index) {
                    bar.finish_with_message("done");
                }
            }
            SseProgressEvent::Queued { position } => {
                if position > 0 {
                    pb.set_message(format!("Queued (position {})", position));
                    pb.enable_steady_tick(Duration::from_millis(100));
                }
            }
            SseProgressEvent::PullComplete { model } => {
                for (_, (bar, _)) in download_bars.drain() {
                    bar.finish_and_clear();
                }
                if let Some(multi) = download_multi.take() {
                    multi.clear().ok();
                }
                pb.suspend(|| {
                    status!("{} Pull complete: {}", theme::icon_done(), model.bold());
                });
            }
        }
    }
    if let Some(db) = denoise_bar.take() {
        db.finish_and_clear();
    }
    for (_, (bar, _)) in download_bars.drain() {
        bar.finish_and_clear();
    }
    if let Some(multi) = download_multi.take() {
        multi.clear().ok();
    }
    pb.finish_and_clear();
}

fn truncate_name(name: &str, max_len: usize) -> String {
    let char_count = name.chars().count();
    if char_count <= max_len || max_len < 8 {
        return name.to_string();
    }
    let suffix_len = max_len - 3;
    let suffix: String = name.chars().skip(char_count - suffix_len).collect();
    format!("...{suffix}")
}

fn format_cache_hit_badge(resource: &str) -> String {
    format!(
        "{} {} {}",
        theme::icon_done(),
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

    // ── SmoothedRate tests ───────────────────────────────────────────────

    #[test]
    fn smoothed_rate_empty_returns_unknown() {
        let rate = SmoothedRate::new(8);
        assert_eq!(rate.rate_bps(), 0.0);
        assert!(rate.eta_secs(1000).is_infinite());
    }

    #[test]
    fn smoothed_rate_single_sample_returns_zero() {
        let mut rate = SmoothedRate::new(8);
        rate.samples.push_back((Instant::now(), 500));
        assert_eq!(rate.rate_bps(), 0.0);
    }

    #[test]
    fn smoothed_rate_computes_correct_rate() {
        let mut rate = SmoothedRate::new(8);
        let now = Instant::now();
        // 1000 bytes over 1 second = 1000 B/s
        rate.samples.push_back((now - Duration::from_secs(1), 0));
        rate.samples.push_back((now, 1000));
        let bps = rate.rate_bps();
        assert!((bps - 1000.0).abs() < 50.0, "expected ~1000 B/s, got {bps}");
    }

    #[test]
    fn smoothed_rate_eta_calculation() {
        let mut rate = SmoothedRate::new(8);
        let now = Instant::now();
        // 500 bytes over 1 second = 500 B/s, 500 bytes remaining
        rate.samples.push_back((now - Duration::from_secs(1), 0));
        rate.samples.push_back((now, 500));
        let eta = rate.eta_secs(1000);
        assert!((eta - 1.0).abs() < 0.1, "expected ~1s ETA, got {eta}");
    }

    #[test]
    fn smoothed_rate_evicts_old_samples() {
        let mut rate = SmoothedRate::new(4);
        for i in 0..10 {
            rate.record(i * 100);
        }
        assert_eq!(rate.samples.len(), 4);
    }

    #[test]
    fn smoothed_rate_never_negative() {
        let mut rate = SmoothedRate::new(8);
        let now = Instant::now();
        // position stays the same — rate should be 0, not negative
        rate.samples.push_back((now - Duration::from_secs(1), 1000));
        rate.samples.push_back((now, 1000));
        assert_eq!(rate.rate_bps(), 0.0);
    }

    // ── format_speed / format_eta tests ──────────────────────────────────

    #[test]
    fn format_speed_zero() {
        assert_eq!(format_speed(0.0), "0 B/s");
    }

    #[test]
    fn format_speed_bytes() {
        assert_eq!(format_speed(512.0), "512 B/s");
    }

    #[test]
    fn format_speed_kib() {
        assert_eq!(format_speed(2048.0), "2.00 KiB/s");
    }

    #[test]
    fn format_speed_mib() {
        assert_eq!(format_speed(36.99 * 1_048_576.0), "36.99 MiB/s");
    }

    #[test]
    fn format_speed_gib() {
        assert_eq!(format_speed(1_073_741_824.0), "1.00 GiB/s");
    }

    #[test]
    fn format_eta_zero() {
        assert_eq!(format_eta(0.0), "0s");
    }

    #[test]
    fn format_eta_seconds() {
        assert_eq!(format_eta(45.0), "45s");
    }

    #[test]
    fn format_eta_minutes() {
        assert_eq!(format_eta(125.0), "2m05s");
    }

    #[test]
    fn format_eta_hours() {
        assert_eq!(format_eta(3661.0), "1h01m01s");
    }

    #[test]
    fn format_eta_very_large() {
        assert_eq!(format_eta(999_999.0), "--");
    }

    #[test]
    fn format_eta_negative() {
        assert_eq!(format_eta(-5.0), "0s");
    }

    #[test]
    fn format_eta_infinity() {
        assert_eq!(format_eta(f64::INFINITY), "--");
    }

    #[test]
    fn format_eta_nan() {
        assert_eq!(format_eta(f64::NAN), "--");
    }
}
