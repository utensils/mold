use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Gauge, Paragraph};

use crate::app::{App, ProgressLogEntry, ProgressStyle as PStyle};
use crate::ui::theme::Theme;

/// Render the progress panel at the bottom of the Generate view.
pub fn render(frame: &mut Frame, app: &App, area: Rect, focused: bool) {
    let theme = &app.theme;
    let progress = &app.generate.progress;

    let border_style = if focused {
        theme.border_focused()
    } else {
        theme.border()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" Progress ")
        .title_style(if focused {
            theme.title_focused()
        } else {
            theme.title()
        })
        .style(Style::default().bg(theme.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    // Count active progress bars
    let has_denoise = progress.denoise_total > 0 && progress.denoise_step < progress.denoise_total;
    let has_weight = progress.weight_total > 0 && progress.weight_loaded < progress.weight_total;
    let has_download = progress.download_total > 0;
    let has_spinner = progress.current_stage.is_some();

    let bar_lines =
        has_denoise as u16 + has_weight as u16 + has_download as u16 + has_spinner as u16;
    let log_height = inner.height.saturating_sub(bar_lines);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(log_height), Constraint::Min(0)])
        .split(inner);

    // ── Completed stage log ────────────────────────────────
    render_log(frame, theme, &progress.log, layout[0]);

    // ── Active progress bars ───────────────────────────────
    if layout[1].height > 0 {
        let mut bar_area = layout[1];

        if has_spinner {
            if let Some(stage) = &progress.current_stage {
                let spinner_char = spinner_frame();
                let line = Line::from(vec![
                    Span::styled(
                        format!("{spinner_char} "),
                        Style::default().fg(theme.accent),
                    ),
                    Span::styled(stage, Style::default().fg(theme.text)),
                ]);
                let p = Paragraph::new(line);
                let row = Rect {
                    height: 1,
                    ..bar_area
                };
                frame.render_widget(p, row);
                bar_area.y += 1;
                bar_area.height = bar_area.height.saturating_sub(1);
            }
        }

        if has_download && bar_area.height > 0 {
            let pct = if progress.download_batch_total > 0 {
                (progress.download_batch_bytes as f64 / progress.download_batch_total as f64)
                    .min(1.0)
            } else {
                0.0
            };
            let transfer = if let (Some(rate), Some(eta_secs)) =
                (progress.download_rate_bps, progress.download_eta_secs)
            {
                format!(
                    ", {}/s, eta {}",
                    format_bytes_binary(rate),
                    format_eta(eta_secs.ceil() as u64)
                )
            } else {
                String::new()
            };
            let label = if progress.download_total_files > 0 {
                format!(
                    "[{}/{}] {} [{}/{} total{}]",
                    progress.download_file_index + 1,
                    progress.download_total_files,
                    progress.download_filename,
                    format_bytes(progress.download_batch_bytes),
                    format_bytes(progress.download_batch_total),
                    transfer,
                )
            } else {
                format!(
                    "{} [{}/{} total{}]",
                    progress.download_filename,
                    format_bytes(progress.download_batch_bytes),
                    format_bytes(progress.download_batch_total),
                    transfer,
                )
            };
            let gauge = Gauge::default()
                .ratio(pct)
                .label(label)
                .gauge_style(Style::default().fg(theme.warning).bg(theme.progress_empty));
            let row = Rect {
                height: 1,
                ..bar_area
            };
            frame.render_widget(gauge, row);
            bar_area.y += 1;
            bar_area.height = bar_area.height.saturating_sub(1);
        }

        if has_weight && bar_area.height > 0 {
            let pct = if progress.weight_total > 0 {
                (progress.weight_loaded as f64 / progress.weight_total as f64).min(1.0)
            } else {
                0.0
            };
            let label = format!(
                "Loading {} [{}/{}]",
                progress.weight_component,
                format_bytes(progress.weight_loaded),
                format_bytes(progress.weight_total),
            );
            let gauge = Gauge::default()
                .ratio(pct)
                .label(label)
                .gauge_style(theme.progress_filled())
                .style(theme.progress_empty());
            let row = Rect {
                height: 1,
                ..bar_area
            };
            frame.render_widget(gauge, row);
            bar_area.y += 1;
            bar_area.height = bar_area.height.saturating_sub(1);
        }

        if has_denoise && bar_area.height > 0 {
            let pct = progress.denoise_step as f64 / progress.denoise_total as f64;
            let rate = if progress.denoise_elapsed_ms > 0 && progress.denoise_step > 0 {
                progress.denoise_step as f64 / (progress.denoise_elapsed_ms as f64 / 1000.0)
            } else {
                0.0
            };
            let label = format!(
                "Denoising {}/{} [{:.1} it/s]",
                progress.denoise_step, progress.denoise_total, rate,
            );
            let gauge = Gauge::default()
                .ratio(pct.min(1.0))
                .label(label)
                .gauge_style(theme.progress_filled())
                .style(theme.progress_empty());
            let row = Rect {
                height: 1,
                ..bar_area
            };
            frame.render_widget(gauge, row);
        }
    }
}

fn render_log(frame: &mut Frame, theme: &Theme, log: &[ProgressLogEntry], area: Rect) {
    if area.height == 0 {
        return;
    }

    let visible = log.len().min(area.height as usize);
    let start = log.len().saturating_sub(visible);

    let lines: Vec<Line> = log[start..]
        .iter()
        .map(|entry| {
            let (icon, style) = match entry.style {
                PStyle::Done => ("\u{2713}", theme.success()), // checkmark
                PStyle::Info => ("\u{2022}", theme.dim()),     // bullet
                PStyle::Warning => ("!", theme.warning()),
                PStyle::Error => ("\u{2717}", theme.error()), // x mark
            };
            Line::from(vec![
                Span::styled(format!("{icon} "), style),
                Span::styled(&entry.message, style),
            ])
        })
        .collect();

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, area);
}

pub(crate) fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1}G", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.0}M", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.0}K", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}B")
    }
}

pub(crate) fn format_eta(seconds: u64) -> String {
    match seconds {
        0..=59 => format!("{seconds}s"),
        60..=3599 => format!("{}m{:02}s", seconds / 60, seconds % 60),
        _ => format!("{}h{:02}m", seconds / 3600, (seconds % 3600) / 60),
    }
}

pub(crate) fn format_bytes_binary(bytes: f64) -> String {
    if bytes >= 1_073_741_824.0 {
        format!("{:.2}GiB", bytes / 1_073_741_824.0)
    } else if bytes >= 1_048_576.0 {
        format!("{:.2}MiB", bytes / 1_048_576.0)
    } else if bytes >= 1024.0 {
        format!("{:.2}KiB", bytes / 1024.0)
    } else {
        format!("{:.0}B", bytes)
    }
}

fn spinner_frame() -> char {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let frames = [
        '\u{2801}', '\u{2802}', '\u{2804}', '\u{2840}', '\u{2820}', '\u{2810}', '\u{2808}',
        '\u{2800}',
    ];
    frames[(ms / 100 % frames.len() as u128) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_eta_short_values() {
        assert_eq!(format_eta(7), "7s");
        assert_eq!(format_eta(65), "1m05s");
        assert_eq!(format_eta(3665), "1h01m");
    }

    #[test]
    fn format_bytes_binary_uses_cli_style_units() {
        assert_eq!(format_bytes_binary(512.0), "512B");
        assert_eq!(format_bytes_binary(2_048.0), "2.00KiB");
        assert_eq!(format_bytes_binary(3.5 * 1_048_576.0), "3.50MiB");
    }
}
