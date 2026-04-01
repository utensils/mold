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

    // Decide what to show in the available space
    let has_denoise = progress.denoise_total > 0 && progress.denoise_step < progress.denoise_total;
    let has_weight = progress.weight_total > 0 && progress.weight_loaded < progress.weight_total;
    let has_spinner = progress.current_stage.is_some();

    // Layout: log lines, then active bars
    let bar_lines = has_denoise as u16 + has_weight as u16 + has_spinner as u16;
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

    // Show the most recent entries that fit
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

fn format_bytes(bytes: u64) -> String {
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

fn spinner_frame() -> char {
    // Simple rotating spinner based on time
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
