use ratatui::prelude::*;
use ratatui::widgets::{Gauge, Paragraph};

use crate::app::{App, ProgressLogEntry, ProgressStyle as PStyle};
use crate::ui::theme::Theme;
use crate::ui::widgets::panel_block;

/// Render the progress/timeline panel.
///
/// The panel shows a rolling log of completed stages on top and any active
/// bars (spinner, download, weight-load, denoise) pinned to the bottom. The
/// caller supplies the panel title so the same renderer can serve both the
/// legacy "Progress" label and the design-system "Timeline" label.
pub fn render(frame: &mut Frame, app: &App, area: Rect, focused: bool) {
    render_with_title(frame, app, area, focused, "Timeline");
}

/// Identical to [`render`] but with a caller-supplied panel title.
pub fn render_with_title(frame: &mut Frame, app: &App, area: Rect, focused: bool, title: &str) {
    let theme = &app.theme;
    let progress = &app.generate.progress;

    let block = panel_block(theme, title, focused, None);

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    // Count active progress bars — `timeline_rows` is shared with the
    // unit tests so the "downloading-but-no-bytes-yet" edge case can't
    // silently regress the Timeline into a blank pane.
    let rows = timeline_rows(progress);

    let bar_lines = rows.total();
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

        if rows.overall {
            // Heartbeat row — visible for the entire generation so the
            // user can see the pipeline is still alive between stages.
            // Format: `▶ Generating · 4.3s · [3] Loading T5 encoder`.
            let total = progress
                .generation_elapsed()
                .map(format_elapsed)
                .unwrap_or_else(|| "0.0s".to_string());
            let mut spans: Vec<Span> = vec![
                Span::styled("\u{25B6} ", Style::default().fg(theme.accent)),
                Span::styled("Generating", Style::default().fg(theme.text)),
                Span::styled(" \u{00B7} ", theme.dim()),
                Span::styled(total, theme.dim()),
            ];
            if let Some(stage) = &progress.current_stage {
                spans.push(Span::styled(" \u{00B7} ", theme.dim()));
                if progress.stage_index > 0 {
                    spans.push(Span::styled(
                        format!("[{}] ", progress.stage_index),
                        theme.dim(),
                    ));
                }
                spans.push(Span::styled(
                    stage.as_str(),
                    Style::default().fg(theme.text),
                ));
            }
            let row = Rect {
                height: 1,
                ..bar_area
            };
            frame.render_widget(Paragraph::new(Line::from(spans)), row);
            bar_area.y += 1;
            bar_area.height = bar_area.height.saturating_sub(1);
        }

        if rows.spinner && bar_area.height > 0 {
            if let Some(stage) = &progress.current_stage {
                let spinner_char = spinner_frame();
                let mut spans = vec![
                    Span::styled(
                        format!("{spinner_char} "),
                        Style::default().fg(theme.accent),
                    ),
                    Span::styled(stage.as_str(), Style::default().fg(theme.text)),
                ];
                // Stage-local elapsed time — makes slow load stages feel
                // less frozen when no sub-gauge is available to update.
                if let Some(dur) = progress.stage_elapsed() {
                    spans.push(Span::styled(" \u{00B7} ", theme.dim()));
                    spans.push(Span::styled(format_elapsed(dur), theme.dim()));
                }
                let row = Rect {
                    height: 1,
                    ..bar_area
                };
                frame.render_widget(Paragraph::new(Line::from(spans)), row);
                bar_area.y += 1;
                bar_area.height = bar_area.height.saturating_sub(1);
            }
        }

        if rows.placeholder && bar_area.height > 0 {
            // hf-hub is mid-handshake — we know a pull is in flight
            // but not yet how big it is. Show a spinner row so the
            // user never sees a blank Timeline during a long pull.
            let spinner_char = spinner_frame();
            let filename = progress.download_filename.trim();
            let label = if filename.is_empty() {
                "Preparing model download...".to_string()
            } else {
                format!("Preparing {filename}...")
            };
            let line = Line::from(vec![
                Span::styled(
                    format!("{spinner_char} "),
                    Style::default().fg(theme.warning),
                ),
                Span::styled(label, Style::default().fg(theme.text)),
            ]);
            let row = Rect {
                height: 1,
                ..bar_area
            };
            frame.render_widget(Paragraph::new(line), row);
            bar_area.y += 1;
            bar_area.height = bar_area.height.saturating_sub(1);
        }

        if rows.download && bar_area.height > 0 {
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

        if rows.weight && bar_area.height > 0 {
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

        if rows.denoise && bar_area.height > 0 {
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

/// Whether the Timeline panel has anything to draw in its "active bars"
/// region for the given progress snapshot. Exposed as a pure predicate
/// so the "downloading-but-no-bytes-yet" placeholder behaviour is unit
/// testable without spinning up a real frame.
///
/// `has_download` reflects the full gauge (filename/bytes/eta).
/// `has_placeholder` reflects the indeterminate "Preparing download…"
/// row that should appear whenever `downloading` is true but no concrete
/// bytes have arrived yet *and* no spinner stage is set — without it the
/// Timeline stays blank during the `hf-hub` handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TimelineRows {
    pub overall: bool,
    pub spinner: bool,
    pub download: bool,
    pub placeholder: bool,
    pub weight: bool,
    pub denoise: bool,
}

impl TimelineRows {
    pub fn total(self) -> u16 {
        self.overall as u16
            + self.spinner as u16
            + self.download as u16
            + self.placeholder as u16
            + self.weight as u16
            + self.denoise as u16
    }
}

pub(crate) fn timeline_rows(progress: &crate::app::ProgressState) -> TimelineRows {
    let has_denoise = progress.denoise_total > 0 && progress.denoise_step < progress.denoise_total;
    let has_weight = progress.weight_total > 0 && progress.weight_loaded < progress.weight_total;
    let has_download = progress.download_total > 0;
    let has_spinner = progress.current_stage.is_some();
    // Downloading is set on the very first `hf-hub` event, before the
    // file-size resolver has populated `download_total`. Show an
    // indeterminate "preparing" row so the Timeline is never empty
    // while a pull is actually in flight.
    let has_placeholder = progress.is_downloading() && !has_download && !has_spinner;
    // The Overall row is the "you are still generating" heartbeat. It
    // renders whenever a generation is in flight *and* we're not purely
    // downloading (in which case the pull rows already tell the story).
    let has_overall = progress.generation_started_at.is_some() && !progress.is_downloading();
    TimelineRows {
        overall: has_overall,
        spinner: has_spinner,
        download: has_download,
        placeholder: has_placeholder,
        weight: has_weight,
        denoise: has_denoise,
    }
}

/// Format `d` as a compact elapsed timer, e.g. `4.3s`, `1m12s`, `1h02m`.
pub(crate) fn format_elapsed(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        let total_ms = d.as_millis();
        format!("{:.1}s", total_ms as f64 / 1000.0)
    } else if secs < 3600 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::ProgressState;

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

    #[test]
    fn timeline_shows_placeholder_when_downloading_has_no_bytes_yet() {
        // Codex-adjacent bug: during the hf-hub pre-flight the TUI set
        // `progress.downloading = true`, cleared `current_stage`, and had
        // `download_total = 0`. The Timeline then had nothing to render,
        // leaving the pane blank while the status bar said "Preparing…".
        let mut progress = ProgressState::default();
        progress.downloading = true;
        progress.download_total = 0;
        progress.current_stage = None;

        let rows = timeline_rows(&progress);
        assert!(
            rows.placeholder,
            "expected an indeterminate placeholder row while waiting on hf-hub"
        );
        assert!(!rows.download);
        assert!(!rows.spinner);
        assert_eq!(rows.total(), 1);
    }

    #[test]
    fn timeline_skips_placeholder_once_download_bar_is_live() {
        // Once real byte counts arrive, the full download gauge takes
        // over — the placeholder must not double up with it.
        let mut progress = ProgressState::default();
        progress.downloading = true;
        progress.download_total = 100;
        progress.download_bytes = 10;
        progress.current_stage = None;

        let rows = timeline_rows(&progress);
        assert!(rows.download);
        assert!(!rows.placeholder);
    }

    #[test]
    fn timeline_skips_placeholder_when_spinner_stage_set() {
        // A visible spinner/stage line is already telling the user what's
        // happening, so the placeholder would be redundant.
        let mut progress = ProgressState::default();
        progress.downloading = true;
        progress.current_stage = Some("Verifying weights".into());

        let rows = timeline_rows(&progress);
        assert!(rows.spinner);
        assert!(!rows.placeholder);
    }

    #[test]
    fn timeline_idle_when_not_downloading() {
        let progress = ProgressState::default();
        let rows = timeline_rows(&progress);
        assert_eq!(rows.total(), 0);
    }

    #[test]
    fn timeline_shows_overall_row_while_generating_even_without_gauges() {
        // User-reported: during the model-loading phase of a local run the
        // Timeline went silent between StageStart events — no gauge, no
        // spinner. The Overall row is the heartbeat that's always visible
        // while a generation is in flight, so the user can tell the
        // pipeline is still progressing.
        let mut progress = ProgressState::default();
        progress.mark_generation_start();
        assert!(progress.generation_started_at.is_some());

        let rows = timeline_rows(&progress);
        assert!(
            rows.overall,
            "Overall row must render for the duration of any generation"
        );
    }

    #[test]
    fn timeline_overall_hides_when_only_downloading() {
        // Pure pull (no subsequent generation) already has the download
        // gauge/placeholder telling the story — the Overall heartbeat
        // would just duplicate it.
        let mut progress = ProgressState::default();
        progress.downloading = true;
        progress.download_total = 100;
        progress.download_bytes = 10;
        // No generation started — we're only pulling.
        let rows = timeline_rows(&progress);
        assert!(!rows.overall);
    }

    #[test]
    fn timeline_overall_row_coexists_with_stage_spinner() {
        // During a real generation: Overall heartbeat on top, active
        // spinner row beneath it, plus whatever gauge applies.
        let mut progress = ProgressState::default();
        progress.mark_generation_start();
        progress.current_stage = Some("Loading T5 encoder".into());

        let rows = timeline_rows(&progress);
        assert!(rows.overall);
        assert!(rows.spinner);
        assert_eq!(rows.total(), 2);
    }

    #[test]
    fn format_elapsed_sub_minute_has_decimal() {
        assert_eq!(
            format_elapsed(std::time::Duration::from_millis(250)),
            "0.2s"
        );
        assert_eq!(
            format_elapsed(std::time::Duration::from_millis(4_300)),
            "4.3s"
        );
    }

    #[test]
    fn format_elapsed_rolls_into_minutes_and_hours() {
        assert_eq!(format_elapsed(std::time::Duration::from_secs(75)), "1m15s");
        assert_eq!(
            format_elapsed(std::time::Duration::from_secs(3_725)),
            "1h02m"
        );
    }
}
