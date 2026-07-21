//! Create Timeline panel — the glyph-styled session log.
//!
//! Restyles the log path of the old `ui::progress` renderer: `•` info,
//! `✓` success (stage lines keep their `[0.3s]` suffixes), `!` warning,
//! `✗` error, and `★` for loaded-model lines. The Recent strip's role is
//! absorbed here via the `✓ Saved <file>` entries the completion handler
//! pushes. Download/weight gauges stay pinned to the bottom during pulls
//! (`ui::progress` keeps those helpers); denoise progress lives in the
//! Preview panel now.

use ratatui::prelude::*;
use ratatui::widgets::{Gauge, Paragraph};

use crate::app::{App, ProgressLogEntry, ProgressStyle as PStyle};
use crate::ui::progress;
use crate::ui::theme::Theme;
use crate::ui::widgets::panel_block;

/// Log rows visible in the panel (its height budget lives in
/// `ui::generate::TIMELINE_HEIGHT`, guarded by a const assert there).
pub(crate) const TIMELINE_VISIBLE_ROWS: usize = 5;

/// Empty-state copy (mock-exact).
pub(crate) const EMPTY_STATE: &str = "\u{2014} idle. no runs this session.";

/// The range of `log` that fits `visible` rows — the newest entries win.
pub(crate) fn tail_range(len: usize, visible: usize) -> std::ops::Range<usize> {
    len.saturating_sub(visible)..len
}

/// Glyph for one log entry. Loaded-model success lines get the mock's `★`.
pub(crate) fn entry_glyph(style: PStyle, message: &str) -> &'static str {
    match style {
        PStyle::Done if message.to_lowercase().contains("loaded") => "\u{2605}",
        PStyle::Done => "\u{2713}",
        PStyle::Info => "\u{2022}",
        PStyle::Warning => "!",
        PStyle::Error => "\u{2717}",
    }
}

fn entry_style(theme: &Theme, style: PStyle) -> Style {
    match style {
        PStyle::Done => theme.success(),
        PStyle::Info => theme.info(),
        PStyle::Warning => theme.warning(),
        PStyle::Error => theme.error(),
    }
}

/// Render the Timeline panel.
pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;
    let progress_state = &app.generate.progress;

    let block = panel_block(theme, "Timeline", false, None);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    // Bottom-pinned gauges while a pull is in flight.
    let rows = progress::timeline_rows(progress_state);
    let show_download = rows.download || rows.placeholder;
    let gauge_rows = (show_download as u16 + rows.weight as u16).min(inner.height);
    let log_height = inner.height - gauge_rows;

    if progress_state.log.is_empty() && gauge_rows == 0 {
        frame.render_widget(
            Paragraph::new(EMPTY_STATE).style(theme.faint()),
            Rect { height: 1, ..inner },
        );
        return;
    }

    // ── Glyph log tail ─────────────────────────────────────────
    let visible = (log_height as usize).min(TIMELINE_VISIBLE_ROWS);
    let range = tail_range(progress_state.log.len(), visible);
    let lines: Vec<Line> = progress_state.log[range]
        .iter()
        .map(|entry: &ProgressLogEntry| {
            Line::from(vec![
                Span::styled(
                    format!("{} ", entry_glyph(entry.style, &entry.message)),
                    entry_style(theme, entry.style),
                ),
                Span::styled(entry.message.clone(), Style::default().fg(theme.text)),
            ])
        })
        .collect();
    frame.render_widget(
        Paragraph::new(lines),
        Rect {
            height: log_height,
            ..inner
        },
    );

    // ── Pull gauges ────────────────────────────────────────────
    let mut y = inner.y + log_height;
    if show_download && y < inner.y + inner.height {
        let row = Rect {
            x: inner.x,
            y,
            width: inner.width,
            height: 1,
        };
        if rows.download {
            let pct = if progress_state.download_batch_total > 0 {
                (progress_state.download_batch_bytes as f64
                    / progress_state.download_batch_total as f64)
                    .min(1.0)
            } else {
                0.0
            };
            let label = format!(
                "{} [{}/{}]",
                progress_state.download_filename,
                progress::format_bytes(progress_state.download_batch_bytes),
                progress::format_bytes(progress_state.download_batch_total),
            );
            let gauge = Gauge::default()
                .ratio(pct)
                .label(label)
                .gauge_style(Style::default().fg(theme.warning).bg(theme.progress_empty));
            frame.render_widget(gauge, row);
        } else {
            let filename = progress_state.download_filename.trim();
            let label = if filename.is_empty() {
                "Preparing model download...".to_string()
            } else {
                format!("Preparing {filename}...")
            };
            let line = Line::from(vec![
                Span::styled(
                    format!("{} ", crate::ui::chrome::spinner_glyph(app)),
                    Style::default().fg(theme.warning),
                ),
                Span::styled(label, Style::default().fg(theme.text)),
            ]);
            frame.render_widget(Paragraph::new(line), row);
        }
        y += 1;
    }
    if rows.weight && y < inner.y + inner.height {
        let pct = if progress_state.weight_total > 0 {
            (progress_state.weight_loaded as f64 / progress_state.weight_total as f64).min(1.0)
        } else {
            0.0
        };
        let label = format!(
            "Loading {} [{}/{}]",
            progress_state.weight_component,
            progress::format_bytes(progress_state.weight_loaded),
            progress::format_bytes(progress_state.weight_total),
        );
        let gauge = Gauge::default()
            .ratio(pct)
            .label(label)
            .gauge_style(app.theme.progress_filled())
            .style(app.theme.progress_empty());
        frame.render_widget(
            gauge,
            Rect {
                x: inner.x,
                y,
                width: inner.width,
                height: 1,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeline_shows_last_n_entries_only() {
        // The tail wins — older entries scroll away.
        assert_eq!(tail_range(3, 5), 0..3);
        assert_eq!(tail_range(12, 5), 7..12);
        assert_eq!(tail_range(0, 5), 0..0);
        // The panel budget is the compile-time contract in ui::generate.
        assert_eq!(TIMELINE_VISIBLE_ROWS, 5);
    }

    #[test]
    fn timeline_glyphs_match_entry_styles() {
        assert_eq!(
            entry_glyph(PStyle::Info, "Connecting to server"),
            "\u{2022}"
        );
        assert_eq!(
            entry_glyph(PStyle::Done, "Saved mold-1.png [0.3s]"),
            "\u{2713}"
        );
        assert_eq!(entry_glyph(PStyle::Warning, "Upscale cancelled"), "!");
        assert_eq!(entry_glyph(PStyle::Error, "Generation failed"), "\u{2717}");
    }

    #[test]
    fn timeline_marks_loaded_model_lines_with_star() {
        assert_eq!(entry_glyph(PStyle::Done, "Model loaded [3.1s]"), "\u{2605}");
        // Only success lines get the star treatment.
        assert_eq!(entry_glyph(PStyle::Error, "Model loaded?"), "\u{2717}");
    }

    #[test]
    fn timeline_empty_state_message() {
        assert_eq!(EMPTY_STATE, "\u{2014} idle. no runs this session.");
    }
}
