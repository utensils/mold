//! Create Preview panel — idle hint, developing progress, finished print.
//!
//! Three states (mock-exact):
//! 1. **Idle** — a dim `◇` and "Press Enter to generate".
//! 2. **Generating** — braille spinner + `Developing… {step}/{total}`
//!    (spec §11 vocabulary) + a 22-char `█░` bar + `pct · it/s · eta`.
//!    While a model pull is in flight the existing download status text
//!    takes the stage line.
//! 3. **Done** — the print via the existing `StatefulImage` resize path
//!    (NOT the gallery grid's fixed-protocol path) + a caption row
//!    `model · seed · time · host`.

use ratatui::prelude::*;
use ratatui::widgets::Paragraph;
use ratatui_image::StatefulImage;

use crate::app::App;
use crate::ui::widgets::panel_block;

/// Width of the `█░` progress bar in cells.
const BAR_WIDTH: usize = 22;

/// Pure formatter for the generating state: the stage line, the bar, and
/// the `pct · rate · eta` line. Guards division by zero when no step has
/// completed yet or no time has elapsed.
pub(crate) fn preview_progress(
    step: usize,
    total: usize,
    elapsed_ms: u64,
) -> (String, String, String) {
    let total = total.max(1);
    let step = step.min(total);
    let line = format!("Developing\u{2026} {step}/{total}");
    let filled = (step * BAR_WIDTH / total).min(BAR_WIDTH);
    let mut bar = String::new();
    for _ in 0..filled {
        bar.push('\u{2588}');
    }
    for _ in filled..BAR_WIDTH {
        bar.push('\u{2591}');
    }
    let pct = step * 100 / total;
    let rate = if elapsed_ms > 0 && step > 0 {
        step as f64 / (elapsed_ms as f64 / 1000.0)
    } else {
        0.0
    };
    let eta_s = if step > 0 {
        ((total - step) as u64).saturating_mul(elapsed_ms) / (step as u64 * 1000)
    } else {
        0
    };
    (
        line,
        bar,
        format!("{pct}% \u{00b7} {rate:.1} it/s \u{00b7} eta {eta_s}s"),
    )
}

/// Caption under a finished print: `model · seed · time · host`.
pub(crate) fn preview_caption(model: &str, seed: u64, time_ms: u64, host_label: &str) -> String {
    format!(
        "{model} \u{00b7} seed {seed} \u{00b7} {:.1}s \u{00b7} {host_label}",
        time_ms as f64 / 1000.0
    )
}

/// Display name of the sticky generation target ("This Mac" unless a
/// registered host is pinned).
pub(crate) fn target_host_label(app: &App) -> String {
    match &app.target {
        crate::hosts::GenTarget::Host(id) => app
            .machines
            .registry
            .get(id)
            .map(|e| e.display_name())
            .unwrap_or_else(|| id.clone()),
        _ => crate::hosts::local_display_name().to_string(),
    }
}

/// Render the Preview panel.
pub fn render(frame: &mut Frame, app: &mut App, area: Rect) {
    let block = panel_block(&app.theme, "Preview", false, None);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    if app.generate.image_state.is_some() {
        render_done(frame, app, inner);
        return;
    }

    if app.generate.generating {
        render_generating(frame, app, inner);
    } else {
        render_idle(frame, app, inner);
    }
}

fn render_done(frame: &mut Frame, app: &mut App, inner: Rect) {
    // Caption row (only when we actually finished a run — an image can
    // also be present from an animation preview mid-session).
    let caption = app.generate.last_seed.map(|seed| {
        preview_caption(
            &app.generate.params.model,
            seed,
            app.generate.last_generation_time_ms.unwrap_or(0),
            &target_host_label(app),
        )
    });
    let image_area = if caption.is_some() && inner.height > 1 {
        Rect {
            height: inner.height - 1,
            ..inner
        }
    } else {
        inner
    };
    if let Some(ref mut image_state) = app.generate.image_state {
        let widget = StatefulImage::default().resize(ratatui_image::Resize::Scale(None));
        frame.render_stateful_widget(widget, image_area, image_state);
    }
    if let Some(caption) = caption {
        if inner.height > 1 {
            let row = Rect {
                y: inner.y + inner.height - 1,
                height: 1,
                ..inner
            };
            frame.render_widget(
                Paragraph::new(caption)
                    .style(app.theme.dim())
                    .alignment(Alignment::Center),
                row,
            );
        }
    }
}

fn render_generating(frame: &mut Frame, app: &App, inner: Rect) {
    let theme = &app.theme;
    let progress = &app.generate.progress;

    // The stage line: denoise progress once it starts, the download text
    // during a pull, else the current pipeline stage.
    let (stage_line, bar, stats) = if progress.denoise_total > 0 {
        let (line, bar, stats) = preview_progress(
            progress.denoise_step,
            progress.denoise_total,
            progress.denoise_elapsed_ms,
        );
        (line, Some(bar), Some(stats))
    } else if progress.is_downloading() {
        (progress.download_status_text().to_string(), None, None)
    } else if let Some(stage) = &progress.current_stage {
        (format!("{stage}\u{2026}"), None, None)
    } else {
        ("Working\u{2026}".to_string(), None, None)
    };

    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(
            crate::ui::chrome::spinner_glyph(app).to_string(),
            Style::default().fg(theme.info),
        )),
        Line::default(),
        Line::from(Span::styled(stage_line, Style::default().fg(theme.text))),
    ];
    if let Some(bar) = bar {
        lines.push(Line::from(Span::styled(
            bar,
            Style::default().fg(theme.accent),
        )));
    }
    if let Some(stats) = stats {
        lines.push(Line::from(Span::styled(stats, theme.faint())));
    }
    render_centered(frame, inner, lines);
}

fn render_idle(frame: &mut Frame, app: &App, inner: Rect) {
    let theme = &app.theme;
    let lines = vec![
        Line::from(Span::styled("\u{25c7}", theme.faint())),
        Line::default(),
        Line::from(vec![
            Span::styled("Press ", theme.dim()),
            Span::styled("Enter", Style::default().fg(theme.accent)),
            Span::styled(" to generate", theme.dim()),
        ]),
    ];
    render_centered(frame, inner, lines);
}

fn render_centered(frame: &mut Frame, inner: Rect, lines: Vec<Line>) {
    let height = (lines.len() as u16).min(inner.height);
    let rect = Rect {
        x: inner.x,
        y: inner.y + (inner.height - height) / 2,
        width: inner.width,
        height,
    };
    frame.render_widget(Paragraph::new(lines).alignment(Alignment::Center), rect);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_progress_formats_percent_rate_eta() {
        // 12/28 after 6.6s → ~1.8 it/s, eta (16 * 6600 / 12) ms ≈ 8s.
        let (line, bar, stats) = preview_progress(12, 28, 6_600);
        assert_eq!(line, "Developing\u{2026} 12/28");
        assert_eq!(bar.chars().count(), 22);
        assert_eq!(bar.chars().filter(|c| *c == '\u{2588}').count(), 9); // 12*22/28
        assert_eq!(stats, "42% \u{00b7} 1.8 it/s \u{00b7} eta 8s");
    }

    #[test]
    fn preview_progress_handles_zero_elapsed() {
        // First step callback can arrive with elapsed 0 — no div-by-zero.
        let (_, bar, stats) = preview_progress(0, 28, 0);
        assert_eq!(bar.chars().filter(|c| *c == '\u{2588}').count(), 0);
        assert_eq!(stats, "0% \u{00b7} 0.0 it/s \u{00b7} eta 0s");
        // Degenerate totals clamp instead of panicking.
        let (line, _, _) = preview_progress(5, 0, 100);
        assert_eq!(line, "Developing\u{2026} 1/1");
    }

    #[test]
    fn preview_progress_full_bar_at_completion() {
        let (_, bar, stats) = preview_progress(28, 28, 14_000);
        assert!(bar.chars().all(|c| c == '\u{2588}'));
        assert!(stats.starts_with("100%"), "{stats}");
    }

    #[test]
    fn preview_caption_joins_with_middots() {
        assert_eq!(
            preview_caption("flux2-klein:q4", 771204, 4_020, "This Mac"),
            "flux2-klein:q4 \u{00b7} seed 771204 \u{00b7} 4.0s \u{00b7} This Mac"
        );
    }
}
