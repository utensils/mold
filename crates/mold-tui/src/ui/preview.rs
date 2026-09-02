//! Create Preview panel — idle hint, developing progress, finished print.
//!
//! Three states (mock-exact):
//! 1. **Idle** — a dim `◇` and "Press Enter to generate".
//! 2. **Generating** — the latest centered fixed-protocol latent preview plus
//!    `Developing… {step}/{total} · pct` when available; otherwise a braille
//!    spinner + 22-char `█░` bar + `pct · it/s · eta`. While a model pull is
//!    in flight the existing download status text takes the stage line.
//! 3. **Done** — the print via the existing `StatefulImage` resize path
//!    (NOT the gallery grid's fixed-protocol path) + a caption row
//!    `model · seed · time · host`.

use ratatui::prelude::*;
use ratatui::widgets::Paragraph;
use ratatui_image::{Image, Resize, StatefulImage};

use crate::app::App;
use crate::ui::widgets::panel_block;

/// Width of the `█░` progress bar in cells.
const BAR_WIDTH: usize = 22;
/// A live image always keeps one text row for denoise position. This makes
/// progress observable even when a terminal graphics protocol is unavailable.
const LIVE_PREVIEW_STATUS_ROWS: u16 = 1;

fn live_preview_areas(inner: Rect) -> (Rect, Rect) {
    let status_height = LIVE_PREVIEW_STATUS_ROWS.min(inner.height);
    let image_height = inner.height.saturating_sub(status_height);
    (
        Rect {
            height: image_height,
            ..inner
        },
        Rect {
            y: inner.y + image_height,
            height: status_height,
            ..inner
        },
    )
}

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

/// `49152` → `49,152`. Counts of triangles and vertices run to six or seven
/// digits, where an unseparated number stops being readable at a glance.
/// The grouping is the shared `mold_core::format::group_thousands`, so the
/// caption reads exactly as the CLI and Discord print the same counts.
pub(crate) use mold_core::format::group_thousands as format_thousands;

/// The one-line statistics of a finished 3-D print:
/// `49,152 tris · 24,576 verts · 1.00×0.80×0.60`.
///
/// The extent is the axis-aligned box the mesh occupies in its own
/// coordinate space (`MeshData.bounds_*`), carried on the wire precisely so
/// no client has to parse the geometry to describe it.
pub(crate) fn mesh_summary(
    vertices: u32,
    faces: u32,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
) -> String {
    let extent = |axis: usize| (bounds_max[axis] - bounds_min[axis]).abs();
    format!(
        "{} tris \u{00b7} {} verts \u{00b7} {:.2}\u{00d7}{:.2}\u{00d7}{:.2}",
        format_thousands(faces),
        format_thousands(vertices),
        extent(0),
        extent(1),
        extent(2)
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

    if app.generate.generating && app.generate.live_preview_image.is_some() {
        render_live_preview(frame, app, inner);
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

fn render_live_preview(frame: &mut Frame, app: &mut App, inner: Rect) {
    let (image_area, status_area) = live_preview_areas(inner);
    let cache_valid = app
        .generate
        .live_preview_protocol
        .as_ref()
        .is_some_and(|(w, h, _)| *w == image_area.width && *h == image_area.height);

    if !cache_valid {
        app.generate.live_preview_protocol = None;
        if image_area.width > 0 && image_area.height > 0 {
            if let Some(image) = app.generate.live_preview_image.clone() {
                if let Ok(protocol) =
                    app.picker
                        .new_protocol(image, image_area, Resize::Scale(None))
                {
                    app.generate.live_preview_protocol =
                        Some((image_area.width, image_area.height, protocol));
                }
            }
        }
    }

    if let Some((_, _, ref mut protocol)) = app.generate.live_preview_protocol {
        let fitted = protocol.area();
        let centered = crate::ui::gallery::center_rect(image_area, fitted.width, fitted.height);
        frame.render_widget(Image::new(protocol), centered);
    } else {
        render_generating(frame, app, image_area);
    }

    if status_area.height > 0 {
        let progress = &app.generate.progress;
        let total = progress.denoise_total.max(1);
        let step = progress.denoise_step.min(total);
        let status = format!(
            "Developing\u{2026} {step}/{total} \u{00b7} {}%",
            step * 100 / total
        );
        frame.render_widget(
            Paragraph::new(status)
                .style(app.theme.dim())
                .alignment(Alignment::Center),
            status_area,
        );
    }
}

/// Where the finished print and its caption rows go inside the panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DoneAreas {
    image: Rect,
    /// The `tris · verts · extent` row of a 3-D print, above the caption.
    summary: Option<Rect>,
    /// The `model · seed · time · host` row on the last line.
    caption: Option<Rect>,
}

/// Split `inner` into the image and up to two one-line caption rows at the
/// bottom. The caption needs a second line to exist at all (a one-row panel
/// is all image); the mesh summary needs a third, and only ever sits above
/// a caption. The image keeps every remaining row, so the rows rendered
/// never exceed the panel and never overlap the picture.
fn done_areas(inner: Rect, has_caption: bool, has_summary: bool) -> DoneAreas {
    let caption = has_caption && inner.height > 1;
    let summary = caption && has_summary && inner.height > 2;
    let rows = u16::from(caption) + u16::from(summary);
    let row_from_bottom = |offset: u16| Rect {
        y: inner.y + inner.height - offset,
        height: 1,
        ..inner
    };
    DoneAreas {
        image: Rect {
            height: inner.height - rows,
            ..inner
        },
        summary: summary.then(|| row_from_bottom(2)),
        caption: caption.then(|| row_from_bottom(1)),
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
    // A finished 3-D print gets a second caption row with its statistics
    // (`tris · verts · bounds`) above the usual one, when there is room.
    let areas = done_areas(
        inner,
        caption.is_some(),
        app.generate.last_mesh_summary.is_some(),
    );
    if let Some(ref mut image_state) = app.generate.image_state {
        let widget = StatefulImage::default().resize(ratatui_image::Resize::Scale(None));
        frame.render_stateful_widget(widget, areas.image, image_state);
    }
    if let (Some(row), Some(summary)) = (areas.summary, app.generate.last_mesh_summary.as_deref()) {
        frame.render_widget(
            Paragraph::new(summary)
                .style(app.theme.dim())
                .alignment(Alignment::Center),
            row,
        );
    }
    if let (Some(row), Some(caption)) = (areas.caption, caption) {
        frame.render_widget(
            Paragraph::new(caption)
                .style(app.theme.dim())
                .alignment(Alignment::Center),
            row,
        );
    }
}

fn render_generating(frame: &mut Frame, app: &App, inner: Rect) {
    let theme = &app.theme;
    let progress = &app.generate.progress;

    // Keep nested pipeline work (for example MiniMax H3's streamed blocks)
    // attached to the truthful denoise counter. Once N/N is complete, switch
    // to the actual final-output stage instead of leaving a stale full bar.
    let denoising = progress.denoise_total > 0 && progress.denoise_step < progress.denoise_total;
    let (stage_line, bar, stats) = if denoising {
        let (mut line, bar, stats) = preview_progress(
            progress.denoise_step,
            progress.denoise_total,
            progress.denoise_elapsed_ms,
        );
        if let Some(stage) = &progress.current_stage {
            line.push_str(" · ");
            line.push_str(stage);
        }
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
    fn live_preview_reserves_one_status_row_without_squeezing_the_image() {
        let inner = Rect::new(4, 7, 48, 8);
        let (image, status) = live_preview_areas(inner);
        assert_eq!(image, Rect::new(4, 7, 48, 7));
        assert_eq!(status, Rect::new(4, 14, 48, 1));
    }

    /// The finished-print layout: the caption rows are carved off the
    /// bottom of the panel, never exceed it, and never overlap the image.
    #[test]
    fn done_areas_fit_the_caption_rows_inside_the_panel() {
        let inner = Rect::new(4, 7, 48, 8);
        let both = done_areas(inner, true, true);
        assert_eq!(both.image, Rect::new(4, 7, 48, 6));
        assert_eq!(both.summary, Some(Rect::new(4, 13, 48, 1)));
        assert_eq!(both.caption, Some(Rect::new(4, 14, 48, 1)));
        let rows = u16::from(both.summary.is_some()) + u16::from(both.caption.is_some());
        assert_eq!(both.image.height + rows, inner.height);
        assert_eq!(both.image.bottom(), both.summary.unwrap().y);
        assert_eq!(both.summary.unwrap().bottom(), both.caption.unwrap().y);
        assert_eq!(both.caption.unwrap().bottom(), inner.bottom());

        let caption_only = done_areas(inner, true, false);
        assert_eq!(caption_only.image, Rect::new(4, 7, 48, 7));
        assert_eq!(caption_only.summary, None);
        assert_eq!(caption_only.caption, Some(Rect::new(4, 14, 48, 1)));

        // No caption, no rows: an in-flight animation preview is all image.
        let none = done_areas(inner, false, true);
        assert_eq!(none.image, inner);
        assert_eq!(none.summary, None);
        assert_eq!(none.caption, None);
    }

    /// Short panels drop the summary first, then the caption, so a row is
    /// never drawn over the picture and never outside the panel.
    #[test]
    fn done_areas_drop_rows_before_squeezing_the_image_to_nothing() {
        let three = done_areas(Rect::new(0, 0, 40, 3), true, true);
        assert_eq!(three.image.height, 1);
        assert!(three.summary.is_some() && three.caption.is_some());

        let two = done_areas(Rect::new(0, 0, 40, 2), true, true);
        assert_eq!(two.image.height, 1);
        assert_eq!(two.summary, None, "no room for a second caption row");
        assert_eq!(two.caption, Some(Rect::new(0, 1, 40, 1)));

        let one = done_areas(Rect::new(0, 0, 40, 1), true, true);
        assert_eq!(one.image, Rect::new(0, 0, 40, 1));
        assert_eq!(one.summary, None);
        assert_eq!(one.caption, None);

        let zero = done_areas(Rect::new(0, 0, 40, 0), true, true);
        assert_eq!(zero.image.height, 0);
        assert!(zero.summary.is_none() && zero.caption.is_none());
    }

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
    fn mesh_summary_reads_tris_verts_and_extent() {
        assert_eq!(
            mesh_summary(24_576, 49_152, [-0.5, -0.4, -0.3], [0.5, 0.4, 0.3]),
            "49,152 tris \u{00b7} 24,576 verts \u{00b7} 1.00\u{00d7}0.80\u{00d7}0.60"
        );
        // Small counts carry no separator; a degenerate box reads as zeros
        // rather than as a panic.
        assert_eq!(
            mesh_summary(8, 12, [0.0; 3], [0.0; 3]),
            "12 tris \u{00b7} 8 verts \u{00b7} 0.00\u{00d7}0.00\u{00d7}0.00"
        );
    }

    /// Pinned on the shared formatter the caption now uses.
    #[test]
    fn format_thousands_groups_every_three_digits() {
        assert_eq!(format_thousands(0), "0");
        assert_eq!(format_thousands(999), "999");
        assert_eq!(format_thousands(1_000), "1,000");
        assert_eq!(format_thousands(2_000_000), "2,000,000");
    }

    #[test]
    fn preview_caption_joins_with_middots() {
        assert_eq!(
            preview_caption("flux2-klein:q4", 771204, 4_020, "This Mac"),
            "flux2-klein:q4 \u{00b7} seed 771204 \u{00b7} 4.0s \u{00b7} This Mac"
        );
    }
}
