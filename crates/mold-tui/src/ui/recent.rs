//! Recent Strip — a compact list of the most-recently generated images
//! rendered in the bottom-left of the Generate view.
//!
//! In the web design kit this is a horizontal strip of thumbnails. Terminals
//! can't reliably render multiple inline images side-by-side across all
//! graphics protocols (Kitty/Sixel/iTerm2/halfblocks), so the TUI uses a
//! vertical text strip instead — one row per entry with the filename, model
//! id, and a relative `diff` label (`latest`, `-1`, `-2`, …).

use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

use crate::app::App;

use super::widgets::panel_block;

/// Maximum number of recent entries shown in the strip.
const MAX_ENTRIES: usize = 4;

/// Render the Recent strip inside `area`.
pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;
    let entries = &app.gallery.entries;

    let count = entries.len().min(MAX_ENTRIES);
    let hint = if entries.is_empty() {
        None
    } else {
        Some(format!("{count} recent"))
    };
    let block = panel_block(theme, "Recent", false, hint.as_deref());
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    if entries.is_empty() {
        let empty = Paragraph::new("No images yet — press Enter to generate.")
            .style(theme.dim())
            .alignment(Alignment::Left);
        frame.render_widget(empty, inner);
        return;
    }

    let width = inner.width as usize;
    let lines: Vec<Line> = entries
        .iter()
        .take(count.min(inner.height as usize))
        .enumerate()
        .map(|(i, entry)| build_line(theme, entry, i, width))
        .collect();

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);
}

/// Build a single row: `● <filename>  <model>  <diff>`.
///
/// We compute widths so the model column right-aligns with a small gap to the
/// diff label. Truncation uses `.` characters to keep the layout stable on
/// narrow terminals.
fn build_line<'a>(
    theme: &crate::ui::theme::Theme,
    entry: &'a crate::app::GalleryEntry,
    index: usize,
    width: usize,
) -> Line<'a> {
    let name = entry.filename();
    let model = entry.metadata.model.clone();
    let diff = diff_label(index);

    // Reserve space: dot(2) + diff(8 max) + gaps(4) → minimum 14 chars of chrome.
    let chrome = 2 + diff.len() + 4;
    let body_width = width.saturating_sub(chrome);
    let name_width = body_width.saturating_mul(3) / 5; // roughly 60% for filename
    let model_width = body_width.saturating_sub(name_width);

    let name_cell = truncate(&name, name_width);
    let model_cell = truncate(&model, model_width);

    let mut spans = Vec::with_capacity(6);
    spans.push(Span::styled("● ", Style::default().fg(theme.accent)));
    spans.push(Span::styled(
        format!("{:<w$}", name_cell, w = name_width),
        theme.param_value(),
    ));
    spans.push(Span::raw("  "));
    spans.push(Span::styled(
        format!("{:<w$}", model_cell, w = model_width),
        theme.dim(),
    ));
    spans.push(Span::raw("  "));
    spans.push(Span::styled(diff, theme.dim()));
    Line::from(spans)
}

fn diff_label(index: usize) -> &'static str {
    match index {
        0 => "latest",
        1 => "-1",
        2 => "-2",
        3 => "-3",
        _ => "-n",
    }
}

fn truncate(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut = max.saturating_sub(1).max(1);
    let head: String = s.chars().take(cut).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_shorter_than_max_is_unchanged() {
        assert_eq!(truncate("abc", 10), "abc");
    }

    #[test]
    fn truncate_longer_than_max_ends_with_ellipsis() {
        assert_eq!(truncate("abcdefgh", 4), "abc…");
    }

    #[test]
    fn truncate_zero_is_empty() {
        assert_eq!(truncate("abc", 0), "");
    }

    #[test]
    fn diff_labels_cover_first_four() {
        assert_eq!(diff_label(0), "latest");
        assert_eq!(diff_label(1), "-1");
        assert_eq!(diff_label(3), "-3");
        assert_eq!(diff_label(5), "-n");
    }
}
