//! Shared UI widgets — thin helpers over `ratatui` primitives that match the
//! mold design system (inset panel titles, single-line borders, focus colour,
//! key/value rows, swatch grids).
//!
//! These are pure functions that build widgets for a caller to render with
//! `frame.render_widget(…, area)`. They have no state of their own.

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use super::theme::{Theme, ThemePreset};

/// A standard framed panel with a `  Title  ` inset in the top-left.
///
/// Focus is signalled by border and title colour only — no second border, no
/// glow. Pass `hint` to place dim right-aligned text in the title strip (e.g.
/// counts, filters).
pub fn panel_block<'a>(
    theme: &Theme,
    title: &'a str,
    focused: bool,
    hint: Option<&'a str>,
) -> Block<'a> {
    let (border_style, title_style) = if focused {
        (theme.border_focused(), theme.title_focused())
    } else {
        (theme.border(), theme.title())
    };

    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(format!(" {title} "))
        .title_style(title_style)
        .style(Style::default().bg(theme.bg));

    if let Some(hint) = hint {
        block = block
            .title_top(Line::from(Span::styled(format!(" {hint} "), theme.dim())).right_aligned());
    }
    block
}

/// Render a `label<pad>value` row, matching the design system's KV row style.
///
/// Labels are left-aligned in a fixed-width column so values line up across
/// adjacent rows. `muted` drops the value to `text_dim` for advisory content.
pub fn kv_row_line<'a>(
    theme: &Theme,
    label: &'a str,
    value: &'a str,
    label_width: usize,
    muted: bool,
) -> Line<'a> {
    let label_text = format!("{:<w$}", label, w = label_width);
    let value_style = if muted {
        theme.dim()
    } else {
        theme.param_value()
    };
    Line::from(vec![
        Span::styled(label_text, theme.param_label()),
        Span::styled(value, value_style),
    ])
}

/// Truncate `s` to at most `max` characters, appending `…` when the string
/// was cut. Counts Unicode scalars (via `.chars()`), not bytes, so it is safe
/// on multi-byte characters. Returns an empty string when `max == 0`, and a
/// bare `…` when `max == 1` and truncation is needed — so the returned char
/// count never exceeds `max`.
pub fn truncate_with_ellipsis(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    if max == 1 {
        return "…".to_string();
    }
    let cut = max - 1;
    let head: String = s.chars().take(cut).collect();
    format!("{head}…")
}

/// Draw the theme swatch grid used by the Appearance panel.
///
/// Each swatch is rendered as `●<space>Label` plus a trailing `✓` on the
/// current selection. The row renders on a single `Line` with no wrap —
/// terminals narrower than the combined label width clip the overflow. The
/// Appearance panel is always drawn wide enough for seven swatches on any
/// realistic TUI width, so wrapping isn't needed.
pub fn render_theme_swatches(
    frame: &mut Frame,
    theme: &Theme,
    area: Rect,
    current: ThemePreset,
    focused: bool,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let mut spans: Vec<Span> = Vec::new();
    for (i, preset) in ThemePreset::ALL.iter().enumerate() {
        let is_active = *preset == current;
        let dot_style = Style::default().fg(preset.swatch());
        let label_style = if is_active {
            if focused {
                theme.param_selected().add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme.text).add_modifier(Modifier::BOLD)
            }
        } else {
            theme.dim()
        };

        if i > 0 {
            spans.push(Span::raw("   "));
        }
        spans.push(Span::styled("● ", dot_style));
        spans.push(Span::styled(preset.label(), label_style));
        if is_active {
            spans.push(Span::styled(" ✓", theme.success()));
        }
    }

    let para = Paragraph::new(Line::from(spans));
    frame.render_widget(para, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn panel_block_focused_and_unfocused_render_without_panicking() {
        let theme = Theme::mocha();
        let area = Rect::new(0, 0, 20, 3);
        let backend = TestBackend::new(20, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                let block = panel_block(&theme, "Title", true, Some("hint"));
                frame.render_widget(block, area);
            })
            .unwrap();
        terminal
            .draw(|frame| {
                let block = panel_block(&theme, "Title", false, None);
                frame.render_widget(block, area);
            })
            .unwrap();
    }

    #[test]
    fn kv_row_line_padding_left_aligns_label() {
        let theme = Theme::mocha();
        let line = kv_row_line(&theme, "k", "v", 8, false);
        // Two spans: label + value. Label is left-padded to 8 chars so the
        // value span starts at visual column 8.
        let spans: Vec<&str> = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], "k       ");
        assert_eq!(spans[1], "v");
    }

    #[test]
    fn kv_row_line_muted_does_not_change_label_width() {
        let theme = Theme::mocha();
        let plain = kv_row_line(&theme, "key", "value", 5, false);
        let muted = kv_row_line(&theme, "key", "value", 5, true);
        assert_eq!(plain.spans[0].content, muted.spans[0].content);
    }

    #[test]
    fn truncate_with_ellipsis_shorter_than_max_is_unchanged() {
        assert_eq!(truncate_with_ellipsis("abc", 10), "abc");
    }

    #[test]
    fn truncate_with_ellipsis_longer_than_max_ends_with_ellipsis() {
        assert_eq!(truncate_with_ellipsis("abcdefgh", 4), "abc…");
    }

    #[test]
    fn truncate_with_ellipsis_zero_is_empty() {
        assert_eq!(truncate_with_ellipsis("abc", 0), "");
    }

    #[test]
    fn truncate_with_ellipsis_max_one_returns_bare_ellipsis() {
        // Regression: the previous impl did `cut = max(saturating_sub(1), 1)`
        // so `max=1` produced a 2-char string like "a…", violating the
        // "at most `max` characters" contract and visually overflowing
        // 1-column layouts.
        assert_eq!(truncate_with_ellipsis("abc", 1), "…");
        assert_eq!(truncate_with_ellipsis("abc", 1).chars().count(), 1);
        // Strings shorter than `max` are still passed through unchanged.
        assert_eq!(truncate_with_ellipsis("a", 1), "a");
    }

    #[test]
    fn render_theme_swatches_is_safe_with_zero_area() {
        // Sanity check for the early-return guard on width==0 / height==0.
        let theme = Theme::mocha();
        let backend = TestBackend::new(40, 1);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                let empty = Rect::new(0, 0, 0, 0);
                render_theme_swatches(frame, &theme, empty, ThemePreset::Latte, true);
            })
            .unwrap();
    }
}
