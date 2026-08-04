//! Create parameters panel renderer — essentials rows + Advanced accordion.
//!
//! The row *model* (which rows exist) lives in [`crate::ui::create_form`];
//! this module turns `GenerateState.rows` into panel lines, keeps the
//! selection visible (`clamp_scroll`), and exposes the line↔row math the
//! mouse hit-test shares (`row_at_line`). Rows are not all one line tall:
//! Model carries a dim description line, the Advanced header carries its
//! dashed rule, and the inline Negative editor is a 3-row textarea.

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::app::{GenerateFocus, GenerateState, ParamField};
use crate::ui::create_form::{self, CreateRow, DETAIL_MAX, DETAIL_MIN};
use crate::ui::theme::Theme;
use crate::ui::widgets::{panel_block, truncate_with_ellipsis};

/// Height of the inline Negative editor row (bordered textarea).
pub(crate) const NEGATIVE_EDITOR_HEIGHT: u16 = 3;
/// Label column width for essentials rows ("Prompt strength" + gap).
const LABEL_W: usize = 16;

/// Panel lines one row occupies.
pub(crate) fn row_height(row: &CreateRow, has_desc: bool) -> u16 {
    match row {
        CreateRow::Field(ParamField::Model) if has_desc => 2,
        CreateRow::AdvancedHeader => 2, // dashed rule + disclosure line
        CreateRow::NegativeEditor => NEGATIVE_EDITOR_HEIGHT,
        _ => 1,
    }
}

/// Line offset of every row plus the total line count as the last entry.
pub(crate) fn row_offsets(rows: &[CreateRow], has_desc: bool) -> Vec<u16> {
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    let mut y = 0u16;
    for row in rows {
        offsets.push(y);
        y += row_height(row, has_desc);
    }
    offsets.push(y);
    offsets
}

/// Which row a panel line (0-based, scroll already applied) falls on —
/// the shared math between the renderer and mouse hit-testing.
pub(crate) fn row_at_line(rows: &[CreateRow], has_desc: bool, line: usize) -> Option<usize> {
    let mut y = 0usize;
    for (i, row) in rows.iter().enumerate() {
        y += row_height(row, has_desc) as usize;
        if line < y {
            return Some(i);
        }
    }
    None
}

/// Scroll (in lines) that keeps the selected row fully visible, moving as
/// little as possible from `current`.
pub(crate) fn clamp_scroll(offsets: &[u16], sel: usize, view_h: usize, current: usize) -> usize {
    if offsets.len() < 2 || view_h == 0 {
        return 0;
    }
    let sel = sel.min(offsets.len() - 2);
    let sel_start = offsets[sel] as usize;
    let sel_end = offsets[sel + 1] as usize;
    let total = *offsets.last().unwrap() as usize;
    let mut scroll = current.min(total.saturating_sub(view_h));
    if sel_start < scroll {
        scroll = sel_start;
    }
    if sel_end > scroll + view_h {
        scroll = sel_end.saturating_sub(view_h);
    }
    scroll
}

/// Render the parameter form panel. Returns the inline Negative editor's
/// rect when it was drawn (stored for mouse hit-testing).
pub fn render(
    frame: &mut Frame,
    theme: &Theme,
    state: &mut GenerateState,
    area: Rect,
    focused: bool,
) -> Option<Rect> {
    let block = panel_block(theme, "Parameters", focused, None);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return None;
    }

    let has_desc = !state.model_description.is_empty();
    let rows = state.rows.clone();
    if rows.is_empty() {
        return None;
    }
    let offsets = row_offsets(&rows, has_desc);
    let sel = state.param_index.min(rows.len() - 1);
    let scroll = clamp_scroll(&offsets, sel, inner.height as usize, state.param_scroll);
    state.param_scroll = scroll;

    let neg_empty = state
        .negative_prompt
        .lines()
        .iter()
        .all(|l| l.trim().is_empty());
    let active = create_form::advanced_active_count(&state.params, neg_empty);

    let mut neg_rect = None;
    for (i, row) in rows.iter().enumerate() {
        let h = row_height(row, has_desc);
        let start = offsets[i] as isize - scroll as isize;
        if start + h as isize <= 0 || start >= inner.height as isize {
            continue;
        }
        let selected = focused && i == sel;
        if *row == CreateRow::NegativeEditor {
            // The bordered textarea doesn't clip line-by-line; draw it only
            // when its top edge is on-screen and clamp the bottom.
            if start < 0 {
                continue;
            }
            let rect = Rect {
                x: inner.x,
                y: inner.y + start as u16,
                width: inner.width,
                height: h.min(inner.height - start as u16),
            };
            render_negative_editor(frame, theme, state, rect, selected);
            neg_rect = Some(rect);
            continue;
        }
        let lines = row_lines(theme, state, row, selected, active, neg_empty, inner.width);
        for (j, line) in lines.into_iter().enumerate() {
            let y = start + j as isize;
            if y < 0 || y >= inner.height as isize {
                continue;
            }
            let rect = Rect {
                x: inner.x,
                y: inner.y + y as u16,
                width: inner.width,
                height: 1,
            };
            frame.render_widget(Paragraph::new(line), rect);
        }
    }
    neg_rect
}

fn render_negative_editor(
    frame: &mut Frame,
    theme: &Theme,
    state: &mut GenerateState,
    rect: Rect,
    selected: bool,
) {
    let editing = state.focus == GenerateFocus::NegativePrompt;
    let border_style = if editing {
        theme.border_focused()
    } else if selected {
        theme.param_selected()
    } else {
        theme.border()
    };
    let ta = &mut state.negative_prompt;
    ta.set_block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .style(Style::default().bg(theme.bg)),
    );
    ta.set_style(Style::default().fg(theme.text).bg(theme.bg));
    ta.set_cursor_style(if editing {
        Style::default().fg(theme.bg).bg(theme.accent)
    } else {
        Style::default()
    });
    ta.set_placeholder_style(theme.dim());
    frame.render_widget(&*ta, rect);
}

/// Whether a field shows the `◀▶` adjust affordance when selected.
fn adjustable(field: ParamField) -> bool {
    !matches!(
        field,
        ParamField::Model
            | ParamField::Lora
            | ParamField::StgBlocks
            | ParamField::SourceImage
            | ParamField::MaskImage
            | ParamField::ControlImage
            | ParamField::ControlModel
    )
}

fn field_value(state: &GenerateState, field: ParamField) -> String {
    match field {
        // Detail: 8-dot gauge + the raw step count.
        ParamField::Steps => format!(
            "{} {}",
            create_form::detail_dots(state.params.steps, DETAIL_MIN, DETAIL_MAX),
            state.params.steps
        ),
        _ => state.params.display_value(&field),
    }
}

#[allow(clippy::too_many_arguments)]
fn row_lines(
    theme: &Theme,
    state: &GenerateState,
    row: &CreateRow,
    selected: bool,
    active_count: usize,
    neg_empty: bool,
    width: u16,
) -> Vec<Line<'static>> {
    let width = width as usize;
    match row {
        CreateRow::Field(field) | CreateRow::SectionField(_, field) => {
            let indent = if matches!(row, CreateRow::SectionField(..)) {
                "  "
            } else {
                ""
            };
            let label = format!("{indent}{:<w$}", field.label(), w = LABEL_W - indent.len());
            let value = field_value(state, *field);
            let (label_style, value_style) = if selected {
                (theme.param_selected(), theme.param_selected())
            } else {
                (theme.param_label(), theme.param_value())
            };
            let suffix = match field {
                ParamField::Model => " \u{25be}".to_string(),
                f if selected && adjustable(*f) => " \u{25c0}\u{25b6}".to_string(),
                _ => String::new(),
            };
            let mut lines = vec![Line::from(vec![
                Span::styled(label, label_style),
                Span::styled(value, value_style),
                Span::styled(
                    suffix,
                    if selected {
                        theme.param_selected()
                    } else {
                        theme.info()
                    },
                ),
            ])];
            if *field == ParamField::Model && !state.model_description.is_empty() {
                lines.push(Line::from(Span::styled(
                    truncate_with_ellipsis(&state.model_description, width),
                    theme.dim(),
                )));
            }
            lines
        }
        CreateRow::AdvancedHeader => {
            let rule = Line::from(Span::styled(
                "\u{2504}".repeat(width),
                Style::default().fg(theme.border),
            ));
            let caret = if state.advanced.open {
                "\u{25be} "
            } else {
                "\u{25b8} "
            };
            let title_style = if selected {
                theme.param_selected()
            } else {
                Style::default().fg(theme.text)
            };
            let mut spans = vec![
                Span::styled(caret.to_string(), theme.info()),
                Span::styled("Advanced".to_string(), title_style),
            ];
            let mut used = caret.chars().count() + "Advanced".len();
            if active_count > 0 {
                let badge = format!(" ({active_count})");
                used += badge.chars().count();
                spans.push(Span::styled(badge, theme.info()));
            }
            let hint = create_form::advanced_header_hint(&state.capabilities, &state.advanced);
            let hint = truncate_with_ellipsis(&hint, width.saturating_sub(used + 2));
            let pad = width.saturating_sub(used + hint.chars().count());
            spans.push(Span::raw(" ".repeat(pad)));
            spans.push(Span::styled(hint, theme.faint()));
            vec![rule, Line::from(spans)]
        }
        CreateRow::Section(sec) => {
            let expanded = state.advanced.expanded == Some(*sec);
            let caret = if expanded { " \u{25be} " } else { " \u{25b8} " };
            let label = sec.label().to_string();
            let label_style = if selected {
                theme.param_selected()
            } else {
                Style::default().fg(theme.text)
            };
            let mut spans = vec![
                Span::styled(caret.to_string(), theme.faint()),
                Span::styled(label.clone(), label_style),
            ];
            let used = caret.chars().count() + label.chars().count();
            let summary = create_form::section_summary(*sec, &state.params, neg_empty);
            let summary = truncate_with_ellipsis(&summary, width.saturating_sub(used + 2));
            let pad = width.saturating_sub(used + summary.chars().count());
            spans.push(Span::raw(" ".repeat(pad)));
            spans.push(Span::styled(summary, theme.faint()));
            vec![Line::from(spans)]
        }
        CreateRow::ResetDefaults => {
            let style = if selected {
                theme.param_selected()
            } else {
                theme.dim()
            };
            vec![Line::from(Span::styled(
                "\u{21ba} Reset to model defaults".to_string(),
                style,
            ))]
        }
        // Drawn by `render_negative_editor`.
        CreateRow::NegativeEditor => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::create_form::AdvSection;

    fn rows_fixture() -> Vec<CreateRow> {
        vec![
            CreateRow::Field(ParamField::Model),
            CreateRow::Field(ParamField::Size),
            CreateRow::AdvancedHeader,
            CreateRow::Section(AdvSection::Negative),
            CreateRow::NegativeEditor,
            CreateRow::ResetDefaults,
        ]
    }

    #[test]
    fn row_offsets_account_for_multiline_rows() {
        // Model(2 with desc) + Size(1) + Header(2) + Section(1) + Editor(3) + Reset(1)
        let offsets = row_offsets(&rows_fixture(), true);
        assert_eq!(offsets, vec![0, 2, 3, 5, 6, 9, 10]);
        // Without a model description the Model row is one line.
        let offsets = row_offsets(&rows_fixture(), false);
        assert_eq!(offsets, vec![0, 1, 2, 4, 5, 8, 9]);
    }

    #[test]
    fn row_at_line_maps_description_and_editor_lines_to_their_row() {
        let rows = rows_fixture();
        // Line 1 is the Model description — still the Model row.
        assert_eq!(row_at_line(&rows, true, 0), Some(0));
        assert_eq!(row_at_line(&rows, true, 1), Some(0));
        assert_eq!(row_at_line(&rows, true, 2), Some(1));
        // Lines 6..9 are the 3-row Negative editor.
        assert_eq!(row_at_line(&rows, true, 6), Some(4));
        assert_eq!(row_at_line(&rows, true, 8), Some(4));
        assert_eq!(row_at_line(&rows, true, 9), Some(5));
        // Past the end → no row.
        assert_eq!(row_at_line(&rows, true, 10), None);
    }

    #[test]
    fn param_scroll_keeps_selected_row_visible() {
        let rows = rows_fixture();
        let offsets = row_offsets(&rows, true); // total 10 lines
                                                // Selecting the last row in a 4-line viewport scrolls down…
        let scroll = clamp_scroll(&offsets, 5, 4, 0);
        assert_eq!(scroll, 6, "last row (lines 9..10) must be inside 6..10");
        // …and selecting the first row scrolls back up.
        let scroll = clamp_scroll(&offsets, 0, 4, scroll);
        assert_eq!(scroll, 0);
        // A selection already visible doesn't move the viewport.
        assert_eq!(clamp_scroll(&offsets, 1, 4, 0), 0);
        // Degenerate viewport never panics.
        assert_eq!(clamp_scroll(&offsets, 3, 0, 0), 0);
    }
}
