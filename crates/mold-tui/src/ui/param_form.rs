use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};

use crate::app::{GenerateState, ParamField};
use crate::ui::theme::Theme;

/// Render the parameter form panel.
pub fn render(frame: &mut Frame, theme: &Theme, state: &GenerateState, area: Rect, focused: bool) {
    let border_style = if focused {
        theme.border_focused()
    } else {
        theme.border()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" Parameters ")
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

    // Build the lines for all visible fields
    let mut lines: Vec<Line> = Vec::new();
    let mut selected_line: Option<usize> = None;

    for (i, field) in state.visible_fields.iter().enumerate() {
        // Section header
        if let Some(header) = field.section_header() {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("\u{2500}\u{2500} {header} "),
                    Style::default()
                        .fg(theme.text_dim)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "\u{2500}".repeat(inner.width.saturating_sub(header.len() as u16 + 5) as usize),
                    Style::default().fg(theme.border),
                ),
            ]));
        }

        let is_selected = focused && i == state.param_index;
        if is_selected {
            selected_line = Some(lines.len());
        }

        let label = format!("{:<10}", field.label());
        let value = if *field == ParamField::SeedValue && state.params.seed.is_none() {
            if let Some(last) = state.last_seed {
                // In random mode, show the last seed used from generation
                format!("{last} (last)")
            } else {
                state.params.display_value(field)
            }
        } else {
            state.params.display_value(field)
        };

        let (label_style, value_style) = if is_selected {
            (theme.param_selected(), theme.param_selected())
        } else {
            (theme.param_label(), theme.param_value())
        };

        // Add indicator for dropdown/toggle fields
        let suffix = match field {
            ParamField::Model => " \u{25bc}", // down triangle
            ParamField::Format | ParamField::Mode | ParamField::Expand | ParamField::Offload => {
                if is_selected {
                    " \u{25c0}\u{25b6}"
                } else {
                    ""
                } // left/right arrows
            }
            ParamField::Width
            | ParamField::Height
            | ParamField::Steps
            | ParamField::Guidance
            | ParamField::Batch
            | ParamField::Strength
            | ParamField::ControlScale => {
                if is_selected {
                    " \u{25c0}\u{25b6}"
                } else {
                    ""
                }
            }
            _ => "",
        };

        lines.push(Line::from(vec![
            Span::styled(label, label_style),
            Span::styled(value, value_style),
            Span::styled(
                suffix,
                if is_selected {
                    theme.param_selected()
                } else {
                    theme.dim()
                },
            ),
        ]));
    }

    // Scroll to keep the selected line visible
    let scroll_offset = if let Some(sel) = selected_line {
        if sel >= inner.height as usize {
            sel.saturating_sub(inner.height as usize / 2)
        } else {
            0
        }
    } else {
        0
    };

    let paragraph = Paragraph::new(lines.clone()).scroll((scroll_offset as u16, 0));
    frame.render_widget(paragraph, inner);

    // Scrollbar if content overflows
    if lines.len() > inner.height as usize {
        let mut scrollbar_state =
            ScrollbarState::new(lines.len().saturating_sub(inner.height as usize))
                .position(scroll_offset);
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .style(Style::default().fg(theme.border));
        frame.render_stateful_widget(scrollbar, inner, &mut scrollbar_state);
    }
}
