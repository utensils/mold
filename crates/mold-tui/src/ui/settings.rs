use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};

use crate::app::{App, SettingsFieldType, SettingsRow};

/// Render the Settings view — a single scrollable list of config fields.
pub fn render(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = &app.theme;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border_focused())
        .title(" Settings ")
        .title_style(theme.title_focused())
        .style(Style::default().bg(theme.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let rows = app.build_settings_rows();
    let row_index = app.settings.row_index;

    let mut lines: Vec<Line> = Vec::new();
    let mut selected_line: Option<usize> = None;

    for (i, row) in rows.iter().enumerate() {
        match row {
            SettingsRow::SectionHeader { name } => {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("\u{2500}\u{2500} {name} "),
                        Style::default()
                            .fg(theme.text_dim)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "\u{2500}"
                            .repeat(inner.width.saturating_sub(name.len() as u16 + 5) as usize),
                        Style::default().fg(theme.border),
                    ),
                ]));
            }
            SettingsRow::Field {
                key,
                label,
                field_type,
            } => {
                let is_selected = i == row_index;
                if is_selected {
                    selected_line = Some(lines.len());
                }

                let is_read_only = matches!(field_type, SettingsFieldType::ReadOnly);
                let value = app.settings_display_value(key);
                let env_var = App::settings_env_override(key);

                let label_text = format!("{:<12}", label);

                // Choose styles based on selection and read-only state
                let (label_style, value_style) = if is_selected {
                    (theme.param_selected(), theme.param_selected())
                } else if is_read_only {
                    (theme.dim(), theme.dim())
                } else {
                    (theme.param_label(), theme.param_value())
                };

                // Determine suffix
                let suffix = if is_read_only {
                    ""
                } else {
                    match field_type {
                        SettingsFieldType::Text | SettingsFieldType::Path => {
                            if is_selected {
                                " \u{25bc}"
                            } else {
                                ""
                            }
                        }
                        SettingsFieldType::Number { .. }
                        | SettingsFieldType::Toggle { .. }
                        | SettingsFieldType::Bool => {
                            if is_selected {
                                " \u{25c0}\u{25b6}"
                            } else {
                                ""
                            }
                        }
                        _ => "",
                    }
                };

                let mut spans = vec![
                    Span::styled(label_text, label_style),
                    Span::styled(value, value_style),
                ];

                // Env override indicator
                if let Some(_var) = env_var {
                    spans.push(Span::styled(
                        " (env)",
                        if is_selected {
                            theme.param_selected()
                        } else {
                            theme.warning()
                        },
                    ));
                }

                // Suffix
                spans.push(Span::styled(
                    suffix,
                    if is_selected {
                        theme.param_selected()
                    } else {
                        theme.dim()
                    },
                ));

                lines.push(Line::from(spans));
            }
        }
    }

    // Show save error if any
    if let Some(ref err) = app.settings.save_error {
        lines.push(Line::from(Span::styled(err.as_str(), theme.error())));
    }

    // Scroll to keep selected line visible
    let scroll_offset = if let Some(sel) = selected_line {
        if sel >= inner.height as usize {
            sel.saturating_sub(inner.height as usize / 2)
        } else {
            0
        }
    } else {
        0
    };
    app.settings.scroll_offset = scroll_offset;

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
