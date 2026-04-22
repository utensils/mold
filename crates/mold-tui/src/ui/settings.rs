//! Settings view — two stacked panels:
//!
//! 1. **Appearance** — a row of theme swatches. Focused when `SettingsFocus::Appearance`;
//!    Left/Right cycles presets with immediate live apply.
//! 2. **Configuration** — the flat list of editable settings fields. Focused when
//!    `SettingsFocus::Configuration`; j/k navigates, +/- adjusts, Enter opens
//!    text/path popups.
//!
//! Up from the top field hands focus back to Appearance; Down from Appearance
//! moves into Configuration. The config list renders with a scrollbar once it
//! exceeds the panel height.

use ratatui::prelude::*;
use ratatui::widgets::{Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState};

use crate::app::{App, SettingsFieldType, SettingsFocus, SettingsRow};

use super::widgets::{panel_block, render_theme_swatches};

/// Height of the Appearance panel (title + 1 content row + 2 borders).
const APPEARANCE_HEIGHT: u16 = 4;

/// Render the Settings view.
pub fn render(frame: &mut Frame, app: &mut App, area: Rect) {
    let [appearance_area, config_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(APPEARANCE_HEIGHT), Constraint::Min(5)])
        .areas(area);

    render_appearance(frame, app, appearance_area);
    render_configuration(frame, app, config_area);
}

/// Top panel — the theme swatch grid.
fn render_appearance(frame: &mut Frame, app: &App, area: Rect) {
    let focused = app.settings.focus == SettingsFocus::Appearance;
    let hint = format!(
        "theme · {}",
        app.settings.theme_preset.label().to_lowercase()
    );
    let block = panel_block(&app.theme, "Appearance", focused, Some(&hint));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }
    render_theme_swatches(frame, &app.theme, inner, app.settings.theme_preset, focused);
}

/// Bottom panel — the scrollable list of editable fields.
fn render_configuration(frame: &mut Frame, app: &mut App, area: Rect) {
    let focused = app.settings.focus == SettingsFocus::Configuration;
    let block = panel_block(&app.theme, "Configuration", focused, None);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let rows = app.build_settings_rows();
    let row_index = app.settings.row_index;
    let mut lines: Vec<Line> = Vec::with_capacity(rows.len());
    let mut selected_line: Option<usize> = None;

    for (i, row) in rows.iter().enumerate() {
        let is_selected = focused && i == row_index;
        if is_selected {
            selected_line = Some(lines.len());
        }
        lines.push(build_settings_line(app, row, is_selected, inner.width));
    }

    if let Some(err) = &app.settings.save_error {
        lines.push(Line::from(Span::styled(err.as_str(), app.theme.error())));
    }

    // Keep the selected line on-screen by choosing a scroll offset.
    let scroll_offset = selected_line
        .filter(|s| *s >= inner.height as usize)
        .map(|s| s.saturating_sub(inner.height as usize / 2))
        .unwrap_or(0);
    app.settings.scroll_offset = scroll_offset;

    let paragraph = Paragraph::new(lines.clone()).scroll((scroll_offset as u16, 0));
    frame.render_widget(paragraph, inner);

    // Scrollbar — only when content overflows.
    if lines.len() > inner.height as usize {
        let mut state = ScrollbarState::new(lines.len().saturating_sub(inner.height as usize))
            .position(scroll_offset);
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .style(Style::default().fg(app.theme.border));
        frame.render_stateful_widget(scrollbar, inner, &mut state);
    }
}

/// Build a single rendered line for the Configuration list.
fn build_settings_line<'a>(
    app: &App,
    row: &'a SettingsRow,
    is_selected: bool,
    width: u16,
) -> Line<'a> {
    match row {
        SettingsRow::SectionHeader { name } => {
            let rule_width = (width as usize).saturating_sub(name.chars().count() + 5);
            Line::from(vec![
                Span::styled(
                    format!("\u{2500}\u{2500} {name} "),
                    Style::default()
                        .fg(app.theme.text_dim)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "\u{2500}".repeat(rule_width),
                    Style::default().fg(app.theme.border),
                ),
            ])
        }
        SettingsRow::Field {
            key,
            label,
            field_type,
        } => {
            let is_read_only = matches!(field_type, SettingsFieldType::ReadOnly);
            let value = app.settings_display_value(key);
            let env_var = App::settings_env_override(key);
            let label_text = format!("{:<12}", label);

            let (label_style, value_style) = if is_selected {
                (app.theme.param_selected(), app.theme.param_selected())
            } else if is_read_only {
                (app.theme.dim(), app.theme.dim())
            } else {
                (app.theme.param_label(), app.theme.param_value())
            };

            let suffix = if is_read_only {
                ""
            } else {
                match field_type {
                    SettingsFieldType::Text | SettingsFieldType::Path if is_selected => " \u{25bc}",
                    SettingsFieldType::Number { .. }
                    | SettingsFieldType::Toggle { .. }
                    | SettingsFieldType::Bool
                        if is_selected =>
                    {
                        " \u{25c0}\u{25b6}"
                    }
                    _ => "",
                }
            };

            let mut spans = vec![
                Span::styled(label_text, label_style),
                Span::styled(value, value_style),
            ];

            if env_var.is_some() {
                spans.push(Span::styled(
                    " (env)",
                    if is_selected {
                        app.theme.param_selected()
                    } else {
                        app.theme.warning()
                    },
                ));
            }

            spans.push(Span::styled(
                suffix,
                if is_selected {
                    app.theme.param_selected()
                } else {
                    app.theme.dim()
                },
            ));

            Line::from(spans)
        }
    }
}
