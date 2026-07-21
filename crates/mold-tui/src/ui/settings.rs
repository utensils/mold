//! Settings view — two stacked panels:
//!
//! 1. **Appearance** — the theme card grid (see [`super::theme_cards`]).
//!    Focused when `SettingsFocus::Appearance`; Left/Right cycles presets
//!    linearly and Up/Down moves by grid rows, all with immediate live
//!    apply. The panel height follows the card-row count and scrolls by
//!    whole rows when the terminal is too short.
//! 2. **Configuration** — the flat list of editable settings fields
//!    (Preferences first, then the config sections). Focused when
//!    `SettingsFocus::Configuration`; j/k navigates, +/- adjusts, Enter
//!    opens text/path popups.
//!
//! Up from the top field hands focus back to Appearance; Down from the
//! Appearance grid's bottom row moves into Configuration. The config list
//! renders with a scrollbar once it exceeds the panel height.

use ratatui::prelude::*;
use ratatui::widgets::{Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState};

use crate::app::{App, SettingsFieldType, SettingsFocus, SettingsRow};

use super::theme_cards::{appearance_panel_height, card_grid, render_theme_cards};
use super::widgets::panel_block;

/// Fixed label-column width for Configuration rows — wide enough for the
/// longest Preferences label ("Reduce Motion") plus one space.
pub(crate) const LABEL_WIDTH: usize = 14;

/// Minimum rows reserved for the Configuration panel below Appearance.
const MIN_CONFIG_HEIGHT: u16 = 5;

/// Render the Settings view.
pub fn render(frame: &mut Frame, app: &mut App, area: Rect) {
    // Appearance is sized to its card rows, clipped so Configuration
    // always keeps at least MIN_CONFIG_HEIGHT rows.
    let inner_width = area.width.saturating_sub(2);
    let appearance_height = appearance_panel_height(
        inner_width,
        area.height.saturating_sub(MIN_CONFIG_HEIGHT).max(1),
    )
    .min(area.height);

    let [appearance_area, config_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(appearance_height),
            Constraint::Min(MIN_CONFIG_HEIGHT),
        ])
        .areas(area);

    render_appearance(frame, app, appearance_area);
    render_configuration(frame, app, config_area);
}

/// Top panel — the theme card grid.
fn render_appearance(frame: &mut Frame, app: &mut App, area: Rect) {
    let focused = app.settings.focus == SettingsFocus::Appearance;
    // The hint shows the canonical slug (identical to the lowercased label
    // for single-word presets) — `scripts/tui-uat.sh theme-set` parses it.
    let hint = format!("theme · {}", app.settings.theme_preset.slug());
    let block = panel_block(&app.theme, "Appearance", focused, Some(&hint));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }
    // Record the rendered column count so ↑/↓ move by exactly one visual
    // row (the same pattern gallery uses for its grid).
    let (cols, _) = card_grid(crate::ui::theme::ThemePreset::ALL.len(), inner.width);
    app.settings.appearance_cols = cols;
    render_theme_cards(frame, &app.theme, inner, app.settings.theme_preset, focused);
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
            let label_text = format!("{:<width$}", label, width = LABEL_WIDTH);

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
