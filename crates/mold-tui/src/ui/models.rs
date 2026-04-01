use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState};

use crate::app::App;

/// Render the Models view.
pub fn render(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = &app.theme;

    // Split into: installed table, available table, details
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(40),
            Constraint::Length(4),
        ])
        .split(area);

    let (installed, available): (Vec<_>, Vec<_>) = app
        .models
        .catalog
        .iter()
        .enumerate()
        .partition(|(_, m)| m.downloaded);

    // ── Installed models ───────────────────────────────────
    render_model_table(
        frame,
        theme,
        " Installed ",
        &installed,
        app.models.selected,
        layout[0],
        true,
    );

    // ── Available models ───────────────────────────────────
    let avail_offset = installed.len();
    render_model_table(
        frame,
        theme,
        " Available ",
        &available,
        app.models.selected.wrapping_sub(avail_offset),
        layout[1],
        app.models.selected >= avail_offset,
    );

    // ── Details ────────────────────────────────────────────
    let details_block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border())
        .title(" Details ")
        .title_style(theme.title())
        .style(Style::default().bg(theme.bg));

    if let Some(model) = app.models.catalog.get(app.models.selected) {
        let line1 = format!("{} \u{2014} {}", model.name, model.defaults.description);
        let line2 = format!(
            "Family: {} \u{00b7} Size: {:.1}G \u{00b7} HF: {}",
            model.family, model.size_gb, model.hf_repo,
        );
        let details = Paragraph::new(vec![
            Line::from(Span::styled(line1, Style::default().fg(theme.text))),
            Line::from(Span::styled(line2, theme.dim())),
        ])
        .block(details_block);
        frame.render_widget(details, layout[2]);
    } else {
        frame.render_widget(Paragraph::new("").block(details_block), layout[2]);
    }
}

fn render_model_table(
    frame: &mut Frame,
    theme: &crate::ui::theme::Theme,
    title: &str,
    models: &[(usize, &mold_core::ModelInfoExtended)],
    selected_relative: usize,
    area: Rect,
    is_active_section: bool,
) {
    let border_style = if is_active_section {
        theme.border_focused()
    } else {
        theme.border()
    };

    let header = Row::new(vec![
        Cell::from("NAME").style(
            Style::default()
                .fg(theme.text_dim)
                .add_modifier(Modifier::BOLD),
        ),
        Cell::from("FAMILY").style(
            Style::default()
                .fg(theme.text_dim)
                .add_modifier(Modifier::BOLD),
        ),
        Cell::from("SIZE").style(
            Style::default()
                .fg(theme.text_dim)
                .add_modifier(Modifier::BOLD),
        ),
        Cell::from("STEPS").style(
            Style::default()
                .fg(theme.text_dim)
                .add_modifier(Modifier::BOLD),
        ),
        Cell::from("GUIDE").style(
            Style::default()
                .fg(theme.text_dim)
                .add_modifier(Modifier::BOLD),
        ),
        Cell::from("DIM").style(
            Style::default()
                .fg(theme.text_dim)
                .add_modifier(Modifier::BOLD),
        ),
        Cell::from("STATUS").style(
            Style::default()
                .fg(theme.text_dim)
                .add_modifier(Modifier::BOLD),
        ),
    ])
    .height(1);

    let rows: Vec<Row> = models
        .iter()
        .map(|(_, m)| {
            let status = if m.is_loaded {
                "loaded"
            } else if m.downloaded {
                "ready"
            } else {
                ""
            };
            let marker = if m.is_loaded { "\u{2605} " } else { "  " };
            let dim = format!("{}\u{00b2}", m.defaults.default_width,);
            Row::new(vec![
                Cell::from(format!("{marker}{}", m.name)),
                Cell::from(m.family.to_uppercase()),
                Cell::from(format!("{:.1}G", m.size_gb)),
                Cell::from(m.defaults.default_steps.to_string()),
                Cell::from(format!("{:.1}", m.defaults.default_guidance)),
                Cell::from(dim),
                Cell::from(status),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Min(22),
            Constraint::Length(8),
            Constraint::Length(6),
            Constraint::Length(5),
            Constraint::Length(5),
            Constraint::Length(7),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .title(title)
            .title_style(if is_active_section {
                theme.title_focused()
            } else {
                theme.title()
            })
            .style(Style::default().bg(theme.bg)),
    )
    .row_highlight_style(theme.list_selected())
    .highlight_symbol("> ");

    let mut state = TableState::default();
    if is_active_section && !models.is_empty() {
        state.select(Some(selected_relative.min(models.len().saturating_sub(1))));
    }

    frame.render_stateful_widget(table, area, &mut state);
}
