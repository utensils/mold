use ratatui::prelude::*;
use ratatui::widgets::{Cell, Paragraph, Row, Table, TableState};

use crate::app::App;
use crate::ui::widgets::{kv_row_line, panel_block};

/// Height reserved for the Details + Actions row at the bottom of the view.
const INSPECTOR_HEIGHT: u16 = 7;

/// Render the Models view.
pub fn render(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = &app.theme;

    // Layout: two table panes on top, inspector row (Details + Actions) below.
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),
            Constraint::Min(6),
            Constraint::Length(INSPECTOR_HEIGHT),
        ])
        .split(area);

    let (installed, available): (Vec<_>, Vec<_>) = app
        .models
        .catalog
        .iter()
        .enumerate()
        .partition(|(_, m)| m.downloaded);

    app.layout.models_table = area;

    render_model_table(
        frame,
        theme,
        "Installed",
        &installed,
        app.models.selected,
        layout[0],
        true,
    );

    let avail_offset = installed.len();
    render_model_table(
        frame,
        theme,
        "Available",
        &available,
        app.models.selected.wrapping_sub(avail_offset),
        layout[1],
        app.models.selected >= avail_offset,
    );

    render_inspector_row(frame, app, layout[2]);
}

/// Render the Details + Actions row shown beneath the Models tables.
fn render_inspector_row(frame: &mut Frame, app: &App, area: Rect) {
    let [details_area, actions_area] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .areas(area);

    render_details_panel(frame, app, details_area);
    render_actions_panel(frame, app, actions_area);
}

fn render_details_panel(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;
    let block = panel_block(theme, "Details", false, None);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let Some(model) = app.models.catalog.get(app.models.selected) else {
        let empty = Paragraph::new("no matches").style(theme.dim());
        frame.render_widget(empty, inner);
        return;
    };

    let defaults = format!(
        "{} steps · CFG {:.1}",
        model.defaults.default_steps, model.defaults.default_guidance,
    );
    let size = format!("{:.1}G", model.size_gb);
    let family = model.family.to_uppercase();

    let lines = vec![
        Line::from(Span::styled(
            model.name.clone(),
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            model.defaults.description.clone(),
            theme.dim(),
        )),
        Line::from(""),
        kv_row_line(theme, "Family", &family, 8, false),
        kv_row_line(theme, "Size", &size, 8, false),
        kv_row_line(theme, "Default", &defaults, 8, false),
        kv_row_line(theme, "HF", &model.hf_repo, 8, true),
    ];
    let para = Paragraph::new(lines);
    frame.render_widget(para, inner);
}

/// Key reference for the Models view. Matches the design system's
/// "Actions" pane — a static shortcut legend, not a clickable menu.
fn render_actions_panel(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;
    let block = panel_block(theme, "Actions", false, None);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let actions: &[(&str, &str, &str)] = &[
        ("p", "Pull", "download / update"),
        ("Enter", "Load", "load into GPU"),
        ("u", "Unload", "free GPU memory"),
        ("r", "Remove", "delete local weights"),
        ("/", "Filter", "search by name"),
    ];

    let lines: Vec<Line> = actions
        .iter()
        .map(|(key, label, hint)| {
            Line::from(vec![
                Span::styled(format!("{:<6}", key), theme.status_key()),
                Span::styled(" ", theme.param_label()),
                Span::styled(format!("{:<8}", label), theme.param_value()),
                Span::styled(*hint, theme.dim()),
            ])
        })
        .collect();

    let para = Paragraph::new(lines);
    frame.render_widget(para, inner);
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
    let hint = format!("{} models", models.len());
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
    .block(panel_block(theme, title, is_active_section, Some(&hint)))
    .row_highlight_style(theme.list_selected())
    .highlight_symbol("> ");

    let mut state = TableState::default();
    if is_active_section && !models.is_empty() {
        state.select(Some(selected_relative.min(models.len().saturating_sub(1))));
    }

    frame.render_stateful_widget(table, area, &mut state);
}
