pub mod generate;
pub mod gallery;
pub mod models;
pub mod param_form;
pub mod popup;
pub mod progress;
pub mod theme;

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Tabs};

use crate::action::View;
use crate::app::App;

/// Top-level render function — draws the frame chrome and delegates to the active view.
pub fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    let theme = &app.theme;

    // Fill background
    frame.render_widget(Block::default().style(theme.base()), area);

    // Main layout: title bar + content + status bar
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Tab bar
            Constraint::Min(10),   // Content area
            Constraint::Length(1), // Status bar
        ])
        .split(area);

    // ── Tab bar ─────────────────────────────────────────────────
    render_tab_bar(frame, app, layout[0]);

    // ── Content ─────────────────────────────────────────────────
    match app.active_view {
        View::Generate => generate::render(frame, app, layout[1]),
        View::Gallery => gallery::render(frame, app, layout[1]),
        View::Models => models::render(frame, app, layout[1]),
    }

    // ── Status bar ──────────────────────────────────────────────
    render_status_bar(frame, app, layout[2]);

    // ── Popup overlay ───────────────────────────────────────────
    if app.popup.is_some() {
        popup::render(frame, app);
    }
}

fn render_tab_bar(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    let tab_titles: Vec<Line> = View::ALL
        .iter()
        .enumerate()
        .map(|(i, view)| {
            let style = if *view == app.active_view {
                theme.tab_active()
            } else {
                theme.tab_inactive()
            };
            Line::from(format!(" {} {} ", i + 1, view.label())).style(style)
        })
        .collect();

    let tabs = Tabs::new(tab_titles)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(theme.border())
                .title(" mold ")
                .title_style(
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                )
                .style(Style::default().bg(theme.tab_bg))
                .padding(Padding::horizontal(1)),
        )
        .select(app.active_view.index())
        .divider(" ")
        .highlight_style(theme.tab_active());

    frame.render_widget(tabs, area);
}

fn render_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    let shortcuts = match app.active_view {
        View::Generate => {
            if app.generate.generating {
                vec![("", "Generating...")]
            } else {
                vec![
                    ("Enter", "Generate"),
                    ("^E", "Expand"),
                    ("^S", "Save"),
                    ("^M", "Model"),
                    ("^R", "Seed"),
                    ("Tab", "Focus"),
                    ("?", "Help"),
                    ("q", "Quit"),
                ]
            }
        }
        View::Gallery => vec![
            ("Enter", "Re-gen"),
            ("e", "Edit"),
            ("d", "Delete"),
            ("o", "Open"),
            ("hjkl", "Pan"),
            ("?", "Help"),
            ("q", "Quit"),
        ],
        View::Models => vec![
            ("Enter", "Select"),
            ("p", "Pull"),
            ("r", "Remove"),
            ("u", "Unload"),
            ("/", "Filter"),
            ("?", "Help"),
            ("q", "Quit"),
        ],
    };

    let mut spans = Vec::new();
    for (i, (key, desc)) in shortcuts.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  ", theme.status_bar()));
        }
        if !key.is_empty() {
            spans.push(Span::styled(*key, theme.status_key()));
            spans.push(Span::styled(" ", theme.status_bar()));
        }
        spans.push(Span::styled(*desc, theme.status_bar()));
    }

    let bar = Paragraph::new(Line::from(spans))
        .style(theme.status_bar());
    frame.render_widget(bar, area);
}
