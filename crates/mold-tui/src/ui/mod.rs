pub mod gallery;
pub mod generate;
pub mod info;
pub mod models;
pub mod param_form;
pub mod popup;
pub mod progress;
pub mod settings;
pub mod theme;

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Gauge, Padding, Paragraph, Tabs};

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
            Constraint::Length(3), // Tab bar
            Constraint::Min(10),   // Content area
            Constraint::Length(1), // Status bar
        ])
        .split(area);

    // Store layout areas for mouse hit-testing
    app.layout.tab_bar = layout[0];

    // ── Tab bar ─────────────────────────────────────────────────
    render_tab_bar(frame, app, layout[0]);

    // ── Content ─────────────────────────────────────────────────
    match app.active_view {
        View::Generate => generate::render(frame, app, layout[1]),
        View::Gallery => {
            gallery::render(frame, app, layout[1]);
            // Upscale progress bar overlay at bottom of gallery area
            if app.upscale_in_progress {
                render_upscale_progress(frame, app, layout[1]);
            }
        }
        View::Models => models::render(frame, app, layout[1]),
        View::Settings => settings::render(frame, app, layout[1]),
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

    let version = format!("mold {} ", mold_core::build_info::version_string());
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
                .title_top(
                    Line::from(Span::styled(version, Style::default().fg(theme.text_dim)))
                        .right_aligned(),
                )
                .style(Style::default().bg(theme.tab_bg))
                .padding(Padding::horizontal(1)),
        )
        .select(app.active_view.index())
        .divider(" ")
        .highlight_style(theme.tab_active());

    frame.render_widget(tabs, area);
}

/// Render an upscale progress bar at the bottom of the gallery area.
fn render_upscale_progress(frame: &mut Frame, app: &App, gallery_area: Rect) {
    let theme = &app.theme;

    // 3-row overlay at the bottom of the gallery area
    let bar_height = 3u16;
    if gallery_area.height < bar_height + 2 {
        return;
    }
    let area = Rect {
        x: gallery_area.x,
        y: gallery_area.y + gallery_area.height - bar_height,
        width: gallery_area.width,
        height: bar_height,
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border_focused())
        .title(" Upscaling ")
        .title_style(theme.title_focused())
        .style(Style::default().bg(theme.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let (tile, total) = app.upscale_tile_progress.unwrap_or((0, 0));
    let (pct, label) = if total > 0 {
        let p = tile as f64 / total as f64;
        (p, format!("Tile {tile}/{total}"))
    } else {
        (0.0, "Loading model...".to_string())
    };

    let gauge = Gauge::default()
        .ratio(pct.min(1.0))
        .label(label)
        .gauge_style(theme.progress_filled())
        .style(theme.progress_empty());

    frame.render_widget(gauge, inner);
}

fn render_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Pre-compute upscale status text so its lifetime covers the shortcut vec.
    let upscale_status = if app.upscale_in_progress {
        if let Some((tile, total)) = app.upscale_tile_progress {
            format!("Upscaling tile {tile}/{total}...")
        } else {
            "Upscaling...".to_string()
        }
    } else {
        String::new()
    };

    let shortcuts = match app.active_view {
        View::Generate => {
            if app.generate.generating {
                let status = if app.generate.progress.is_downloading() {
                    app.generate.progress.download_status_text()
                } else {
                    "Generating..."
                };
                vec![("", status)]
            } else if app.generate.focus == crate::app::GenerateFocus::Navigation {
                vec![
                    ("1-4", "Views"),
                    ("Alt+\u{2190}\u{2192}", "Views"),
                    ("Enter", "Edit"),
                    ("?", "Help"),
                    ("q", "Quit"),
                ]
            } else if app.generate.focus == crate::app::GenerateFocus::Parameters {
                vec![
                    ("Enter", "Edit"),
                    ("+/-", "Adjust"),
                    ("^G", "Generate"),
                    ("^M", "Model"),
                    ("Tab", "Focus"),
                    ("Esc", "Nav"),
                    ("?", "Help"),
                ]
            } else {
                vec![
                    ("Enter", "Generate"),
                    ("^G", "Generate"),
                    ("^M", "Model"),
                    ("^R", "Seed"),
                    ("Tab", "Focus"),
                    ("Esc", "Nav"),
                ]
            }
        }
        View::Gallery => {
            if app.upscale_in_progress {
                vec![("Esc", "Cancel"), ("", upscale_status.as_str())]
            } else if app.gallery.view_mode == crate::app::GalleryViewMode::Detail {
                vec![
                    ("e", "Edit"),
                    ("r", "Regen"),
                    ("u", "Upscale"),
                    ("d", "Delete"),
                    ("o/Enter", "Open"),
                    ("j/k", "Prev/Next"),
                    ("Esc", "Grid"),
                ]
            } else {
                vec![
                    ("hjkl", "Navigate"),
                    ("Enter", "Details"),
                    ("e", "Edit"),
                    ("u", "Upscale"),
                    ("d", "Delete"),
                    ("Esc", "Back"),
                    ("?", "Help"),
                    ("q", "Quit"),
                ]
            }
        }
        View::Models => vec![
            ("1-4", "Views"),
            ("Enter", "Select"),
            ("p", "Pull"),
            ("u", "Unload"),
            ("Esc", "Back"),
            ("?", "Help"),
            ("q", "Quit"),
        ],
        View::Settings => vec![
            ("j/k", "Navigate"),
            ("+/-", "Adjust"),
            ("Enter", "Edit"),
            ("Esc", "Back"),
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

    let bar = Paragraph::new(Line::from(spans)).style(theme.status_bar());
    frame.render_widget(bar, area);
}
