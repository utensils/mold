use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph};
use ratatui_image::StatefulImage;

use crate::app::App;

/// Render the Gallery view.
pub fn render(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = &app.theme;

    // Three-row layout: list+preview, details
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(5)])
        .split(area);

    // Top row: history list (left) + image preview (right)
    let top_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(28), Constraint::Min(20)])
        .split(layout[0]);

    // ── History list ───────────────────────────────────────
    let list_block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border_focused())
        .title(" History ")
        .title_style(theme.title_focused())
        .style(Style::default().bg(theme.bg));

    if app.gallery.entries.is_empty() {
        let empty = Paragraph::new("No images found")
            .style(theme.dim())
            .alignment(Alignment::Center)
            .block(list_block);
        frame.render_widget(empty, top_layout[0]);
    } else {
        let items: Vec<ListItem> = app
            .gallery
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let preview = if entry.prompt_preview.len() > 18 {
                    format!("{}...", &entry.prompt_preview[..15])
                } else {
                    entry.prompt_preview.clone()
                };
                let time = entry
                    .generation_time_ms
                    .map(|ms| format!("{:.1}s", ms as f64 / 1000.0))
                    .unwrap_or_default();

                let style = if i == app.gallery.selected {
                    theme.list_selected()
                } else {
                    Style::default().fg(theme.text)
                };

                let marker = if i == app.gallery.selected {
                    "\u{25b8} "
                } else {
                    "  "
                };

                ListItem::new(Line::from(vec![
                    Span::styled(marker, style),
                    Span::styled(preview, style),
                    Span::styled(format!(" {time:>5}"), Style::default().fg(theme.text_dim)),
                ]))
            })
            .collect();

        let list = List::new(items).block(list_block);
        let mut state = ListState::default().with_selected(Some(app.gallery.selected));
        frame.render_stateful_widget(list, top_layout[0], &mut state);
    }

    // ── Image preview ──────────────────────────────────────
    let preview_block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border())
        .title(" Image ")
        .title_style(theme.title())
        .style(Style::default().bg(theme.bg));

    let preview_inner = preview_block.inner(top_layout[1]);
    frame.render_widget(preview_block, top_layout[1]);

    if let Some(ref mut image_state) = app.gallery.image_state {
        let image_widget = StatefulImage::default();
        frame.render_stateful_widget(image_widget, preview_inner, image_state);
    } else if !app.gallery.entries.is_empty() {
        let msg = Paragraph::new("Select an image")
            .style(theme.dim())
            .alignment(Alignment::Center);
        let center = Rect {
            x: preview_inner.x,
            y: preview_inner.y + preview_inner.height / 2,
            width: preview_inner.width,
            height: 1,
        };
        frame.render_widget(msg, center);
    }

    // ── Details ────────────────────────────────────────────
    let details_block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border())
        .title(" Details ")
        .title_style(theme.title())
        .style(Style::default().bg(theme.bg));

    if let Some(entry) = app.gallery.entries.get(app.gallery.selected) {
        let line1 = format!(
            "{} \u{00b7} {}x{} \u{00b7} seed {}",
            entry.model,
            entry.width,
            entry.height,
            entry
                .seed
                .map(|s| s.to_string())
                .unwrap_or_else(|| "?".into()),
        );
        let line2 = format!("\"{}\"\n", entry.prompt_preview);
        let line3 = format!("File: {}", entry.path.display());

        let details = Paragraph::new(vec![
            Line::from(Span::styled(line1, Style::default().fg(theme.text))),
            Line::from(Span::styled(line2, theme.dim())),
            Line::from(Span::styled(line3, theme.dim())),
        ])
        .block(details_block);
        frame.render_widget(details, layout[1]);
    } else {
        let empty = Paragraph::new("").block(details_block);
        frame.render_widget(empty, layout[1]);
    }
}
