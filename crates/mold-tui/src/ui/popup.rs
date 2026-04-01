use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};

use crate::app::{App, Popup};

/// Render the active popup overlay.
pub fn render(frame: &mut Frame, app: &mut App) {
    match &app.popup {
        Some(Popup::Help) => render_help(frame, app),
        Some(Popup::ModelSelector { .. }) => render_model_selector(frame, app),
        Some(Popup::Confirm { message, .. }) => render_confirm(frame, app, message.clone()),
        None => {}
    }
}

fn centered_rect(area: Rect, width_pct: u16, height_pct: u16) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - height_pct) / 2),
            Constraint::Percentage(height_pct),
            Constraint::Percentage((100 - height_pct) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - width_pct) / 2),
            Constraint::Percentage(width_pct),
            Constraint::Percentage((100 - width_pct) / 2),
        ])
        .split(vertical[1])[1]
}

fn render_help(frame: &mut Frame, app: &App) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 60, 70);

    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.popup_border())
        .title(" Keybindings ")
        .title_style(theme.title_focused())
        .style(theme.popup_bg());

    let help_text = vec![
        Line::from(Span::styled(
            "Navigation",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from("  Tab / Shift+Tab    Cycle focus between panels"),
        Line::from("  Alt+1/2/3          Switch to Generate/Gallery/Models"),
        Line::from("  Esc                Close popup / cancel"),
        Line::from("  q / Ctrl+C         Quit"),
        Line::from(""),
        Line::from(Span::styled(
            "Generate View",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from("  Enter              Start generation"),
        Line::from("  Ctrl+E             Expand prompt via LLM"),
        Line::from("  Ctrl+S             Save current image"),
        Line::from("  Ctrl+R             Randomize seed"),
        Line::from("  Ctrl+M             Open model selector"),
        Line::from("  Ctrl+K             Compare models"),
        Line::from("  j/k                Navigate parameters"),
        Line::from("  +/- or Left/Right  Adjust parameter value"),
        Line::from(""),
        Line::from(Span::styled(
            "Gallery View",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from("  j/k                Navigate history"),
        Line::from("  Enter              Re-generate with same params"),
        Line::from("  e                  Edit parameters & generate"),
        Line::from("  d                  Delete image"),
        Line::from("  o                  Open in system viewer"),
        Line::from("  hjkl               Pan image viewport"),
        Line::from("  +/-                Zoom in/out"),
        Line::from(""),
        Line::from(Span::styled(
            "Models View",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from("  Enter              Select as default model"),
        Line::from("  p                  Pull (download) model"),
        Line::from("  r                  Remove model"),
        Line::from("  u                  Unload from GPU"),
        Line::from("  /                  Filter by name"),
    ];

    let paragraph = Paragraph::new(help_text)
        .block(block)
        .style(Style::default().fg(theme.text))
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_model_selector(frame: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 55, 50);

    frame.render_widget(Clear, area);

    if let Some(Popup::ModelSelector {
        filter,
        selected,
        filtered,
    }) = &app.popup
    {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.popup_border())
            .title(" Select Model ")
            .title_style(theme.title_focused())
            .style(theme.popup_bg());

        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.height < 3 {
            return;
        }

        // Filter input
        let filter_display = if filter.is_empty() {
            "Type to filter...".to_string()
        } else {
            filter.clone()
        };
        let filter_style = if filter.is_empty() {
            theme.dim()
        } else {
            Style::default().fg(theme.text)
        };
        let filter_line = Paragraph::new(format!("Filter: {filter_display}"))
            .style(filter_style);
        let filter_area = Rect {
            x: inner.x,
            y: inner.y,
            width: inner.width,
            height: 1,
        };
        frame.render_widget(filter_line, filter_area);

        // Model list
        let list_area = Rect {
            x: inner.x,
            y: inner.y + 2,
            width: inner.width,
            height: inner.height.saturating_sub(2),
        };

        let items: Vec<ListItem> = filtered
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let style = if i == *selected {
                    theme.list_selected()
                } else {
                    Style::default().fg(theme.text)
                };
                let marker = if i == *selected { "\u{25b8} " } else { "  " };
                ListItem::new(format!("{marker}{name}")).style(style)
            })
            .collect();

        let list = List::new(items);
        let mut state = ListState::default().with_selected(Some(*selected));
        frame.render_stateful_widget(list, list_area, &mut state);
    }
}

fn render_confirm(frame: &mut Frame, app: &App, message: String) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 40, 20);

    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.popup_border())
        .title(" Confirm ")
        .title_style(theme.title_focused())
        .style(theme.popup_bg());

    let text = vec![
        Line::from(message),
        Line::from(""),
        Line::from(vec![
            Span::styled("y", theme.status_key()),
            Span::styled(" Confirm  ", Style::default().fg(theme.text)),
            Span::styled("n", theme.status_key()),
            Span::styled(" Cancel", Style::default().fg(theme.text)),
        ]),
    ];

    let paragraph = Paragraph::new(text)
        .block(block)
        .alignment(Alignment::Center);

    frame.render_widget(paragraph, area);
}
