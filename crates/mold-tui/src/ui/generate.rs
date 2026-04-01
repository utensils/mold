use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui_image::StatefulImage;

use crate::app::{App, GenerateFocus};
use crate::ui::{info, param_form, progress, theme::Theme};

/// Render the Generate view.
pub fn render(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = &app.theme;

    // Determine if the negative prompt should be shown
    let show_negative = app.generate.capabilities.supports_negative_prompt;

    // Main vertical layout:
    // - Prompt (2-4 lines)
    // - Negative prompt (if applicable, 2 lines)
    // - Middle row: Parameters (left) + Preview (right)
    // - Progress (3-5 lines)
    // - Error message (if any, 1 line)
    let prompt_height = 4;
    let neg_height = if show_negative { 3 } else { 0 };
    let progress_height = 5;
    let error_height = if app.generate.error_message.is_some() {
        1
    } else {
        0
    };

    let constraints = if show_negative {
        vec![
            Constraint::Length(prompt_height),
            Constraint::Length(neg_height),
            Constraint::Min(8),
            Constraint::Length(progress_height),
            Constraint::Length(error_height),
        ]
    } else {
        vec![
            Constraint::Length(prompt_height),
            Constraint::Min(8),
            Constraint::Length(progress_height),
            Constraint::Length(error_height),
        ]
    };

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let (prompt_area, neg_area, middle_area, progress_area, error_area) = if show_negative {
        (
            layout[0],
            Some(layout[1]),
            layout[2],
            layout[3],
            layout.get(4).copied(),
        )
    } else {
        (
            layout[0],
            None,
            layout[1],
            layout[2],
            layout.get(3).copied(),
        )
    };

    // Store layout areas for mouse hit-testing
    app.layout.prompt = prompt_area;
    if let Some(neg) = neg_area {
        app.layout.negative_prompt = neg;
    }

    // ── Prompt ─────────────────────────────────────────────
    render_text_area(
        frame,
        theme,
        &mut app.generate.prompt,
        " Prompt ",
        prompt_area,
        app.generate.focus == GenerateFocus::Prompt,
    );

    // ── Negative Prompt ────────────────────────────────────
    if let Some(neg_area) = neg_area {
        render_text_area(
            frame,
            theme,
            &mut app.generate.negative_prompt,
            " Negative ",
            neg_area,
            app.generate.focus == GenerateFocus::NegativePrompt,
        );
    }

    // ── Middle row: Parameters/Info (left) + Preview (right) ──
    let middle_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(38), // Left column (params + info)
            Constraint::Min(20),    // Preview panel
        ])
        .split(middle_area);

    // Left column: Parameters (top) + Info (bottom, 5 lines)
    let left_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(5)])
        .split(middle_layout[0]);

    // Store areas for mouse
    app.layout.parameters = left_layout[0];
    app.layout.preview = middle_layout[1];

    // Parameters
    param_form::render(
        frame,
        theme,
        &app.generate,
        left_layout[0],
        app.generate.focus == GenerateFocus::Parameters,
    );

    // Info panel
    info::render(frame, app, left_layout[1]);

    // Preview
    render_preview(frame, app, middle_layout[1]);

    // ── Progress ───────────────────────────────────────────
    progress::render(frame, app, progress_area, false);

    // ── Error message ──────────────────────────────────────
    if let Some(error_area) = error_area {
        if let Some(err) = &app.generate.error_message {
            let error_line = Paragraph::new(format!(" \u{2717} {err}")).style(app.theme.error());
            frame.render_widget(error_line, error_area);
        }
    }
}

fn render_text_area<'a>(
    frame: &mut Frame,
    theme: &Theme,
    textarea: &mut tui_textarea::TextArea<'a>,
    title: &'a str,
    area: Rect,
    focused: bool,
) {
    let border_style = if focused {
        theme.border_focused()
    } else {
        theme.border()
    };
    let title_style = if focused {
        theme.title_focused()
    } else {
        theme.title()
    };

    textarea.set_block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .title(title)
            .title_style(title_style)
            .style(Style::default().bg(theme.bg)),
    );

    textarea.set_style(Style::default().fg(theme.text).bg(theme.bg));
    textarea.set_cursor_style(if focused {
        Style::default().fg(theme.bg).bg(theme.accent)
    } else {
        Style::default()
    });
    textarea.set_placeholder_style(theme.dim());

    frame.render_widget(&*textarea, area);
}

fn render_preview(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = &app.theme;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border())
        .title(" Preview ")
        .title_style(theme.title())
        .style(Style::default().bg(theme.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    // Show the image if we have one
    if let Some(ref mut image_state) = app.generate.image_state {
        let image_widget = StatefulImage::default();
        frame.render_stateful_widget(image_widget, inner, image_state);

        // Show seed/time info below the image if there's room
        if let (Some(seed), Some(time_ms)) =
            (app.generate.last_seed, app.generate.last_generation_time_ms)
        {
            if inner.height > 2 {
                let seed_str = seed.to_string();
                let seed_display = if seed_str.len() > 10 {
                    format!("{}...", &seed_str[..8])
                } else {
                    seed_str
                };
                let info = format!("{seed_display} | {:.1}s", time_ms as f64 / 1000.0);
                let info_area = Rect {
                    x: inner.x,
                    y: inner.y + inner.height - 1,
                    width: inner.width,
                    height: 1,
                };
                let p = Paragraph::new(info)
                    .style(theme.dim())
                    .alignment(Alignment::Center);
                frame.render_widget(p, info_area);
            }
        }
    } else if app.generate.generating {
        // Show a "generating..." message
        let msg = Paragraph::new("Generating...")
            .style(theme.dim())
            .alignment(Alignment::Center);
        let center = Rect {
            x: inner.x,
            y: inner.y + inner.height / 2,
            width: inner.width,
            height: 1,
        };
        frame.render_widget(msg, center);
    } else {
        // Empty state
        let msg = Paragraph::new("Press Enter to generate")
            .style(theme.dim())
            .alignment(Alignment::Center);
        let center = Rect {
            x: inner.x,
            y: inner.y + inner.height / 2,
            width: inner.width,
            height: 1,
        };
        frame.render_widget(msg, center);
    }
}
