//! Generate view — the primary workspace.
//!
//! Layout matches the mold design system:
//!
//! ```text
//! ┌── Prompt (full width) ────────────────────────────────────────┐
//! ├── Negative (optional, full width) ────────────────────────────┤
//! ├── Parameters (≥38 col) ────┬── Preview (flex) ────────────────┤
//! │                            │                                  │
//! │   Info (5 rows)            │                                  │
//! ├── Recent (≥38 col) ────────┼── Timeline (flex) ───────────────┤
//! └── Error (1 row when set) ──┴──────────────────────────────────┘
//! ```
//!
//! - Left column is fixed at 38 cells so the Parameters form lines up even on
//!   very wide terminals.
//! - `Recent` and `Timeline` share a `BOTTOM_ROW_HEIGHT` row below the main
//!   preview area.
//! - Negative prompt is hidden entirely for models that don't support it.
//! - Error row is 0 cells tall when no error is present — it collapses out of
//!   the layout rather than leaving a gap.

use ratatui::prelude::*;
use ratatui::widgets::Paragraph;
use ratatui_image::StatefulImage;
use tui_textarea::TextArea;

use crate::app::{App, GenerateFocus};
use crate::ui::widgets::panel_block;
use crate::ui::{info, param_form, progress, recent, theme::Theme};

/// Width of the left column (parameters + info + recent).
const LEFT_COL_WIDTH: u16 = 38;
/// Height of the Info sub-panel at the bottom of the left column.
const INFO_HEIGHT: u16 = 5;
/// Height of the bottom row (Recent + Timeline panels).
const BOTTOM_ROW_HEIGHT: u16 = 5;
/// Height of the Prompt textarea (including its border).
const PROMPT_HEIGHT: u16 = 4;
/// Height of the Negative prompt textarea when visible (including border).
const NEGATIVE_HEIGHT: u16 = 3;
/// Height of the collapsed Negative prompt summary (one dim row, no border).
const NEGATIVE_COLLAPSED_HEIGHT: u16 = 1;

/// Render the Generate view.
pub fn render(frame: &mut Frame, app: &mut App, area: Rect) {
    let layout = vertical_layout(app, area);

    render_prompt(frame, app, layout.prompt);
    if let Some(neg_area) = layout.negative {
        render_negative(frame, app, neg_area);
    }
    render_middle_row(frame, app, layout.middle);
    render_bottom_row(frame, app, layout.bottom);
    if let Some(err_area) = layout.error {
        render_error(frame, app, err_area);
    }
}

// ── Layout ──────────────────────────────────────────────────────────

/// Resolved areas for each row in the Generate view.
///
/// `negative` is `Some` whenever the Negative prompt row has reserved space,
/// whether full-height or collapsed. Whether it renders as a textarea or as
/// the dim summary line is decided inside `render_negative`.
struct GenerateLayout {
    prompt: Rect,
    negative: Option<Rect>,
    middle: Rect,
    bottom: Rect,
    error: Option<Rect>,
}

fn vertical_layout(app: &App, area: Rect) -> GenerateLayout {
    let neg_height = negative_row_height(app);
    let show_error = app.generate.error_message.is_some();

    let mut constraints: Vec<Constraint> = Vec::with_capacity(5);
    constraints.push(Constraint::Length(PROMPT_HEIGHT));
    if neg_height > 0 {
        constraints.push(Constraint::Length(neg_height));
    }
    constraints.push(Constraint::Min(6));
    constraints.push(Constraint::Length(BOTTOM_ROW_HEIGHT));
    if show_error {
        constraints.push(Constraint::Length(1));
    }

    let rects = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let mut idx = 0;
    let prompt = rects[idx];
    idx += 1;
    let negative = if neg_height > 0 {
        let r = rects[idx];
        idx += 1;
        Some(r)
    } else {
        None
    };
    let middle = rects[idx];
    idx += 1;
    let bottom = rects[idx];
    idx += 1;
    let error = if show_error { Some(rects[idx]) } else { None };

    GenerateLayout {
        prompt,
        negative,
        middle,
        bottom,
        error,
    }
}

/// How many vertical cells the Negative row needs (including borders).
///
/// - 0: model doesn't support negative prompts — row is skipped entirely.
/// - `NEGATIVE_COLLAPSED_HEIGHT`: supported but user has collapsed it.
/// - `NEGATIVE_HEIGHT`: supported and expanded (full textarea).
fn negative_row_height(app: &App) -> u16 {
    if !app.generate.capabilities.supports_negative_prompt {
        0
    } else if app.generate.negative_collapsed {
        NEGATIVE_COLLAPSED_HEIGHT
    } else {
        NEGATIVE_HEIGHT
    }
}

// ── Middle row: Parameters/Info + Preview ───────────────────────────

fn render_middle_row(frame: &mut Frame, app: &mut App, area: Rect) {
    let [left, right] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(LEFT_COL_WIDTH), Constraint::Min(20)])
        .areas(area);

    let [params_area, info_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(INFO_HEIGHT)])
        .areas(left);

    app.layout.parameters = params_area;
    app.layout.preview = right;

    param_form::render(
        frame,
        &app.theme,
        &app.generate,
        params_area,
        app.generate.focus == GenerateFocus::Parameters,
    );
    info::render(frame, app, info_area);
    render_preview(frame, app, right);
}

// ── Bottom row: Recent + Timeline ───────────────────────────────────

fn render_bottom_row(frame: &mut Frame, app: &App, area: Rect) {
    let [recent_area, timeline_area] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(LEFT_COL_WIDTH), Constraint::Min(20)])
        .areas(area);

    recent::render(frame, app, recent_area);
    progress::render_with_title(frame, app, timeline_area, false, "Timeline");
}

// ── Prompt / Negative prompt ────────────────────────────────────────

fn render_prompt(frame: &mut Frame, app: &mut App, area: Rect) {
    app.layout.prompt = area;
    render_text_area(
        frame,
        &app.theme,
        &mut app.generate.prompt,
        "Prompt",
        area,
        app.generate.focus == GenerateFocus::Prompt,
    );
}

fn render_negative(frame: &mut Frame, app: &mut App, area: Rect) {
    app.layout.negative_prompt = area;
    if app.generate.negative_collapsed {
        render_negative_collapsed(frame, app, area);
    } else {
        render_text_area(
            frame,
            &app.theme,
            &mut app.generate.negative_prompt,
            "Negative",
            area,
            app.generate.focus == GenerateFocus::NegativePrompt,
        );
    }
}

/// Dim one-row summary shown when the user has collapsed the Negative panel.
///
/// Mirrors the design system's inline `neg: …` treatment — the field is still
/// visible, just quiet. Users press `Alt+N` to expand it again.
fn render_negative_collapsed(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;
    let summary: String = app
        .generate
        .negative_prompt
        .lines()
        .iter()
        .flat_map(|l| l.chars())
        .take(80)
        .collect();
    let label = if summary.trim().is_empty() {
        "neg: (empty)".to_string()
    } else {
        format!("neg: {summary}")
    };
    let line = Line::from(vec![
        Span::styled(label, theme.dim()),
        Span::raw("   "),
        Span::styled("Alt+N", theme.status_key()),
        Span::styled(" expand", theme.dim()),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn render_text_area<'a>(
    frame: &mut Frame,
    theme: &Theme,
    textarea: &mut TextArea<'a>,
    title: &'a str,
    area: Rect,
    focused: bool,
) {
    textarea.set_block(panel_block(theme, title, focused, None));
    textarea.set_style(Style::default().fg(theme.text).bg(theme.bg));
    textarea.set_cursor_style(if focused {
        Style::default().fg(theme.bg).bg(theme.accent)
    } else {
        Style::default()
    });
    textarea.set_placeholder_style(theme.dim());
    frame.render_widget(&*textarea, area);
}

// ── Preview ─────────────────────────────────────────────────────────

fn render_preview(frame: &mut Frame, app: &mut App, area: Rect) {
    // Preview is display-only in the current focus model, so the panel always
    // renders with the unfocused border colour. If a focus state is added
    // later, flip this flag.
    let block = panel_block(&app.theme, "Preview", false, None);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    if let Some(ref mut image_state) = app.generate.image_state {
        let image_widget = StatefulImage::default().resize(ratatui_image::Resize::Scale(None));
        frame.render_stateful_widget(image_widget, inner, image_state);
        return;
    }

    let status_text = if app.generate.generating {
        if app.generate.progress.is_downloading() {
            app.generate.progress.download_status_text().to_string()
        } else {
            "Generating...".to_string()
        }
    } else {
        "Press Enter to generate".to_string()
    };

    let msg = Paragraph::new(status_text)
        .style(app.theme.dim())
        .alignment(Alignment::Center);
    let center = Rect {
        x: inner.x,
        y: inner.y + inner.height / 2,
        width: inner.width,
        height: 1,
    };
    frame.render_widget(msg, center);
}

// ── Error row ───────────────────────────────────────────────────────

fn render_error(frame: &mut Frame, app: &App, area: Rect) {
    if let Some(err) = &app.generate.error_message {
        let line = Paragraph::new(format!(" \u{2717} {err}")).style(app.theme.error());
        frame.render_widget(line, area);
    }
}
