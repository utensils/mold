//! Create view — the primary workspace.
//!
//! Layout matches the Mold Studio mockup:
//!
//! ```text
//! ┌── Prompt (full width, 4 rows) ─────────────────────────────────┐
//! ├── Parameters (38 col) ─────┬── Preview (flex) ─────────────────┤
//! │   essentials rows          │                                   │
//! │   ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄       │                                   │
//! │   ▸ Advanced  <hint>       ├── Timeline (7 rows) ──────────────┤
//! │   (sections when open)     │   glyph log, 5 visible rows       │
//! └── Error (1 row when set) ──┴───────────────────────────────────┘
//! ```
//!
//! - Left column is fixed at 38 cells so the Parameters form lines up even
//!   on very wide terminals.
//! - The Info sub-panel and Recent strip are retired: telemetry lives in
//!   Machines/chrome, the model description is a dim line under the Model
//!   row, and the Timeline's `✓ Saved <file>` lines absorb Recent.
//! - The Timeline collapses entirely (never squeezes) when the pref is off
//!   or the terminal is too short.
//! - Error row is 0 cells tall when no error is present — it collapses out
//!   of the layout rather than leaving a gap.

use ratatui::prelude::*;
use ratatui::widgets::Paragraph;
use tui_textarea::TextArea;

use crate::app::{App, GenerateFocus};
use crate::ui::widgets::panel_block;
use crate::ui::{param_form, preview, theme::Theme, timeline};

/// Width of the left (Parameters) column.
const LEFT_COL_WIDTH: u16 = 38;
/// Height of the Prompt textarea (including its border).
const PROMPT_HEIGHT: u16 = 4;
/// Height of the Timeline panel (including borders).
pub(crate) const TIMELINE_HEIGHT: u16 = 7;
/// Minimum Preview height the Timeline must never squeeze below.
const MIN_PREVIEW_HEIGHT: u16 = 8;

/// Layout contract: the Timeline's inner area must fit every advertised
/// log row plus the top/bottom border. Fails the build if either constant
/// drifts (successor of the old Recent-strip assert).
const _: () = assert!(TIMELINE_HEIGHT as usize >= timeline::TIMELINE_VISIBLE_ROWS + 2);

/// Render the Create view.
pub fn render(frame: &mut Frame, app: &mut App, area: Rect) {
    let show_error = app.generate.error_message.is_some();

    let mut constraints: Vec<Constraint> =
        vec![Constraint::Length(PROMPT_HEIGHT), Constraint::Min(10)];
    if show_error {
        constraints.push(Constraint::Length(1));
    }
    let rects = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    render_prompt(frame, app, rects[0]);
    render_middle_row(frame, app, rects[1]);
    if show_error {
        render_error(frame, app, rects[2]);
    }
}

/// Whether the Timeline panel renders at all: the pref must be on and the
/// view must be tall enough to keep the Preview at its minimum — collapse,
/// don't squeeze.
fn timeline_visible(app: &App, area: Rect) -> bool {
    app.show_timeline && area.height >= PROMPT_HEIGHT + MIN_PREVIEW_HEIGHT + TIMELINE_HEIGHT
}

// ── Middle row: Parameters + Preview/Timeline ───────────────────────

fn render_middle_row(frame: &mut Frame, app: &mut App, area: Rect) {
    let [left, right] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(LEFT_COL_WIDTH), Constraint::Min(20)])
        .areas(area);

    app.layout.parameters = left;

    let focused = app.generate.focus == GenerateFocus::Parameters;
    // The theme is read-only while the form mutates scroll + textarea
    // state; `theme` and `generate` are disjoint App fields so the borrows
    // split cleanly.
    let App {
        ref theme,
        ref mut generate,
        ..
    } = *app;
    let neg_rect = param_form::render(frame, theme, generate, left, focused);
    // Mouse hit-testing: the inline Negative editor's rect (inside the
    // Parameters panel) or an empty rect when it isn't rendered.
    app.layout.negative_prompt = neg_rect.unwrap_or_default();

    if timeline_visible(app, area) {
        let [preview_area, timeline_area] = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(MIN_PREVIEW_HEIGHT),
                Constraint::Length(TIMELINE_HEIGHT),
            ])
            .areas(right);
        app.layout.preview = preview_area;
        preview::render(frame, app, preview_area);
        timeline::render(frame, app, timeline_area);
    } else {
        app.layout.preview = right;
        preview::render(frame, app, right);
    }
}

// ── Prompt ──────────────────────────────────────────────────────────

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

// ── Error row ───────────────────────────────────────────────────────

fn render_error(frame: &mut Frame, app: &App, area: Rect) {
    if let Some(err) = &app.generate.error_message {
        let line = Paragraph::new(format!(" \u{2717} {err}")).style(app.theme.error());
        frame.render_widget(line, area);
    }
}
