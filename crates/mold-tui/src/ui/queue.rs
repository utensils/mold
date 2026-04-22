//! Queue view — a read-only snapshot of what's running and what ran recently.
//!
//! The existing mold runtime generates one image at a time from the TUI, so
//! there isn't a real multi-job queue to display. Instead the view composes
//! three sources:
//!
//! - **Running** — the active generation (from `app.generate`), shown only
//!   while `generating` is true.
//! - **Recent** — the most recent prompt-history entries, mapped to a
//!   `done` state with a cached generation time from the matching gallery
//!   row when available.
//! - **Footer** — a compact line pointing users at the Generate and Gallery
//!   tabs. When mold grows a proper client-side job queue this is where the
//!   queued-state rows will appear.
//!
//! The view is deliberately simple — no selection, no keyboard shortcuts
//! beyond tab switching — until the underlying runtime exposes real queued
//! work.

use ratatui::prelude::*;
use ratatui::widgets::{Cell, Paragraph, Row, Table, TableState, Wrap};

use crate::app::App;

use super::widgets::{panel_block, truncate_with_ellipsis};

/// Visual state of a Queue row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobState {
    Running,
    Done,
}

impl JobState {
    fn icon(self) -> &'static str {
        match self {
            JobState::Running => "\u{25B6}", // ▶
            JobState::Done => "\u{2713}",    // ✓
        }
    }

    fn label(self) -> &'static str {
        match self {
            JobState::Running => "running",
            JobState::Done => "done",
        }
    }
}

/// A single row in the Queue table.
///
/// This is a view-local struct — callers compose it from `app.generate`,
/// history, and gallery state. Using the app types directly would leak
/// unrelated fields into the render loop.
struct JobRow {
    state: JobState,
    prompt: String,
    model: String,
    time: String,
}

/// Render the Queue view.
pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    let jobs = build_rows(app);

    let [table_area, footer_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(3)])
        .areas(area);

    render_table(frame, app, &jobs, table_area);
    render_footer(frame, app, &jobs, footer_area);
}

fn render_table(frame: &mut Frame, app: &App, jobs: &[JobRow], area: Rect) {
    let theme = &app.theme;

    let running = jobs.iter().filter(|j| j.state == JobState::Running).count();
    let done = jobs.iter().filter(|j| j.state == JobState::Done).count();
    let hint = format!("{running} running · {done} recent");

    let block = panel_block(theme, "Queue", true, Some(&hint));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    if jobs.is_empty() {
        let msg = Paragraph::new("No jobs yet — start a generation from the Generate tab.")
            .style(theme.dim())
            .wrap(Wrap { trim: true });
        frame.render_widget(msg, inner);
        return;
    }

    let header = Row::new(vec![
        Cell::from(" "),
        Cell::from("STATE").style(theme.dim().add_modifier(Modifier::BOLD)),
        Cell::from("PROMPT").style(theme.dim().add_modifier(Modifier::BOLD)),
        Cell::from("MODEL").style(theme.dim().add_modifier(Modifier::BOLD)),
        Cell::from("TIME").style(theme.dim().add_modifier(Modifier::BOLD)),
    ])
    .height(1);

    let rows: Vec<Row> = jobs.iter().map(|j| row_for(theme, j)).collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(2),
            Constraint::Length(8),
            Constraint::Min(20),
            Constraint::Length(24),
            Constraint::Length(10),
        ],
    )
    .header(header)
    // The outer "Queue" panel_block already draws the frame. Using another
    // bordered block here would draw a second set of borders inside and
    // shrink the visible column area — render the table plain, tinted to
    // the panel background so the highlight row blends cleanly.
    .style(Style::default().bg(theme.bg))
    .row_highlight_style(theme.list_selected());

    // Render inside `inner` (not `area`) so the outer frame stays visible.
    let mut state = TableState::default();
    frame.render_stateful_widget(table, inner, &mut state);
}

fn row_for<'a>(theme: &crate::ui::theme::Theme, job: &'a JobRow) -> Row<'a> {
    let icon_style = match job.state {
        JobState::Running => theme.warning(),
        JobState::Done => theme.success(),
    };
    Row::new(vec![
        Cell::from(job.state.icon()).style(icon_style),
        Cell::from(job.state.label()).style(theme.dim()),
        Cell::from(job.prompt.as_str()).style(theme.param_value()),
        Cell::from(job.model.as_str()).style(theme.dim()),
        Cell::from(job.time.as_str()).style(theme.dim()),
    ])
    .height(1)
}

fn render_footer(frame: &mut Frame, app: &App, jobs: &[JobRow], area: Rect) {
    let theme = &app.theme;
    let block = panel_block(theme, "Overview", false, None);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let running = jobs.iter().any(|j| j.state == JobState::Running);
    let line = if running {
        Line::from(vec![
            Span::styled("\u{25B6} ", Style::default().fg(theme.warning)),
            Span::styled("one generation is running · ", theme.dim()),
            Span::styled("press 1 ", theme.status_key()),
            Span::styled("to return to Generate.", theme.dim()),
        ])
    } else if jobs.is_empty() {
        Line::from(Span::styled(
            "queue idle · press 1 to generate, 2 to browse the gallery.",
            theme.dim(),
        ))
    } else {
        Line::from(vec![
            Span::styled("queue idle · ", theme.dim()),
            Span::styled(format!("{} recent · ", jobs.len()), theme.dim()),
            Span::styled("press 2 ", theme.status_key()),
            Span::styled("to open the gallery.", theme.dim()),
        ])
    };

    let para = Paragraph::new(line);
    frame.render_widget(para, inner);
}

// ── Row assembly ────────────────────────────────────────────────────

fn build_rows(app: &App) -> Vec<JobRow> {
    let mut rows: Vec<JobRow> = Vec::new();

    if app.generate.generating {
        // `prompt_preview` only consumes the first non-empty line, so
        // scan straight from the TextArea buffer instead of joining
        // every line into a throwaway allocation. The render path runs
        // on every frame (~60fps) — no reason to copy the full prompt
        // just to look at the head.
        let first_line = app
            .generate
            .prompt
            .lines()
            .iter()
            .map(|s| s.trim())
            .find(|s| !s.is_empty())
            .unwrap_or("");
        rows.push(JobRow {
            state: JobState::Running,
            prompt: prompt_preview(first_line),
            model: app.generate.params.model.clone(),
            time: running_time_label(
                app.generate.progress.denoise_step,
                app.generate.progress.denoise_total,
            ),
        });
    }

    for entry in app.history.recent(8) {
        rows.push(JobRow {
            state: JobState::Done,
            prompt: prompt_preview(&entry.prompt),
            model: if entry.model.is_empty() {
                "—".to_string()
            } else {
                entry.model.clone()
            },
            time: relative_time(entry.timestamp),
        });
    }

    rows
}

fn prompt_preview(prompt: &str) -> String {
    const MAX: usize = 80;
    let first_line = prompt.lines().next().unwrap_or("").trim();
    if first_line.is_empty() {
        return "(empty)".to_string();
    }
    truncate_with_ellipsis(first_line, MAX)
}

/// Short status for a running job — renders the current denoise step if known.
fn running_time_label(step: usize, total: usize) -> String {
    if total > 0 {
        format!("{step}/{total}")
    } else {
        "…".to_string()
    }
}

/// Convert a UNIX-seconds timestamp into a compact relative label.
fn relative_time(ts: u64) -> String {
    if ts == 0 {
        return "—".to_string();
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let delta = now.saturating_sub(ts);
    match delta {
        0..=59 => "now".to_string(),
        60..=3599 => format!("{}m", delta / 60),
        3600..=86_399 => format!("{}h", delta / 3600),
        _ => format!("{}d", delta / 86_400),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_preview_keeps_short_prompts_intact() {
        assert_eq!(prompt_preview("a cat"), "a cat");
    }

    #[test]
    fn prompt_preview_truncates_long_prompts_with_ellipsis() {
        let long = "x".repeat(200);
        let result = prompt_preview(&long);
        assert!(result.ends_with('…'));
        assert!(result.chars().count() <= 80);
    }

    #[test]
    fn prompt_preview_empty_renders_placeholder() {
        assert_eq!(prompt_preview(""), "(empty)");
        assert_eq!(prompt_preview("   "), "(empty)");
    }

    #[test]
    fn relative_time_handles_zero_as_unknown() {
        assert_eq!(relative_time(0), "—");
    }

    #[test]
    fn running_time_label_uses_fraction_when_total_known() {
        assert_eq!(running_time_label(3, 10), "3/10");
        assert_eq!(running_time_label(0, 0), "…");
    }
}
