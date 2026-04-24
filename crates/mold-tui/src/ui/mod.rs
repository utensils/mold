pub mod gallery;
pub mod generate;
pub mod info;
pub mod models;
pub mod param_form;
pub mod popup;
pub mod progress;
pub mod queue;
pub mod recent;
pub mod script_composer;
pub mod settings;
pub mod theme;
pub mod widgets;

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
        View::Queue => queue::render(frame, app, layout[1]),
        View::Settings => settings::render(frame, app, layout[1]),
        View::Script => script_composer::render(frame, &app.script, layout[1], &app.theme),
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

    // Build right-aligned title: connection indicator + version
    let version = format!("mold {} ", mold_core::build_info::version_string());
    let mut right_spans = Vec::new();

    if app.generate.params.inference_mode == crate::app::InferenceMode::Local {
        right_spans.push(Span::styled("local ", Style::default().fg(theme.text_dim)));
    } else if let Some(ref status) = app.resource_info.server_status {
        let host_label = status.hostname.as_deref().unwrap_or("remote");
        right_spans.push(Span::styled(
            format!("{host_label} "),
            Style::default().fg(theme.accent),
        ));
    } else if app.connecting {
        right_spans.push(Span::styled(
            "connecting... ",
            Style::default().fg(theme.warning),
        ));
    }

    right_spans.push(Span::styled(version, Style::default().fg(theme.text_dim)));

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
                .title_top(Line::from(right_spans).right_aligned())
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
    let up = &app.upscale_progress;
    let has_download = up.is_downloading() && up.download_batch_total > 0;

    // Use taller overlay when showing download progress (need extra row)
    let bar_height = if has_download { 4u16 } else { 3u16 };
    if gallery_area.height < bar_height + 2 {
        return;
    }
    let area = Rect {
        x: gallery_area.x,
        y: gallery_area.y + gallery_area.height - bar_height,
        width: gallery_area.width,
        height: bar_height,
    };

    // Clear area first to prevent image protocol artifacts
    frame.render_widget(ratatui::widgets::Clear, area);

    let title = if has_download {
        " Downloading Upscaler "
    } else {
        " Upscaling "
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border_focused())
        .title(title)
        .title_style(theme.title_focused())
        .style(Style::default().bg(theme.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    if has_download {
        // Download progress phase — show download bar with model name, bytes, speed
        let pct = (up.download_batch_bytes as f64 / up.download_batch_total as f64).min(1.0);
        let transfer =
            if let (Some(rate), Some(eta_secs)) = (up.download_rate_bps, up.download_eta_secs) {
                format!(
                    ", {}/s, eta {}",
                    progress::format_bytes_binary(rate),
                    progress::format_eta(eta_secs.ceil() as u64)
                )
            } else {
                String::new()
            };
        let label = if up.download_total_files > 0 {
            format!(
                "[{}/{}] {} [{}/{} total{}]",
                up.download_file_index + 1,
                up.download_total_files,
                up.download_filename,
                progress::format_bytes(up.download_batch_bytes),
                progress::format_bytes(up.download_batch_total),
                transfer,
            )
        } else {
            format!(
                "{} [{}/{} total{}]",
                up.download_filename,
                progress::format_bytes(up.download_batch_bytes),
                progress::format_bytes(up.download_batch_total),
                transfer,
            )
        };

        let gauge = Gauge::default()
            .ratio(pct)
            .label(label)
            .gauge_style(Style::default().fg(theme.warning).bg(theme.progress_empty));

        // Render download bar in the first row, status text in the second
        let row = Rect { height: 1, ..inner };
        frame.render_widget(gauge, row);

        if inner.height > 1 {
            let status_row = Rect {
                y: inner.y + 1,
                height: 1,
                ..inner
            };
            let status = if let Some(ref stage) = up.current_stage {
                stage.clone()
            } else {
                up.download_status_text().to_string()
            };
            let status_text = Paragraph::new(status).style(theme.dim());
            frame.render_widget(status_text, status_row);
        }
    } else {
        // Tile progress phase (or waiting)
        let (tile, total) = app.upscale_tile_progress.unwrap_or((0, 0));
        let (pct, label) = if total > 0 {
            let p = tile as f64 / total as f64;
            (p, format!("Upscaling tile {tile}/{total}"))
        } else if up.current_stage.is_some() {
            // Downloading but no batch data yet (preparing)
            (
                0.0,
                up.current_stage
                    .clone()
                    .unwrap_or_else(|| "Preparing...".to_string()),
            )
        } else if app.server_url.is_some() {
            (0.0, "Processing on server...".to_string())
        } else {
            (0.0, "Loading upscaler model...".to_string())
        };

        let gauge = Gauge::default()
            .ratio(pct.min(1.0))
            .label(label)
            .gauge_style(theme.progress_filled())
            .style(theme.progress_empty());

        frame.render_widget(gauge, inner);
    }
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
                generating_shortcuts(status, app.generate.focus)
            } else if app.generate.focus == crate::app::GenerateFocus::Navigation {
                vec![
                    ("1-6", "Views"),
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
                let neg_label = if app.generate.negative_collapsed {
                    "Neg+"
                } else {
                    "Neg-"
                };
                vec![
                    ("Enter", "Generate"),
                    ("^G", "Generate"),
                    ("^M", "Model"),
                    ("^R", "Seed"),
                    ("Alt+N", neg_label),
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
            ("1-6", "Views"),
            ("Enter", "Select"),
            ("p", "Pull"),
            ("u", "Unload"),
            ("Esc", "Back"),
            ("?", "Help"),
            ("q", "Quit"),
        ],
        View::Queue => vec![
            ("1-6", "Views"),
            ("Esc", "Back"),
            ("?", "Help"),
            ("q", "Quit"),
        ],
        View::Settings => {
            if app.settings.focus == crate::app::SettingsFocus::Appearance {
                vec![
                    ("\u{2190}/\u{2192}", "Theme"),
                    ("j", "Config"),
                    ("Esc", "Back"),
                    ("?", "Help"),
                    ("q", "Quit"),
                ]
            } else {
                vec![
                    ("j/k", "Navigate"),
                    ("+/-", "Adjust"),
                    ("Enter", "Edit"),
                    ("Esc", "Back"),
                    ("?", "Help"),
                    ("q", "Quit"),
                ]
            }
        }
        View::Script => vec![
            ("j/k", "Navigate"),
            ("a/d", "Add/Del"),
            ("t", "Transition"),
            ("i", "Prompt"),
            ("f", "Frames"),
            ("Esc", "Back"),
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

/// Build the status-bar shortcut list for the in-flight generation state.
///
/// When focus is still in the Prompt or Negative textarea (the typical
/// post-submit state), plain `q` is routed to `TextArea::input` rather
/// than the quit action — it just types `q` into the prompt. Advertising
/// `q Quit` in that state is misleading, so the hint is dropped. Users
/// can still bail out via `Esc` (unfocus) or `Ctrl+C` (hard quit), both
/// of which work from any focus.
pub(crate) fn generating_shortcuts(
    status: &str,
    focus: crate::app::GenerateFocus,
) -> Vec<(&str, &str)> {
    let mut v = vec![("", status), ("Alt+1-6", "Views"), ("Esc", "Unfocus")];
    if !matches!(
        focus,
        crate::app::GenerateFocus::Prompt | crate::app::GenerateFocus::NegativePrompt
    ) {
        v.push(("q", "Quit"));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::GenerateFocus;

    #[test]
    fn generating_shortcuts_hides_q_quit_in_prompt_focus() {
        // Codex P3 reproducer: `q` while the prompt textarea holds focus
        // just types a literal `q` because the textarea bypass list
        // doesn't include it. Advertising "q Quit" in that state is a
        // lie — the hint must be suppressed.
        for focus in [GenerateFocus::Prompt, GenerateFocus::NegativePrompt] {
            let entries = generating_shortcuts("Generating...", focus);
            assert!(
                !entries.iter().any(|(k, _)| *k == "q"),
                "q Quit must not be advertised while focus={:?}",
                focus
            );
            assert!(
                entries.iter().any(|(k, _)| *k == "Esc"),
                "Esc Unfocus must stay visible so users can escape into navigation"
            );
        }
    }

    #[test]
    fn generating_shortcuts_shows_q_quit_in_navigation_focus() {
        // In Navigation / Parameters focus `q` bypasses the textarea and
        // actually quits — the hint is honest in these states.
        for focus in [GenerateFocus::Navigation, GenerateFocus::Parameters] {
            let entries = generating_shortcuts("Generating...", focus);
            assert!(
                entries.iter().any(|(k, d)| *k == "q" && *d == "Quit"),
                "q Quit must be advertised in focus={focus:?} because q does quit there"
            );
        }
    }
}
