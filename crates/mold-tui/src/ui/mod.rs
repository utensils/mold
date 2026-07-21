pub mod chrome;
pub mod create_form;
pub mod gallery;
pub mod generate;
pub mod info;
pub mod machines;
pub mod models;
pub mod param_form;
pub mod popup;
pub mod preview;
pub mod progress;
pub mod script_composer;
pub mod settings;
pub mod theme;
pub mod timeline;
pub mod widgets;

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Gauge, Paragraph};

use crate::action::View;
use crate::app::App;

/// Top-level render function — draws the frame chrome and delegates to the active view.
pub fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    let theme = &app.theme;

    // Fill background
    frame.render_widget(Block::default().style(theme.base()), area);

    // Main layout: tab strip + content + activity strip + status bar.
    // Total chrome stays 4 rows (was 3-row tab bar + 1-row status), so
    // per-view content heights are unchanged by the Studio chrome.
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Tab strip: labels + underline
            Constraint::Min(10),   // Content area
            Constraint::Length(1), // Activity strip
            Constraint::Length(1), // Status bar
        ])
        .split(area);

    // Store layout areas for mouse hit-testing and motion effects.
    app.layout.tab_bar = layout[0];
    app.layout.content = layout[1];
    app.layout.activity = layout[2];

    // ── Tab strip ───────────────────────────────────────────────
    chrome::render_tab_strip(frame, app, layout[0]);

    // ── Content ─────────────────────────────────────────────────
    match app.active_view {
        View::Create if app.create_mode == crate::app::CreateMode::Chain => {
            script_composer::render(frame, &app.script, layout[1], &app.theme)
        }
        View::Create => generate::render(frame, app, layout[1]),
        View::Library => {
            gallery::render(frame, app, layout[1]);
            // Upscale progress bar overlay at bottom of gallery area
            if app.upscale_in_progress {
                render_upscale_progress(frame, app, layout[1]);
            }
        }
        View::Models => models::render(frame, app, layout[1]),
        View::Machines => machines::render(frame, app, layout[1]),
        View::Settings => settings::render(frame, app, layout[1]),
    }

    // ── Activity strip + status bar ─────────────────────────────
    chrome::render_activity_strip(frame, app, layout[2]);
    chrome::render_status_bar(frame, app, layout[3]);

    // ── Popup overlay ───────────────────────────────────────────
    if app.popup.is_some() {
        popup::render(frame, app);
    }
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

/// Build the per-workspace status-bar hint table. Rendering lives in
/// [`chrome::render_status_bar`]; the table stays here beside its
/// contract tests.
pub(crate) fn status_shortcuts(app: &App) -> Vec<(String, String)> {
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
        View::Create if app.create_mode == crate::app::CreateMode::Chain => vec![
            ("j/k", "Navigate"),
            ("a/d", "Add/Del"),
            ("t", "Transition"),
            ("i", "Prompt"),
            ("f", "Frames"),
            ("Esc", "Back"),
        ],
        View::Create => {
            if app.generate.generating {
                let status = if app.generate.progress.is_downloading() {
                    app.generate.progress.download_status_text()
                } else {
                    "Generating..."
                };
                generating_shortcuts(status, app.generate.focus)
            } else if app.generate.focus == crate::app::GenerateFocus::Navigation {
                vec![
                    ("^K", "Commands"),
                    ("1-5", "Workspace"),
                    ("Enter", "Edit"),
                    ("c", "Chain"),
                    ("A", "Advanced"),
                    ("?", "Help"),
                    ("q", "Quit"),
                ]
            } else if app.generate.focus == crate::app::GenerateFocus::Parameters {
                vec![
                    ("Enter", "Edit"),
                    ("+/-", "Adjust"),
                    ("A", "Advanced"),
                    ("^G", "Generate"),
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
                    ("Alt+N", "Negative"),
                    ("Tab", "Focus"),
                    ("Esc", "Nav"),
                ]
            }
        }
        View::Library => {
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
            ("^K", "Commands"),
            ("1-5", "Workspace"),
            ("Enter", "Select"),
            ("p", "Pull"),
            ("u", "Unload"),
            ("Esc", "Back"),
            ("?", "Help"),
            ("q", "Quit"),
        ],
        View::Machines => vec![
            ("^K", "Commands"),
            ("j/k", "Select"),
            ("Enter", "Target"),
            ("c", "Connect"),
            ("d", "Forget"),
            ("r", "Refresh"),
            ("Esc", "Back"),
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
    };

    shortcuts
        .into_iter()
        .map(|(k, d)| (k.to_string(), d.to_string()))
        .collect()
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
    let mut v = vec![("", status), ("Alt+1-5", "Workspace"), ("Esc", "Unfocus")];
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
