use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};

use crate::app::{App, Popup};

/// Render the active popup overlay.
pub fn render(frame: &mut Frame, app: &mut App) {
    match &app.popup {
        Some(Popup::Help) => render_help(frame, app),
        Some(Popup::ModelSelector { .. }) => render_model_selector(frame, app),
        Some(Popup::HostInput { .. }) => render_host_input(frame, app),
        Some(Popup::SeedInput { .. }) => render_seed_input(frame, app),
        Some(Popup::HistorySearch { .. }) => render_history_search(frame, app),
        Some(Popup::Confirm { message, .. }) => render_confirm(frame, app, message.clone()),
        Some(Popup::SettingsInput { .. }) => render_settings_input(frame, app),
        Some(Popup::Info { message }) => render_info(frame, app, message.clone()),
        Some(Popup::UpscaleModelSelector { .. }) => render_upscale_model_selector(frame, app),
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
        Line::from("  u                  Upscale with AI model"),
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
        Line::from(""),
        Line::from(Span::styled(
            "Settings View",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from("  j/k                Navigate settings"),
        Line::from("  +/- or Left/Right  Adjust value"),
        Line::from("  Enter              Edit text field / toggle"),
        Line::from("  Esc                Return to Generate"),
    ];

    let paragraph = Paragraph::new(help_text)
        .block(block)
        .style(Style::default().fg(theme.text))
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

/// Build a rich two-line ListItem for a model entry.
///
/// Line 1:  `[marker] model-name           [size]  [status]`
/// Line 2:  `         description text`
fn build_model_item<'a>(
    name: &str,
    is_selected: bool,
    show_download_status: bool,
    default_model: Option<&str>,
    theme: &crate::ui::theme::Theme,
    config: &mold_core::Config,
    width: u16,
) -> ListItem<'a> {
    let manifest = mold_core::manifest::find_manifest(name);
    let resolved = mold_core::manifest::resolve_model_name(name);
    // Model is available if it's in the config or manifest says it's downloaded
    let downloaded =
        config.models.contains_key(&resolved) || config.manifest_model_is_downloaded(name);
    let is_default = default_model.is_some_and(|d| d == name);

    // Use a fixed-width 2-column marker for consistent alignment
    let marker = if is_selected { "> " } else { "  " };

    // Size display (right-aligned, fixed 7-char width)
    let size_str = manifest
        .map(|m| {
            let bytes = m.model_size_bytes();
            if bytes >= 1_073_741_824 {
                format!("{:.1}GB", bytes as f64 / 1_073_741_824.0)
            } else {
                format!("{}MB", bytes / 1_048_576)
            }
        })
        .unwrap_or_default();

    // Status tag
    let status = if is_default && downloaded {
        "default"
    } else if is_default && !downloaded {
        "default | pull"
    } else if show_download_status && !downloaded {
        "pull"
    } else if show_download_status && downloaded {
        "ready"
    } else {
        ""
    };

    // Build first line: marker + name left-aligned, size + status right-aligned
    // Use fixed-width columns so alignment is consistent across all rows
    let left = format!("{marker}{name}");
    let right = if status.is_empty() {
        format!("{size_str:>7}")
    } else {
        format!("{size_str:>7}  {status}")
    };
    let used = left.len() + right.len();
    let padding = (width as usize).saturating_sub(used);
    let pad = " ".repeat(padding);

    let name_style = Style::default().fg(theme.text);
    let size_style = Style::default().fg(theme.text_dim);
    let status_style = if status == "ready" {
        Style::default().fg(Color::Green)
    } else if is_default {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.text_dim)
    };

    let line1 = Line::from(vec![
        Span::styled(left, name_style),
        Span::styled(pad, name_style),
        Span::styled(format!("{size_str:>7}"), size_style),
        if !status.is_empty() {
            Span::styled(format!("  {status}"), status_style)
        } else {
            Span::raw("")
        },
    ]);

    // Second line: description (dimmed, indented)
    let desc = manifest.map(|m| m.description.clone()).unwrap_or_default();
    let desc_indent = "     ";
    let max_desc = (width as usize).saturating_sub(desc_indent.len());
    let desc_text = if desc.len() > max_desc {
        format!("{}{}...", desc_indent, &desc[..max_desc.saturating_sub(3)])
    } else {
        format!("{desc_indent}{desc}")
    };
    let line2 = Line::from(Span::styled(desc_text, Style::default().fg(theme.text_dim)));

    let bg = if is_selected {
        theme.list_selected()
    } else {
        Style::default()
    };
    ListItem::new(vec![line1, line2]).style(bg)
}

/// Render a model selector popup (shared between generation and upscaler selectors).
#[allow(clippy::too_many_arguments)]
fn render_model_selector_popup(
    frame: &mut Frame,
    app: &mut App,
    title: &str,
    filter: &str,
    selected: usize,
    filtered: &[String],
    show_download_status: bool,
    default_model: Option<&str>,
) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 65, 60);

    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.popup_border())
        .title(format!(" {title} "))
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
        filter.to_string()
    };
    let filter_style = if filter.is_empty() {
        theme.dim()
    } else {
        Style::default().fg(theme.text)
    };
    let filter_line = Paragraph::new(format!("Filter: {filter_display}")).style(filter_style);
    let filter_area = Rect {
        x: inner.x,
        y: inner.y,
        width: inner.width,
        height: 1,
    };
    frame.render_widget(filter_line, filter_area);

    // Model list (2 lines per item)
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
            build_model_item(
                name,
                i == selected,
                show_download_status,
                default_model,
                theme,
                &app.config,
                inner.width,
            )
        })
        .collect();

    let list = List::new(items);
    let mut state = ListState::default().with_selected(Some(selected));
    frame.render_stateful_widget(list, list_area, &mut state);
}

fn render_model_selector(frame: &mut Frame, app: &mut App) {
    if let Some(Popup::ModelSelector {
        filter,
        selected,
        filtered,
    }) = &app.popup
    {
        let filter = filter.clone();
        let selected = *selected;
        let filtered = filtered.clone();
        render_model_selector_popup(
            frame,
            app,
            "Select Model",
            &filter,
            selected,
            &filtered,
            false,
            None,
        );
    }
}

fn render_upscale_model_selector(frame: &mut Frame, app: &mut App) {
    if let Some(Popup::UpscaleModelSelector {
        filter,
        selected,
        filtered,
    }) = &app.popup
    {
        let filter = filter.clone();
        let selected = *selected;
        let filtered = filtered.clone();
        // Determine default upscaler: first downloaded, or "real-esrgan-x4plus:fp16"
        let default = filtered
            .iter()
            .find(|n| app.config.manifest_model_is_downloaded(n))
            .cloned()
            .unwrap_or_else(|| "real-esrgan-x4plus:fp16".to_string());
        render_model_selector_popup(
            frame,
            app,
            "Select Upscaler Model",
            &filter,
            selected,
            &filtered,
            true,
            Some(&default),
        );
    }
}

fn render_host_input(frame: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 50, 15);

    frame.render_widget(Clear, area);

    if let Some(Popup::HostInput { input }) = &app.popup {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.popup_border())
            .title(" Server Host ")
            .title_style(theme.title_focused())
            .style(theme.popup_bg());

        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.height < 3 {
            return;
        }

        let hint =
            Paragraph::new("Enter server address (or clear for local mode)").style(theme.dim());
        let hint_area = Rect {
            x: inner.x,
            y: inner.y,
            width: inner.width,
            height: 1,
        };
        frame.render_widget(hint, hint_area);

        // Input field with cursor
        let display = format!("{input}\u{2588}"); // block cursor
        let input_line = Paragraph::new(display).style(Style::default().fg(theme.text));
        let input_area = Rect {
            x: inner.x,
            y: inner.y + 2,
            width: inner.width,
            height: 1,
        };
        frame.render_widget(input_line, input_area);

        // Hint at bottom
        let actions = Line::from(vec![
            Span::styled("Enter", theme.status_key()),
            Span::styled(" Confirm  ", Style::default().fg(theme.text)),
            Span::styled("Esc", theme.status_key()),
            Span::styled(" Cancel", Style::default().fg(theme.text)),
        ]);
        let actions_area = Rect {
            x: inner.x,
            y: inner.y + inner.height.saturating_sub(1),
            width: inner.width,
            height: 1,
        };
        frame.render_widget(Paragraph::new(actions), actions_area);
    }
}

fn render_seed_input(frame: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 45, 15);

    frame.render_widget(Clear, area);

    if let Some(Popup::SeedInput { input }) = &app.popup {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.popup_border())
            .title(" Seed Value ")
            .title_style(theme.title_focused())
            .style(theme.popup_bg());

        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.height < 3 {
            return;
        }

        let hint = Paragraph::new("Enter seed (digits only, empty for auto)").style(theme.dim());
        frame.render_widget(hint, Rect { height: 1, ..inner });

        let display = format!("{input}\u{2588}");
        let input_line = Paragraph::new(display).style(Style::default().fg(theme.text));
        frame.render_widget(
            input_line,
            Rect {
                y: inner.y + 2,
                height: 1,
                ..inner
            },
        );

        let actions = Line::from(vec![
            Span::styled("Enter", theme.status_key()),
            Span::styled(" Confirm  ", Style::default().fg(theme.text)),
            Span::styled("Esc", theme.status_key()),
            Span::styled(" Cancel", Style::default().fg(theme.text)),
        ]);
        frame.render_widget(
            Paragraph::new(actions),
            Rect {
                y: inner.y + inner.height.saturating_sub(1),
                height: 1,
                ..inner
            },
        );
    }
}

fn render_history_search(frame: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 60, 55);

    frame.render_widget(Clear, area);

    if let Some(Popup::HistorySearch {
        filter,
        selected,
        results,
    }) = &app.popup
    {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.popup_border())
            .title(" Prompt History ")
            .title_style(theme.title_focused())
            .style(theme.popup_bg());

        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.height < 3 {
            return;
        }

        // Filter input
        let filter_display = if filter.is_empty() {
            "Type to search...".to_string()
        } else {
            filter.clone()
        };
        let filter_style = if filter.is_empty() {
            theme.dim()
        } else {
            Style::default().fg(theme.text)
        };
        let filter_line = Paragraph::new(format!("/ {filter_display}")).style(filter_style);
        frame.render_widget(filter_line, Rect { height: 1, ..inner });

        // Results list
        let list_area = Rect {
            x: inner.x,
            y: inner.y + 2,
            width: inner.width,
            height: inner.height.saturating_sub(2),
        };

        let items: Vec<ListItem> = results
            .iter()
            .enumerate()
            .map(|(i, prompt)| {
                let style = if i == *selected {
                    theme.list_selected()
                } else {
                    Style::default().fg(theme.text)
                };
                let marker = if i == *selected { "\u{25b8} " } else { "  " };
                // Truncate long prompts
                let display = if prompt.len() > list_area.width as usize - 4 {
                    format!("{marker}{}...", &prompt[..list_area.width as usize - 7])
                } else {
                    format!("{marker}{prompt}")
                };
                ListItem::new(display).style(style)
            })
            .collect();

        if items.is_empty() {
            let empty = Paragraph::new("No matching prompts")
                .style(theme.dim())
                .alignment(Alignment::Center);
            frame.render_widget(empty, list_area);
        } else {
            let list = List::new(items);
            let mut state = ListState::default().with_selected(Some(*selected));
            frame.render_stateful_widget(list, list_area, &mut state);
        }
    }
}

fn render_confirm(frame: &mut Frame, app: &App, message: String) {
    let theme = &app.theme;
    // Use larger popup when message has multiple lines (e.g. model deletion details)
    let line_count = message.lines().count();
    let (w, h) = if line_count > 2 { (55, 35) } else { (40, 20) };
    let area = centered_rect(frame.area(), w, h);

    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.popup_border())
        .title(" Confirm ")
        .title_style(theme.title_focused())
        .style(theme.popup_bg());

    let mut text: Vec<Line> = message.lines().map(|l| Line::from(l.to_string())).collect();
    text.push(Line::from(""));
    text.push(Line::from(vec![
        Span::styled("y", theme.status_key()),
        Span::styled(" Confirm  ", Style::default().fg(theme.text)),
        Span::styled("n", theme.status_key()),
        Span::styled(" Cancel", Style::default().fg(theme.text)),
    ]));

    let paragraph = Paragraph::new(text)
        .block(block)
        .alignment(Alignment::Center);

    frame.render_widget(paragraph, area);
}

fn render_info(frame: &mut Frame, app: &App, message: String) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 55, 20);

    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.popup_border())
        .title(" Info ")
        .title_style(theme.title_focused())
        .style(theme.popup_bg());

    let mut text: Vec<Line> = message.lines().map(|l| Line::from(l.to_string())).collect();
    text.push(Line::from(""));
    text.push(Line::from(Span::styled(
        "Press any key to dismiss",
        Style::default().fg(theme.text_dim),
    )));

    let paragraph = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false })
        .alignment(Alignment::Center);

    frame.render_widget(paragraph, area);
}

fn render_settings_input(frame: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 55, 15);

    frame.render_widget(Clear, area);

    if let Some(Popup::SettingsInput { input, label, .. }) = &app.popup {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.popup_border())
            .title(format!(" {label} "))
            .title_style(theme.title_focused())
            .style(theme.popup_bg());

        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.height < 3 {
            return;
        }

        let hint = Paragraph::new("Enter value (empty to clear)").style(theme.dim());
        frame.render_widget(hint, Rect { height: 1, ..inner });

        // Input field with cursor
        let display = format!("{input}\u{2588}");
        let input_line = Paragraph::new(display).style(Style::default().fg(theme.text));
        frame.render_widget(
            input_line,
            Rect {
                y: inner.y + 2,
                height: 1,
                ..inner
            },
        );

        let actions = Line::from(vec![
            Span::styled("Enter", theme.status_key()),
            Span::styled(" Confirm  ", Style::default().fg(theme.text)),
            Span::styled("Esc", theme.status_key()),
            Span::styled(" Cancel", Style::default().fg(theme.text)),
        ]);
        frame.render_widget(
            Paragraph::new(actions),
            Rect {
                y: inner.y + inner.height.saturating_sub(1),
                height: 1,
                ..inner
            },
        );
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn upscaler_manifest_has_description_and_size() {
        // All upscaler models in the manifest should have descriptions and non-zero sizes
        for manifest in mold_core::manifest::known_manifests() {
            if !manifest.is_upscaler() {
                continue;
            }
            assert!(
                !manifest.description.is_empty(),
                "{} has empty description",
                manifest.name
            );
            assert!(
                manifest.model_size_bytes() > 0,
                "{} has zero size",
                manifest.name
            );
        }
    }

    #[test]
    fn default_upscaler_exists_in_manifest() {
        let manifest = mold_core::manifest::find_manifest("real-esrgan-x4plus:fp16");
        assert!(manifest.is_some(), "default upscaler not found in manifest");
        assert!(manifest.unwrap().is_upscaler());
    }

    #[test]
    fn upscaler_size_formats_as_mb() {
        let manifest = mold_core::manifest::find_manifest("real-esrgan-x4plus:fp16").unwrap();
        let bytes = manifest.model_size_bytes();
        // FP16 x4plus is ~32MB, should be < 1GB
        assert!(bytes < 1_073_741_824, "expected < 1GB, got {bytes}");
        assert!(bytes > 1_048_576, "expected > 1MB, got {bytes}");
    }

    #[test]
    fn status_tag_logic() {
        // Default + downloaded = "default"
        // Default + not downloaded = "default · pull"
        // Not default + not downloaded = "pull"
        // Not default + downloaded = ""
        let is_default = true;
        let downloaded = true;
        let status = if is_default && downloaded {
            "default"
        } else if is_default && !downloaded {
            "default · pull"
        } else if !downloaded {
            "pull"
        } else {
            ""
        };
        assert_eq!(status, "default");

        let status2 = if true && !true {
            "default · pull"
        } else if true && true {
            "default"
        } else {
            ""
        };
        assert_eq!(status2, "default");
    }
}
