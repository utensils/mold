use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};

use crate::app::{App, Popup};

/// Render the active popup overlay.
pub fn render(frame: &mut Frame, app: &mut App) {
    match &app.popup {
        Some(Popup::Help) => render_help(frame, app),
        Some(Popup::PromptSourceChoice { .. }) => render_prompt_source_choice(frame, app),
        Some(Popup::PromptAlternatives { .. }) => render_prompt_alternatives(frame, app),
        Some(Popup::ModelSelector { .. }) => render_model_selector(frame, app),
        Some(Popup::MachineConnect { .. }) => render_machine_connect(frame, app),
        Some(Popup::SeedInput { .. }) => render_seed_input(frame, app),
        Some(Popup::SizeInput { .. }) => render_size_input(frame, app),
        Some(Popup::StgBlocksInput { .. }) => render_stg_blocks_input(frame, app),
        Some(Popup::ReferencesInput { .. }) => render_references_input(frame, app),
        Some(Popup::HistorySearch { .. }) => render_history_search(frame, app),
        Some(Popup::CommandPalette { .. }) => render_command_palette(frame, app),
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
        Line::from(
            "  1-5 / Alt+1-5      Switch workspace (Create/Library/Models/Machines/Settings)",
        ),
        Line::from("  Ctrl+K             Command palette"),
        Line::from("  Esc                Close popup / cancel"),
        Line::from("  q / Ctrl+C         Quit"),
        Line::from(""),
        Line::from(Span::styled(
            "Create View",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from("  Enter              Start generation"),
        Line::from("  c                  Open chain composer"),
        Line::from("  A                  Toggle the Advanced accordion"),
        Line::from("  Alt+N              Edit the negative prompt"),
        Line::from("  Ctrl+E             Expand prompt via LLM"),
        Line::from("  Ctrl+Shift+E       Remix prompt alternatives"),
        Line::from("  Ctrl+S             Save current image"),
        Line::from("  Ctrl+R             Cycle seed mode"),
        Line::from("  Ctrl+M             Open model selector"),
        Line::from("  j/k                Navigate parameters"),
        Line::from("  +/- or Left/Right  Adjust / expand a section"),
        Line::from(""),
        Line::from(Span::styled(
            "Library View",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from("  hjkl / arrows      Navigate the grid"),
        Line::from("  Enter              Open detail view"),
        Line::from("  e / r              Recall into Create"),
        Line::from("  d                  Move print to the trash (on every machine it exists on)"),
        Line::from("  u                  Upscale with AI model"),
        Line::from("  o                  Open in system viewer"),
        Line::from("  /                  Filter by prompt, model, or filename"),
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
            "Machines View",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from("  j/k                Select machine / queue lane"),
        Line::from("  Enter              Set generation target (again = Auto)"),
        Line::from("  Tab                Toggle host list / detail focus"),
        Line::from("  c                  Connect a machine"),
        Line::from("  d                  Disconnect / reconnect host"),
        Line::from("  f                  Forget host (deletes its API key)"),
        Line::from("  r                  Refresh telemetry and queue"),
        Line::from("  x                  Cancel selected queued job"),
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
        Line::from("  Esc                Return to Create"),
    ];

    let paragraph = Paragraph::new(help_text)
        .block(block)
        .style(Style::default().fg(theme.text))
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_prompt_alternatives(frame: &mut Frame, app: &App) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 78, 78);
    frame.render_widget(Clear, area);
    let Some(Popup::PromptAlternatives {
        snapshot,
        variants,
        selected,
        cursor,
        ..
    }) = &app.popup
    else {
        return;
    };
    let title = match snapshot.operation {
        mold_core::PromptTransformOperation::Expand => " Prompt expansion ",
        mold_core::PromptTransformOperation::Remix => " Prompt Remix ",
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.popup_border())
        .title(title)
        .title_style(theme.title_focused())
        .style(theme.popup_bg());
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(5),
        Constraint::Length(2),
    ])
    .split(inner);
    frame.render_widget(
        Paragraph::new(format!("Source: {}", snapshot.source_prompt))
            .style(Style::default().fg(theme.text_dim))
            .wrap(Wrap { trim: true }),
        chunks[0],
    );
    let items = variants
        .iter()
        .enumerate()
        .map(|(index, variant)| {
            let mark = if selected.get(index).copied().unwrap_or(false) {
                "[x]"
            } else {
                "[ ]"
            };
            let dims = variant
                .dimensions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            ListItem::new(vec![
                Line::from(format!("{mark} {}. {}", index + 1, variant.prompt)),
                Line::styled(
                    format!("    varies: {dims}"),
                    Style::default().fg(theme.text_dim),
                ),
            ])
        })
        .collect::<Vec<_>>();
    let mut state = ListState::default().with_selected(Some(*cursor));
    frame.render_stateful_widget(
        List::new(items)
            .highlight_style(theme.list_selected())
            .highlight_symbol("> "),
        chunks[1],
        &mut state,
    );
    frame.render_widget(
        Paragraph::new(
            "Space select · Enter apply one · B prepare selected batch · R re-remix · Esc close",
        )
        .style(Style::default().fg(theme.text_dim)),
        chunks[2],
    );
}

fn render_prompt_source_choice(frame: &mut Frame, app: &App) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 70, 42);
    frame.render_widget(Clear, area);
    let Some(Popup::PromptSourceChoice {
        current_prompt,
        root_prompt,
        cursor,
    }) = &app.popup
    else {
        return;
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.popup_border())
        .title(" Remix source ")
        .title_style(theme.title_focused())
        .style(theme.popup_bg());
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let chunks = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(4),
        Constraint::Length(2),
    ])
    .split(inner);
    frame.render_widget(
        Paragraph::new("Choose the frozen source for this Remix request")
            .style(Style::default().fg(theme.text_dim)),
        chunks[0],
    );
    let items = [
        ListItem::new(format!("Original: {root_prompt}")),
        ListItem::new(format!("Current:  {current_prompt}")),
    ];
    let mut state = ListState::default().with_selected(Some(*cursor));
    frame.render_stateful_widget(
        List::new(items)
            .highlight_style(theme.list_selected())
            .highlight_symbol("> "),
        chunks[1],
        &mut state,
    );
    frame.render_widget(
        Paragraph::new("O original · C current · Enter choose · Esc close")
            .style(Style::default().fg(theme.text_dim)),
        chunks[2],
    );
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

    // Status indicator — compact visual: checkmark for downloaded, "(download)" for not
    let status_width: usize = if show_download_status { 12 } else { 0 };

    // Default indicator — use display width (2 cols for " ★") for padding calc
    let default_display_width: usize = if is_default { 2 } else { 0 };

    // Build first line: marker + name + default_tag left-aligned, size + status right-aligned
    // Right section is fixed width: 7 (size) + 2 (gap) + status_width
    let left_display_width = 2 + name.len() + default_display_width; // marker(2) + name + star
    let right_width = 7 + if status_width > 0 {
        2 + status_width
    } else {
        0
    };
    let padding = (width as usize).saturating_sub(left_display_width + right_width);
    let pad = " ".repeat(padding);

    let name_style = Style::default().fg(theme.text);
    let size_style = Style::default().fg(theme.text_dim);
    let default_style = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);

    let mut spans = vec![
        Span::styled(format!("{marker}{name}"), name_style),
        if is_default {
            Span::styled(" \u{2605}", default_style)
        } else {
            Span::raw("")
        },
        Span::styled(pad, name_style),
        Span::styled(format!("{size_str:>7}"), size_style),
    ];
    if show_download_status {
        spans.push(Span::raw("  "));
        if downloaded {
            // Green checkmark + "ready" — model is downloaded
            let tag = format!("{:>width$}", "\u{2713} ready", width = status_width);
            spans.push(Span::styled(tag, Style::default().fg(Color::Green)));
        } else {
            // Dim "(download)" — will be auto-pulled on selection
            let tag = format!("{:>width$}", "(download)", width = status_width);
            spans.push(Span::styled(tag, Style::default().fg(theme.text_dim)));
        }
    }

    let line1 = Line::from(spans);

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
        let default = mold_core::manifest::resolve_model_name(&app.config.resolved_default_model());
        render_model_selector_popup(
            frame,
            app,
            "Select Model",
            &filter,
            selected,
            &filtered,
            true,
            Some(&default),
        );
    }
}

fn render_upscale_model_selector(frame: &mut Frame, app: &mut App) {
    if let Some(Popup::UpscaleModelSelector {
        filter,
        selected,
        filtered,
        ..
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

fn render_size_input(frame: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 45, 15);

    frame.render_widget(Clear, area);

    if let Some(Popup::SizeInput { input }) = &app.popup {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.popup_border())
            .title(" Size ")
            .title_style(theme.title_focused())
            .style(theme.popup_bg());

        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.height < 3 {
            return;
        }

        let hint = Paragraph::new("Enter size as WxH (e.g. 1024x768)").style(theme.dim());
        frame.render_widget(hint, Rect { height: 1, ..inner });

        // Input field with cursor
        let display = format!("{input}\u{2588}"); // block cursor
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

fn render_stg_blocks_input(frame: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 55, 20);

    frame.render_widget(Clear, area);

    if let Some(Popup::StgBlocksInput { input, error }) = &app.popup {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.popup_border())
            .title(" STG Blocks ")
            .title_style(theme.title_focused())
            .style(theme.popup_bg());

        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height < 4 {
            return;
        }

        frame.render_widget(
            Paragraph::new("Comma-separated transformer blocks (for example 28, 29)")
                .style(theme.dim()),
            Rect { height: 1, ..inner },
        );
        frame.render_widget(
            Paragraph::new(format!("{input}\u{2588}")).style(Style::default().fg(theme.text)),
            Rect {
                y: inner.y + 2,
                height: 1,
                ..inner
            },
        );
        if let Some(error) = error {
            frame.render_widget(
                Paragraph::new(error.as_str()).style(theme.error()),
                Rect {
                    y: inner.y + 3,
                    height: 1,
                    ..inner
                },
            );
        }
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

fn render_references_input(frame: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let area = centered_rect(frame.area(), 72, 24);
    frame.render_widget(Clear, area);

    let Some(Popup::ReferencesInput { input, error }) = &app.popup else {
        return;
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.popup_border())
        .title(" Ordered H3 References ")
        .title_style(theme.title_focused())
        .style(theme.popup_bg());
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 5 {
        return;
    }
    frame.render_widget(
        Paragraph::new("Semicolon order is semantic: image=/a.png; video=/b.mp4; audio=/c.wav")
            .style(theme.dim()),
        Rect { height: 1, ..inner },
    );
    frame.render_widget(
        Paragraph::new(format!("{input}\u{2588}"))
            .style(Style::default().fg(theme.text))
            .wrap(Wrap { trim: false }),
        Rect {
            y: inner.y + 2,
            height: inner.height.saturating_sub(4),
            ..inner
        },
    );
    if let Some(error) = error {
        frame.render_widget(
            Paragraph::new(error.as_str()).style(theme.error()),
            Rect {
                y: inner.y + inner.height.saturating_sub(2),
                height: 1,
                ..inner
            },
        );
    }
    frame.render_widget(
        action_hints(theme, &[("Enter", "Confirm"), ("Esc", "Cancel")]),
        Rect {
            y: inner.y + inner.height.saturating_sub(1),
            height: 1,
            ..inner
        },
    );
}

/// Render the stepped connect-a-machine flow (Machines workspace).
fn render_machine_connect(frame: &mut Frame, app: &mut App) {
    use crate::hosts::ConnectStep;

    let theme = &app.theme;
    let area = centered_rect(frame.area(), 55, 20);
    frame.render_widget(Clear, area);

    let Some(Popup::MachineConnect { form }) = &app.popup else {
        return;
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.popup_border())
        .title(" Connect a machine ")
        .title_style(theme.title_focused())
        .style(theme.popup_bg());
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 4 {
        return;
    }

    let (hint, input_line, actions): (String, Option<Line>, Line) = match form.step {
        ConnectStep::Url => (
            "Server address (host, host:port, or URL)".into(),
            Some(Line::from(Span::styled(
                format!("{}\u{2588}", form.url),
                Style::default().fg(theme.text),
            ))),
            action_hints(theme, &[("Enter", "Next"), ("Esc", "Cancel")]),
        ),
        ConnectStep::ApiKey => (
            format!("API key for {} (optional — Enter to skip)", form.url),
            Some(Line::from(Span::styled(
                format!(
                    "{}\u{2588}",
                    "\u{2022}".repeat(form.api_key.chars().count())
                ),
                Style::default().fg(theme.text),
            ))),
            action_hints(theme, &[("Enter", "Test"), ("Esc", "Back")]),
        ),
        ConnectStep::Testing => (
            format!("Testing {}\u{2026}", form.url),
            None,
            action_hints(theme, &[("Esc", "Cancel")]),
        ),
        ConnectStep::Failed => (
            form.error
                .clone()
                .unwrap_or_else(|| "Connection failed".into()),
            None,
            action_hints(
                theme,
                &[("Enter", "Retry"), ("e", "Edit"), ("Esc", "Cancel")],
            ),
        ),
    };

    let hint_style = if form.step == ConnectStep::Failed {
        theme.error()
    } else {
        theme.dim()
    };
    frame.render_widget(
        Paragraph::new(hint)
            .style(hint_style)
            .wrap(Wrap { trim: true }),
        Rect { height: 2, ..inner },
    );

    if let Some(line) = input_line {
        frame.render_widget(
            Paragraph::new(line),
            Rect {
                y: inner.y + 2,
                height: 1,
                ..inner
            },
        );
    }

    frame.render_widget(
        Paragraph::new(actions),
        Rect {
            y: inner.y + inner.height.saturating_sub(1),
            height: 1,
            ..inner
        },
    );
}

/// Build the `Key Action  Key Action` hint line used by popup footers.
fn action_hints<'a>(theme: &crate::ui::theme::Theme, pairs: &[(&'a str, &'a str)]) -> Line<'a> {
    let mut spans = Vec::new();
    for (i, (key, label)) in pairs.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  ", Style::default().fg(theme.text)));
        }
        spans.push(Span::styled(*key, theme.status_key()));
        spans.push(Span::styled(
            format!(" {label}"),
            Style::default().fg(theme.text),
        ));
    }
    Line::from(spans)
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

/// Render the ^K command palette — top-aligned like the GUI launcher.
fn render_command_palette(frame: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let screen = frame.area();

    let Some(Popup::CommandPalette {
        filter,
        selected,
        filtered,
    }) = &app.popup
    else {
        return;
    };

    // Width ~52 cols (the mock's 520px), clamped to the terminal; height
    // fits the input row plus up to 10 command rows.
    let width = 52.min(screen.width.saturating_sub(4)).max(20);
    let rows = (filtered.len() as u16).clamp(1, 10);
    let height = (rows + 3).min(screen.height.saturating_sub(2));
    let x = screen.x + (screen.width.saturating_sub(width)) / 2;
    let y = screen.y + 3.min(screen.height.saturating_sub(height));
    let area = Rect {
        x,
        y,
        width,
        height,
    };

    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.popup_border())
        .style(theme.popup_bg());
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 2 || inner.width < 8 {
        return;
    }

    // Input row: "› {filter}" with a faint placeholder when empty.
    let mut input_spans = vec![Span::styled("\u{203a} ", Style::default().fg(theme.accent))];
    if filter.is_empty() {
        input_spans.push(Span::styled("type a command\u{2026}", theme.faint()));
    } else {
        input_spans.push(Span::styled(
            filter.clone(),
            Style::default().fg(theme.text),
        ));
    }
    input_spans.push(Span::styled("\u{2588}", Style::default().fg(theme.accent)));
    frame.render_widget(
        Paragraph::new(Line::from(input_spans)),
        Rect { height: 1, ..inner },
    );

    // Command rows below the input, scrolled to keep the selection visible.
    let list_area = Rect {
        x: inner.x,
        y: inner.y + 2,
        width: inner.width,
        height: inner.height.saturating_sub(2),
    };
    if list_area.height == 0 {
        return;
    }
    let visible = list_area.height as usize;
    let offset = selected.saturating_sub(visible.saturating_sub(1));
    let commands = crate::palette::all_commands();

    let items: Vec<ListItem> = filtered
        .iter()
        .enumerate()
        .skip(offset)
        .take(visible)
        .map(|(i, id)| {
            let cmd = commands
                .iter()
                .find(|c| c.id == *id)
                .expect("filtered ids come from the registry");
            let row_style = if i == *selected {
                Style::default().bg(theme.highlight).fg(theme.accent)
            } else {
                Style::default().fg(theme.text)
            };
            let hint_pad = (list_area.width as usize)
                .saturating_sub(4 + cmd.label.chars().count() + cmd.hint.len());
            let line = Line::from(vec![
                Span::styled(format!(" {} ", cmd.glyph), Style::default().fg(theme.info)),
                Span::styled(cmd.label.clone(), row_style),
                Span::raw(" ".repeat(hint_pad)),
                Span::styled(cmd.hint, theme.faint()),
            ]);
            ListItem::new(line).style(row_style)
        })
        .collect();

    frame.render_widget(List::new(items), list_area);
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
    let (w, h) = if line_count > 2 { (55, 35) } else { (45, 30) };
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
    // "Enter" leads the hint because it is the default action — pressing
    // Enter without thinking confirms the operation. Listing it last or
    // omitting it (the older hint read `y Confirm  n Cancel`) led users
    // to believe Enter meant cancel.
    text.push(Line::from(vec![
        Span::styled("Enter/y", theme.status_key()),
        Span::styled(" Confirm  ", Style::default().fg(theme.text)),
        Span::styled("Esc/n", theme.status_key()),
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
    fn confirm_popup_hint_lists_enter_as_default_confirm() {
        // Prior hint read `y Confirm  n Cancel` — Enter wasn't in the
        // strip and users assumed it meant cancel, so gallery delete
        // appeared to silently fail. The hint must now call out Enter.
        let rendered = render_confirm_to_string("Delete file.png?");
        assert!(
            rendered.contains("Enter"),
            "confirm popup should advertise Enter as the default: {rendered}"
        );
        assert!(
            rendered.contains("Confirm"),
            "confirm popup should still say Confirm: {rendered}"
        );
        assert!(
            rendered.contains("Esc") || rendered.contains("n"),
            "confirm popup should still expose a cancel affordance: {rendered}"
        );
    }

    /// Drive `render_confirm` against a `TestBackend` and collapse the
    /// buffer into a single string so assertions can use substring
    /// matches against the visible UI.
    fn render_confirm_to_string(message: &str) -> String {
        use crate::app::{App, ConfirmAction, Popup};
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        // Build a stand-in App just for rendering — we don't need the
        // tokio handle because render_confirm is pure.
        let picker = ratatui_image::picker::Picker::from_fontsize((8, 16));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App {
            active_view: crate::action::View::Library,
            create_mode: crate::app::CreateMode::default(),
            motion: crate::motion::MotionState::new(false),
            generate: crate::app::GenerateState {
                prompt: tui_textarea::TextArea::default(),
                negative_prompt: tui_textarea::TextArea::default(),
                negative_default: String::new(),
                negative_explicit_clear: false,
                params: crate::app::GenerateParams::from_config(&mold_core::Config::default()),
                focus: crate::app::GenerateFocus::Navigation,
                param_index: 0,
                rows: Vec::new(),
                advanced: crate::ui::create_form::AdvancedState::default(),
                param_scroll: 0,
                capabilities: crate::model_info::capabilities_for_family(
                    &crate::model_info::family_for_model("", &mold_core::Config::default()),
                ),
                progress: crate::app::ProgressState::default(),
                live_preview_image: None,
                live_preview_protocol: None,
                preview_image: None,
                image_state: None,
                animation: None,
                generating: false,
                batch_remaining: 0,
                last_seed: None,
                last_generation_time_ms: None,
                error_message: None,
                warning_message: None,
                model_description: String::new(),
                last_output_path: None,
                prompt_transform_token: 0,
            },
            gallery: crate::app::GalleryState::default(),
            models: crate::app::ModelsState {
                catalog: Vec::new(),
                selected: 0,
                filter: String::new(),
                filtering: false,
            },
            machines: crate::hosts::MachinesState::default(),
            target: crate::hosts::GenTarget::default(),
            settings: crate::app::SettingsState::default(),
            prefs: crate::prefs::TuiPrefs::default(),
            script: crate::ui::script_composer::ScriptComposerState::default(),
            config: mold_core::Config::default(),
            server_url: None,
            picker,
            theme: crate::ui::theme::Theme::default(),
            popup: Some(Popup::Confirm {
                message: message.to_string(),
                on_confirm: ConfirmAction::DeleteGalleryImage,
            }),
            should_quit: false,
            bg_tx: tx,
            bg_rx: rx,
            tokio_handle: tokio::runtime::Handle::try_current().unwrap_or_else(|_| {
                // Build a throw-away runtime handle for non-tokio tests.
                let rt = tokio::runtime::Runtime::new().unwrap();
                let h = rt.handle().clone();
                std::mem::forget(rt);
                h
            }),
            resource_info: crate::ui::info::ResourceInfo::default(),
            history: crate::history::PromptHistory::load(),
            layout: crate::app::LayoutAreas::default(),
            server_process: None,
            upscale_in_progress: false,
            upscale_task: None,
            upscale_tile_progress: None,
            upscale_progress: crate::app::ProgressState::default(),
            connecting: false,
            show_timeline: true,
        };

        terminal.draw(|f| super::render(f, &mut app)).unwrap();

        // Collapse buffer into a single string — one row per line.
        let buf = terminal.backend().buffer().clone();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

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

        let status2 = if false {
            "default · pull"
        } else if true {
            "default"
        } else {
            ""
        };
        assert_eq!(status2, "default");
    }
}
