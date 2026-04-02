use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui_image::picker::ProtocolType;
use ratatui_image::{Image, Resize, StatefulImage};

use crate::app::{App, GalleryViewMode};

const CELL_W: u16 = 24;
const CELL_H: u16 = 14;

/// Compute a centered sub-rect for an image within the thumbnail area.
///
/// `ratatui-image` pads fitted images from the top-left of the render rect. To
/// make gallery tiles look balanced, we compute the fitted rect ourselves and
/// render into that centered region instead.
///
/// Halfblocks terminals encode two image rows into one terminal row, so the
/// effective character height is half the reported font height for aspect-fit
/// purposes.
fn centered_thumb_rect(
    area: Rect,
    img_w: u32,
    img_h: u32,
    font_size: (u16, u16),
    protocol: ProtocolType,
) -> Rect {
    if area.width == 0 || area.height == 0 || img_w == 0 || img_h == 0 {
        return area;
    }

    let fw = font_size.0.max(1) as f64;
    let fh = match protocol {
        ProtocolType::Halfblocks => (font_size.1.max(2) / 2).max(1) as f64,
        _ => font_size.1.max(1) as f64,
    };

    // Convert image pixel dimensions to terminal cell units.
    let img_cols = img_w as f64 / fw;
    let img_rows = img_h as f64 / fh;

    // Scale to fit within area, preserving aspect ratio.
    let scale = (area.width as f64 / img_cols).min(area.height as f64 / img_rows);

    let used_w = (img_cols * scale).ceil().min(area.width as f64) as u16;
    let used_h = (img_rows * scale).ceil().min(area.height as f64) as u16;

    let offset_x = area.width.saturating_sub(used_w) / 2;
    let offset_y = area.height.saturating_sub(used_h) / 2;

    Rect::new(area.x + offset_x, area.y + offset_y, used_w, used_h)
}

fn center_rect(area: Rect, used_w: u16, used_h: u16) -> Rect {
    let width = used_w.min(area.width);
    let height = used_h.min(area.height);
    let offset_x = area.width.saturating_sub(width) / 2;
    let offset_y = area.height.saturating_sub(height) / 2;
    Rect::new(area.x + offset_x, area.y + offset_y, width, height)
}

/// Render the Gallery view.
pub fn render(frame: &mut Frame, app: &mut App, area: Rect) {
    match app.gallery.view_mode {
        GalleryViewMode::Grid => render_grid(frame, app, area),
        GalleryViewMode::Detail => render_detail(frame, app, area),
    }
}

fn render_grid(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = &app.theme;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border_focused())
        .title(" Gallery ")
        .title_style(theme.title_focused())
        .style(Style::default().bg(theme.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);
    app.layout.gallery_grid = inner;

    if app.gallery.entries.is_empty() {
        let msg = if app.gallery.scanning {
            "Scanning for images..."
        } else {
            "No images found"
        };
        let empty = Paragraph::new(msg)
            .style(theme.dim())
            .alignment(Alignment::Center);
        let center = Rect {
            x: inner.x,
            y: inner.y + inner.height / 2,
            width: inner.width,
            height: 1,
        };
        frame.render_widget(empty, center);
        return;
    }

    // Compute grid dimensions
    let cols = (inner.width / CELL_W).max(1) as usize;
    app.gallery.grid_cols = cols;
    let visible_rows = (inner.height / CELL_H).max(1) as usize;

    // Ensure selected item is visible
    let selected_row = app.gallery.selected / cols;
    if selected_row < app.gallery.grid_scroll {
        app.gallery.grid_scroll = selected_row;
    } else if selected_row >= app.gallery.grid_scroll + visible_rows {
        app.gallery.grid_scroll = selected_row - visible_rows + 1;
    }

    // Render grid cells
    for vis_row in 0..visible_rows {
        let grid_row = app.gallery.grid_scroll + vis_row;
        for col in 0..cols {
            let idx = grid_row * cols + col;
            if idx >= app.gallery.entries.len() {
                break;
            }

            let cell_x = inner.x + (col as u16) * CELL_W;
            let cell_y = inner.y + (vis_row as u16) * CELL_H;

            // Skip if cell would overflow
            if cell_x + CELL_W > inner.x + inner.width || cell_y + CELL_H > inner.y + inner.height {
                continue;
            }

            let is_selected = idx == app.gallery.selected;
            let cell_area = Rect::new(cell_x, cell_y, CELL_W, CELL_H);

            render_grid_cell(frame, app, cell_area, idx, is_selected);
        }
    }
}

fn render_grid_cell(frame: &mut Frame, app: &mut App, area: Rect, idx: usize, selected: bool) {
    let theme = &app.theme;
    let entry = &app.gallery.entries[idx];

    let border_style = if selected {
        theme.border_focused()
    } else {
        theme.border()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .style(Style::default().bg(theme.bg));

    let cell_inner = block.inner(area);
    frame.render_widget(block, area);

    if cell_inner.height == 0 || cell_inner.width == 0 {
        return;
    }

    // Thumbnail area (all but last 2 rows for filename)
    let thumb_rows = cell_inner.height.saturating_sub(2);
    let thumb_area = Rect::new(cell_inner.x, cell_inner.y, cell_inner.width, thumb_rows);
    let label_area = Rect::new(
        cell_inner.x,
        cell_inner.y + thumb_rows,
        cell_inner.width,
        2.min(cell_inner.height),
    );

    // Load thumbnail lazily if not yet loaded
    if idx < app.gallery.thumbnail_states.len() {
        if app.gallery.thumbnail_states[idx].is_none() {
            let thumb_path = crate::thumbnails::thumbnail_path(&entry.path);
            let mut loaded = false;

            if thumb_path.exists() {
                match image::open(&thumb_path) {
                    Ok(img) => {
                        // Store actual thumbnail pixel dimensions for centering.
                        app.gallery.thumb_dimensions[idx] = Some((img.width(), img.height()));
                        let protocol = app.picker.new_resize_protocol(img);
                        app.gallery.thumbnail_states[idx] = Some(protocol);
                        loaded = true;
                    }
                    Err(_) => {
                        // Corrupt/empty thumbnail — remove so we regenerate below
                        let _ = std::fs::remove_file(&thumb_path);
                    }
                }
            }

            // Regenerate missing thumbnail from source image (local entries only).
            // Server entries are regenerated via the background fetch task.
            if !loaded
                && entry.server_url.is_none()
                && entry.path.is_file()
                && crate::thumbnails::generate_thumbnail(&entry.path).is_ok()
            {
                if let Ok(img) = image::open(&thumb_path) {
                    app.gallery.thumb_dimensions[idx] = Some((img.width(), img.height()));
                    let protocol = app.picker.new_resize_protocol(img);
                    app.gallery.thumbnail_states[idx] = Some(protocol);
                }
            }
        }

        if let Some(ref mut state) = app.gallery.thumbnail_states[idx] {
            let thumb_path = crate::thumbnails::thumbnail_path(&entry.path);
            if let Ok(img) = image::open(&thumb_path) {
                // Build a fixed protocol for the current tile, then center the
                // fitted protocol area within the available thumbnail box.
                if let Ok(mut protocol) =
                    app.picker.new_protocol(img, thumb_area, Resize::Fit(None))
                {
                    let fitted = protocol.area();
                    let centered = center_rect(thumb_area, fitted.width, fitted.height);
                    frame.render_widget(Image::new(&mut protocol), centered);
                } else {
                    // Fall back to the old stateful path if protocol creation fails.
                    let (iw, ih) = app.gallery.thumb_dimensions[idx]
                        .unwrap_or((entry.metadata.width.max(1), entry.metadata.height.max(1)));
                    let font_size = app.picker.font_size();
                    let centered = centered_thumb_rect(
                        thumb_area,
                        iw,
                        ih,
                        font_size,
                        app.picker.protocol_type(),
                    );
                    let image_widget = StatefulImage::default();
                    frame.render_stateful_widget(image_widget, centered, state);
                }
            } else {
                let (iw, ih) = app.gallery.thumb_dimensions[idx]
                    .unwrap_or((entry.metadata.width.max(1), entry.metadata.height.max(1)));
                let font_size = app.picker.font_size();
                let centered =
                    centered_thumb_rect(thumb_area, iw, ih, font_size, app.picker.protocol_type());
                let image_widget = StatefulImage::default();
                frame.render_stateful_widget(image_widget, centered, state);
            }
        }
    }

    // Filename label below thumbnail
    let filename = entry.filename();
    let display_name = if filename.len() > cell_inner.width as usize {
        format!("{}...", &filename[..cell_inner.width as usize - 3])
    } else {
        filename
    };

    let name_style = if selected {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.text_dim)
    };

    let label = Paragraph::new(display_name)
        .style(name_style)
        .alignment(Alignment::Center);
    frame.render_widget(label, label_area);
}

#[cfg(test)]
mod tests {
    use super::{center_rect, centered_thumb_rect};
    use ratatui::layout::Rect;
    use ratatui_image::picker::ProtocolType;

    #[test]
    fn centers_portrait_thumbnails_for_halfblocks() {
        let area = Rect::new(0, 0, 22, 10);
        let rect = centered_thumb_rect(area, 256, 512, (8, 16), ProtocolType::Halfblocks);

        assert_eq!(rect.height, area.height);
        assert!(rect.width < area.width);
        assert_eq!(rect.x, (area.width - rect.width) / 2);
    }

    #[test]
    fn centers_landscape_thumbnails_for_normal_cell_protocols() {
        let area = Rect::new(0, 0, 22, 10);
        let rect = centered_thumb_rect(area, 512, 256, (8, 16), ProtocolType::Kitty);

        assert_eq!(rect.width, area.width);
        assert!(rect.height < area.height);
        assert_eq!(rect.y, (area.height - rect.height) / 2);
    }

    #[test]
    fn gallery_thumbnail_fixed_protocol_area_is_centered() {
        let area = Rect::new(0, 0, 22, 10);
        let rect = center_rect(area, 10, 6);

        assert_eq!(rect, Rect::new(6, 2, 10, 6));
    }
}

fn render_detail(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = &app.theme;

    let entry = match app.gallery.entries.get(app.gallery.selected) {
        Some(e) => e,
        None => {
            let empty = Paragraph::new("No image selected")
                .style(theme.dim())
                .alignment(Alignment::Center);
            frame.render_widget(empty, area);
            return;
        }
    };

    // Horizontal split: metadata (left 2/5) + image (right 3/5)
    let layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);

    // ── Metadata panel ────────────────────────────────────
    let meta_block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border_focused())
        .title(" Details ")
        .title_style(theme.title_focused())
        .style(Style::default().bg(theme.bg));

    let meta_inner = meta_block.inner(layout[0]);
    frame.render_widget(meta_block, layout[0]);

    let meta = &entry.metadata;
    let mut lines: Vec<Line> = Vec::new();

    // Filename
    lines.push(Line::from(Span::styled(
        entry.filename(),
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    // Prompt
    lines.push(Line::from(Span::styled("Prompt", theme.param_label())));
    for prompt_line in meta.prompt.lines() {
        lines.push(Line::from(Span::styled(
            prompt_line.to_string(),
            Style::default().fg(theme.text),
        )));
    }

    // Negative prompt
    if let Some(ref neg) = meta.negative_prompt {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Negative", theme.param_label())));
        for neg_line in neg.lines() {
            lines.push(Line::from(Span::styled(
                neg_line.to_string(),
                Style::default().fg(theme.text),
            )));
        }
    }

    lines.push(Line::from(""));

    // Parameters
    let param_lines = [
        ("Model", meta.model.clone()),
        ("Size", format!("{}x{}", meta.width, meta.height)),
        ("Steps", meta.steps.to_string()),
        ("Guidance", format!("{:.1}", meta.guidance)),
        ("Seed", meta.seed.to_string()),
    ];
    for (label, value) in &param_lines {
        lines.push(Line::from(vec![
            Span::styled(format!("{:<10}", label), theme.param_label()),
            Span::styled(value.clone(), theme.param_value()),
        ]));
    }

    // Optional parameters
    if let Some(strength) = meta.strength {
        lines.push(Line::from(vec![
            Span::styled(format!("{:<10}", "Strength"), theme.param_label()),
            Span::styled(format!("{strength:.2}"), theme.param_value()),
        ]));
    }
    if let Some(ref sched) = meta.scheduler {
        lines.push(Line::from(vec![
            Span::styled(format!("{:<10}", "Scheduler"), theme.param_label()),
            Span::styled(format!("{sched:?}"), theme.param_value()),
        ]));
    }
    if let Some(ref lora) = meta.lora {
        let lora_display = std::path::Path::new(lora)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| lora.clone());
        lines.push(Line::from(vec![
            Span::styled(format!("{:<10}", "LoRA"), theme.param_label()),
            Span::styled(lora_display, theme.param_value()),
        ]));
        if let Some(scale) = meta.lora_scale {
            lines.push(Line::from(vec![
                Span::styled(format!("{:<10}", "LoRA Scl"), theme.param_label()),
                Span::styled(format!("{scale:.2}"), theme.param_value()),
            ]));
        }
    }

    if let Some(gen_ms) = entry.generation_time_ms {
        lines.push(Line::from(vec![
            Span::styled(format!("{:<10}", "Time"), theme.param_label()),
            Span::styled(
                format!("{:.1}s", gen_ms as f64 / 1000.0),
                theme.param_value(),
            ),
        ]));
    }

    lines.push(Line::from(""));

    // File path
    lines.push(Line::from(Span::styled(
        format!("{}", entry.path.display()),
        theme.dim(),
    )));

    lines.push(Line::from(""));

    // Keybinding hints
    let hints: &[(&str, &str)] = &[
        ("e", "Edit"),
        ("r", "Regenerate"),
        ("d", "Delete"),
        ("o/Enter", "Open"),
        ("Esc", "Back"),
    ];
    let hint_spans: Vec<Span> = hints
        .iter()
        .enumerate()
        .flat_map(|(i, (k, desc))| {
            let mut spans = Vec::new();
            if i > 0 {
                spans.push(Span::styled("  ", theme.dim()));
            }
            spans.push(Span::styled(*k, theme.status_key()));
            spans.push(Span::styled(" ", theme.dim()));
            spans.push(Span::styled(*desc, theme.dim()));
            spans
        })
        .collect();
    lines.push(Line::from(hint_spans));

    let details = Paragraph::new(lines).wrap(Wrap { trim: false });
    frame.render_widget(details, meta_inner);

    // ── Image preview ─────────────────────────────────────
    let preview_block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border())
        .title(" Preview ")
        .title_style(theme.title())
        .style(Style::default().bg(theme.bg));

    let preview_inner = preview_block.inner(layout[1]);
    frame.render_widget(preview_block, layout[1]);

    if let Some(ref mut image_state) = app.gallery.image_state {
        let image_widget = StatefulImage::default().resize(Resize::Scale(None));
        frame.render_stateful_widget(image_widget, preview_inner, image_state);
    } else {
        let msg = Paragraph::new("Loading...")
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
}
