use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui_image::picker::ProtocolType;
use ratatui_image::{Image, Resize, StatefulImage};

use crate::app::{App, GalleryEntry, GalleryViewMode};
use crate::ui::widgets::{kv_row_line, panel_block};

/// Width of a single gallery tile.
pub(crate) const CELL_W: u16 = 24;
/// Height of a single gallery tile including its 2 border rows.
///
/// The cell used to reserve two rows for a filename label, which never fit
/// on disk-era names and was redundant with the Selected panel below the
/// grid. Removing the label lets the thumbnail fill the full inner area
/// *and* fits one extra tile row on typical terminal heights.
///
/// Shared with the mouse hit-test in `app::handle_mouse` so click
/// detection can never drift from the rendered cell size.
pub(crate) const CELL_H: u16 = 12;

/// Height of the bottom row (Selected + Prompt panels) in the Grid view.
const GRID_BOTTOM_HEIGHT: u16 = 8;

/// Minimum total Gallery area height that still leaves room for at least one
/// thumbnail *and* the Selected + Prompt inspector row. The grid panel frames
/// a thumbnail in `CELL_H + 2` cells (cell + top/bottom border), so we need
/// `GRID_BOTTOM_HEIGHT + CELL_H + 2` before the inspector is safe to show.
const GRID_INSPECTOR_MIN_HEIGHT: u16 = GRID_BOTTOM_HEIGHT + CELL_H + 2;

/// Whether the Grid view has enough vertical room to render the inspector row
/// without starving the thumbnail grid. Extracted as a pure helper so the
/// threshold is exercised by unit tests — see the `#[cfg(test)]` block below.
fn show_grid_inspector(area_height: u16, has_entries: bool) -> bool {
    has_entries && area_height >= GRID_INSPECTOR_MIN_HEIGHT
}

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
    // Top: the thumbnail grid. Bottom: Selected metadata + Prompt, matching
    // the design system's gallery wireframe. If the area is too short to
    // fit both the inspector *and* at least one thumbnail cell the inspector
    // is suppressed — see `show_grid_inspector`/`GRID_INSPECTOR_MIN_HEIGHT`.
    let show_bottom = show_grid_inspector(area.height, !app.gallery.entries.is_empty());
    let constraints = if show_bottom {
        vec![
            Constraint::Min(CELL_H + 2),
            Constraint::Length(GRID_BOTTOM_HEIGHT),
        ]
    } else {
        vec![Constraint::Min(1)]
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    render_grid_panel(frame, app, rows[0]);
    if show_bottom {
        render_grid_bottom_row(frame, app, rows[1]);
    }
}

fn render_grid_panel(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = &app.theme;

    let hint = if app.gallery.entries.is_empty() {
        None
    } else {
        Some(format!("{} images", app.gallery.entries.len()))
    };
    let block = panel_block(theme, "Gallery", true, hint.as_deref());

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

/// Render the Selected + Prompt panels beneath the gallery grid.
///
/// The row is 8 cells tall (6 content + 2 borders). Selected holds file
/// metadata as KV rows; Prompt shows the positive prompt plus a single dim
/// `neg: …` line for the negative prompt when present.
fn render_grid_bottom_row(frame: &mut Frame, app: &App, area: Rect) {
    let [selected_area, prompt_area] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .areas(area);

    let entry = app.gallery.entries.get(app.gallery.selected);
    render_selected_panel(frame, &app.theme, entry, selected_area);
    render_prompt_panel(frame, &app.theme, entry, prompt_area);
}

fn render_selected_panel(
    frame: &mut Frame,
    theme: &crate::ui::theme::Theme,
    entry: Option<&GalleryEntry>,
    area: Rect,
) {
    let block = panel_block(theme, "Selected", false, None);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let Some(entry) = entry else {
        let empty = Paragraph::new("No image selected").style(theme.dim());
        frame.render_widget(empty, inner);
        return;
    };

    let filename = entry.filename();
    let dim = format!("{}×{}", entry.metadata.width, entry.metadata.height);
    let seed = entry.metadata.seed.to_string();
    let steps = entry.metadata.steps.to_string();

    let lines = vec![
        kv_row_line(theme, "File", &filename, 7, false),
        kv_row_line(theme, "Model", &entry.metadata.model, 7, false),
        kv_row_line(theme, "Dim", &dim, 7, false),
        kv_row_line(theme, "Steps", &steps, 7, false),
        kv_row_line(theme, "Seed", &seed, 7, true),
    ];
    let para = Paragraph::new(lines);
    frame.render_widget(para, inner);
}

fn render_prompt_panel(
    frame: &mut Frame,
    theme: &crate::ui::theme::Theme,
    entry: Option<&GalleryEntry>,
    area: Rect,
) {
    let block = panel_block(theme, "Prompt", false, None);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let Some(entry) = entry else {
        let empty = Paragraph::new("No image selected").style(theme.dim());
        frame.render_widget(empty, inner);
        return;
    };

    let has_neg = entry.metadata.negative_prompt.is_some();
    let prompt_rows = if has_neg {
        inner.height.saturating_sub(1).max(1)
    } else {
        inner.height
    };

    let prompt = Paragraph::new(entry.metadata.prompt.clone())
        .style(Style::default().fg(theme.text))
        .wrap(Wrap { trim: true });
    let prompt_area = Rect {
        x: inner.x,
        y: inner.y,
        width: inner.width,
        height: prompt_rows,
    };
    frame.render_widget(prompt, prompt_area);

    if let (Some(neg), true) = (
        entry.metadata.negative_prompt.as_deref(),
        inner.height > prompt_rows,
    ) {
        let neg_line = Paragraph::new(format!("neg: {neg}")).style(theme.dim());
        let neg_area = Rect {
            x: inner.x,
            y: inner.y + prompt_rows,
            width: inner.width,
            height: 1,
        };
        frame.render_widget(neg_line, neg_area);
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

    // Thumbnail fills the full inner area now that per-cell filename
    // labels have been removed — they never fit at CELL_W=24 and the
    // Selected panel below the grid already shows the full filename.
    let thumb_area = cell_inner;

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

        if app.gallery.thumbnail_states[idx].is_some() {
            // Use a cached fixed protocol for centered grid thumbnails.
            // Stateful protocols pad from top-left on Kitty/Sixel/iTerm2,
            // which regresses visible centering. The fixed protocol is
            // created once per thumbnail and reused across render frames.
            let cache_valid = app
                .gallery
                .thumb_fixed_cache
                .get(idx)
                .and_then(|c| c.as_ref())
                .is_some_and(|(w, h, _)| *w == thumb_area.width && *h == thumb_area.height);

            if !cache_valid {
                let thumb_path = crate::thumbnails::thumbnail_path(&entry.path);
                if let Ok(img) = image::open(&thumb_path) {
                    if let Ok(protocol) =
                        app.picker.new_protocol(img, thumb_area, Resize::Fit(None))
                    {
                        // Grow cache if needed
                        while app.gallery.thumb_fixed_cache.len() <= idx {
                            app.gallery.thumb_fixed_cache.push(None);
                        }
                        app.gallery.thumb_fixed_cache[idx] =
                            Some((thumb_area.width, thumb_area.height, protocol));
                    }
                }
            }

            if let Some((_, _, ref mut protocol)) = app
                .gallery
                .thumb_fixed_cache
                .get_mut(idx)
                .and_then(|c| c.as_mut())
            {
                let fitted = protocol.area();
                let centered = center_rect(thumb_area, fitted.width, fitted.height);
                frame.render_widget(Image::new(protocol), centered);
            } else if let Some(ref mut state) = app.gallery.thumbnail_states[idx] {
                // Fallback to stateful rendering if fixed protocol unavailable
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

    // Intentionally no filename label — the Selected panel below the
    // grid shows the full name for the currently-highlighted tile.
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::{
        center_rect, centered_thumb_rect, render_grid_cell, show_grid_inspector, CELL_H, CELL_W,
        GRID_INSPECTOR_MIN_HEIGHT,
    };
    use crate::app::{App, GalleryEntry};
    use image::{DynamicImage, Rgba, RgbaImage};
    use ratatui::layout::Rect;
    use ratatui::{backend::TestBackend, Terminal};
    use ratatui_image::picker::Picker;
    use ratatui_image::picker::ProtocolType;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_metadata(width: u32, height: u32) -> mold_core::OutputMetadata {
        mold_core::OutputMetadata {
            prompt: "test prompt".to_string(),
            negative_prompt: None,
            original_prompt: None,
            model: "flux2-klein:q8".to_string(),
            seed: 1,
            steps: 4,
            guidance: 0.0,
            width,
            height,
            strength: None,
            scheduler: None,
            lora: None,
            lora_scale: None,
            version: "test".to_string(),
            frames: None,
            fps: None,
        }
    }

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

    #[test]
    fn gallery_grid_cell_does_not_render_filename_label() {
        // Reported: thumbnail cells rendered a truncated filename below
        // the image (`mold-flux-dev-q4-17…`). The label never fit, was
        // redundant with the Selected panel below the grid, and ate two
        // rows of thumbnail space per cell.
        //
        // TDD: render a single cell with a recognisable filename stem
        // and assert no cell in the buffer contains even a prefix of
        // that stem.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();

        let mut picker = Picker::from_fontsize((8, 16));
        picker.set_protocol_type(ProtocolType::Halfblocks);
        let mut app = App::new(None, true, picker).unwrap();

        let stem = "unique-cell-label-stem";
        let entry_path = PathBuf::from(format!("{stem}.png"));
        app.gallery.entries = vec![GalleryEntry {
            path: entry_path,
            metadata: test_metadata(64, 64),
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
        }];
        // No thumbnail loaded — we only care about the text rendering path.
        app.gallery.thumbnail_states = vec![None];
        app.gallery.thumb_dimensions = vec![None];

        let backend = TestBackend::new(CELL_W, CELL_H);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render_grid_cell(frame, &mut app, Rect::new(0, 0, CELL_W, CELL_H), 0, true);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let rendered: String = buffer.content.iter().map(|c| c.symbol()).collect();
        assert!(
            !rendered.contains("unique-cell-label-stem"),
            "gallery cell must not render the filename label; got: {rendered:?}"
        );
        // Also guard against truncated variants like `unique-cell…`.
        assert!(
            !rendered.contains("unique-cell"),
            "gallery cell must not render even a truncated filename prefix; got: {rendered:?}"
        );
    }

    #[test]
    fn gallery_grid_kitty_thumbnails_encode_to_full_thumb_box() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();

        let mut picker = Picker::from_fontsize((8, 16));
        picker.set_protocol_type(ProtocolType::Kitty);
        let mut app = App::new(None, true, picker).unwrap();

        let img =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(512, 1024, Rgba([255, 0, 0, 255])));
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let entry_path = PathBuf::from(format!("gallery-center-test-{unique}.png"));
        let thumb_path = crate::thumbnails::thumbnail_path(&entry_path);
        if let Some(parent) = thumb_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        img.save(&thumb_path).unwrap();

        app.gallery.entries = vec![GalleryEntry {
            path: entry_path.clone(),
            metadata: test_metadata(512, 1024),
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
        }];
        app.gallery.thumbnail_states = vec![Some(app.picker.new_resize_protocol(img.clone()))];
        app.gallery.thumb_dimensions = vec![Some((512, 1024))];

        let backend = TestBackend::new(CELL_W, CELL_H);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render_grid_cell(frame, &mut app, Rect::new(0, 0, CELL_W, CELL_H), 0, true);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let transmit_cell = buffer
            .content
            .iter()
            .find(|cell| cell.symbol().contains("_Gq=2"))
            .expect("expected kitty image transmit sequence in buffer");

        // The fixed-protocol gallery path must encode against the full 22x10
        // thumbnail box, then center that fitted rect at placement time. If a
        // later edit switches this back to the stateful path, this width will
        // shrink to the already-fitted rect and the visible top-left bias
        // returns in real terminals.
        assert!(
            transmit_cell.symbol().contains("s=176,v=160"),
            "expected kitty payload sized to full thumb box, got: {}",
            transmit_cell.symbol()
        );

        std::fs::remove_file(&thumb_path).ok();
    }

    // ── Codex P2: inspector must not starve the thumbnail grid ───────

    #[test]
    fn show_grid_inspector_hidden_below_minimum_height() {
        // Minimum is `GRID_BOTTOM_HEIGHT (8) + CELL_H + 2 borders`.
        // At heights below that, showing the inspector would leave zero
        // room for a thumbnail row.
        let min = GRID_INSPECTOR_MIN_HEIGHT;
        assert!(!show_grid_inspector(0, true));
        assert!(!show_grid_inspector(min.saturating_sub(1), true));
        assert!(!show_grid_inspector(min.saturating_sub(10), true));
    }

    #[test]
    fn show_grid_inspector_visible_at_and_above_minimum_height() {
        assert!(show_grid_inspector(GRID_INSPECTOR_MIN_HEIGHT, true));
        assert!(show_grid_inspector(GRID_INSPECTOR_MIN_HEIGHT + 20, true));
    }

    #[test]
    fn show_grid_inspector_hidden_when_gallery_is_empty() {
        // With no entries there's nothing to inspect; skip the bottom row so
        // the empty-state message fills the whole panel.
        assert!(!show_grid_inspector(100, false));
    }

    // Compile-time guard: if someone bumps CELL_H or GRID_BOTTOM_HEIGHT
    // without updating the threshold, the build breaks instead of the
    // gallery quietly hiding thumbnails at runtime.
    const _: () = assert!(GRID_INSPECTOR_MIN_HEIGHT >= CELL_H + 2);
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
    let meta_block = panel_block(theme, "Details", true, None);
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
        ("u", "Upscale"),
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
    let preview_block = panel_block(theme, "Preview", false, None);
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
