use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui_image::picker::ProtocolType;
use ratatui_image::{Image, Resize, StatefulImage};

use crate::app::{App, GalleryViewMode};
use crate::ui::library_details::{show_details_panel, DETAILS_PANEL_W};
use crate::ui::widgets::panel_block;

/// Width of a single gallery tile.
pub(crate) const CELL_W: u16 = 24;
/// Height of a single gallery tile including its 2 border rows.
///
/// The cell used to reserve two rows for a filename label, which never fit
/// on disk-era names and was redundant with the old Selected panel below
/// the grid. Removing the label lets the thumbnail fill the full inner
/// area *and* fits one extra tile row on typical terminal heights.
///
/// Shared with the mouse hit-test in `app::handle_mouse` so click
/// detection can never drift from the rendered cell size.
pub(crate) const CELL_H: u16 = 12;

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

pub(crate) fn center_rect(area: Rect, used_w: u16, used_h: u16) -> Rect {
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
    // Left: the thumbnail grid. Right: the Details side panel (selected
    // print's thumbnail, prompt, KV rows incl. Machine, action hints).
    // The panel is suppressed on narrow terminals so the grid always
    // keeps at least one tile column — see `show_details_panel`.
    let show_details = show_details_panel(area.width) && !app.gallery.entries.is_empty();
    if show_details {
        let [grid_area, details_area] = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(CELL_W + 2),
                Constraint::Length(DETAILS_PANEL_W),
            ])
            .areas(area);
        render_grid_panel(frame, app, grid_area);
        crate::ui::library_details::render(frame, app, details_area);
    } else {
        render_grid_panel(frame, app, area);
    }
}

fn render_grid_panel(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = &app.theme;

    let hint = crate::gallery_scan::library_hint(
        &app.gallery.entries,
        &app.gallery.filter,
        app.gallery.filtered.len(),
        app.gallery.offline_hosts,
    );
    let block = panel_block(theme, "Library", true, hint.as_deref());

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

    // The grid shows the filtered list; caches stay keyed by entry index.
    if app.gallery.filtered.is_empty() {
        let empty = Paragraph::new(format!("No prints match /{}", app.gallery.filter))
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

    // Ensure the selected item is visible (positions are within the
    // filtered list — the same coordinate space the mouse hit-test uses).
    let selected_pos = app.gallery.selected_pos().unwrap_or(0);
    let selected_row = selected_pos / cols;
    if selected_row < app.gallery.grid_scroll {
        app.gallery.grid_scroll = selected_row;
    } else if selected_row >= app.gallery.grid_scroll + visible_rows {
        app.gallery.grid_scroll = selected_row - visible_rows + 1;
    }

    // Render grid cells
    for vis_row in 0..visible_rows {
        let grid_row = app.gallery.grid_scroll + vis_row;
        for col in 0..cols {
            let pos = grid_row * cols + col;
            if pos >= app.gallery.filtered.len() {
                break;
            }
            let idx = app.gallery.filtered[pos];

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
pub(crate) mod tests {
    use super::{center_rect, centered_thumb_rect, render_grid_cell, CELL_H, CELL_W};
    use crate::app::{App, GalleryEntry};
    use image::{DynamicImage, Rgba, RgbaImage};
    use ratatui::layout::Rect;
    use ratatui::{backend::TestBackend, Terminal};
    use ratatui_image::picker::Picker;
    use ratatui_image::picker::ProtocolType;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    pub(crate) fn test_metadata(width: u32, height: u32) -> mold_core::OutputMetadata {
        mold_core::OutputMetadata {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            job_id: None,
            prompt: "test prompt".to_string(),
            negative_prompt: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
            model: "flux2-klein:q8".to_string(),
            seed: 1,
            steps: 4,
            guidance: 0.0,
            width,
            height,
            generation_width: Some(width),
            generation_height: Some(height),
            strength: None,
            source_image_name: None,
            source_image_sha256: None,
            edit_image_sha256s: None,
            references: None,
            keyframes: None,
            scheduler: None,
            output_format: Some(mold_core::OutputFormat::Png),
            cfg_plus: None,
            lora: None,
            lora_scale: None,
            loras: None,
            control_model: None,
            control_scale: None,
            upscale_model: None,
            gif_preview: None,
            enable_audio: None,
            audio_file_path: None,
            source_video_path: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            pipeline: None,
            pipeline_requested: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            ic_lora_control: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            chain_job_id: None,
            chain: None,
            version: "test".to_string(),
            frames: None,
            fps: None,
            id_image_name: None,
            id_image_sha256: None,
            id_weight: None,
            id_start_step: None,
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
    #[serial_test::serial(mold_env)]
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
            title: None,
            origins: Vec::new(),
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
    #[serial_test::serial(mold_env)]
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
            title: None,
            origins: Vec::new(),
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

    // ── Details side panel layout contract ───────────────────────────
    // The old bottom Selected + Prompt inspector row is retired; its
    // content lives in the Details side panel. These tests pin the grid
    // split so the panel can never starve the grid of its one tile column.

    #[test]
    #[serial_test::serial(mold_env)]
    fn grid_shows_details_panel_when_wide_enough() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();

        let mut picker = Picker::from_fontsize((8, 16));
        picker.set_protocol_type(ProtocolType::Halfblocks);
        let mut app = App::new(None, true, picker).unwrap();
        app.gallery.entries = vec![GalleryEntry {
            path: PathBuf::from("wide-terminal-print.png"),
            metadata: test_metadata(64, 64),
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
            title: None,
            origins: Vec::new(),
        }];
        app.gallery.thumbnail_states = vec![None];
        app.gallery.thumb_dimensions = vec![None];
        app.gallery.thumb_fixed_cache = vec![None];
        app.gallery.refresh_filter();

        let wide = CELL_W + 2 + crate::ui::library_details::DETAILS_PANEL_W;
        let backend = TestBackend::new(wide, CELL_H + 4);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                super::render_grid(frame, &mut app, Rect::new(0, 0, wide, CELL_H + 4));
            })
            .unwrap();
        let rendered: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(
            rendered.contains("Details"),
            "wide grid must render the Details side panel; got: {rendered:?}"
        );
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn grid_hides_details_panel_below_min_width() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();

        let mut picker = Picker::from_fontsize((8, 16));
        picker.set_protocol_type(ProtocolType::Halfblocks);
        let mut app = App::new(None, true, picker).unwrap();
        app.gallery.entries = vec![GalleryEntry {
            path: PathBuf::from("narrow-terminal-print.png"),
            metadata: test_metadata(64, 64),
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
            title: None,
            origins: Vec::new(),
        }];
        app.gallery.thumbnail_states = vec![None];
        app.gallery.thumb_dimensions = vec![None];
        app.gallery.thumb_fixed_cache = vec![None];
        app.gallery.refresh_filter();

        let narrow = CELL_W + 2 + crate::ui::library_details::DETAILS_PANEL_W - 1;
        let backend = TestBackend::new(narrow, CELL_H + 4);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                super::render_grid(frame, &mut app, Rect::new(0, 0, narrow, CELL_H + 4));
            })
            .unwrap();
        let rendered: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(
            !rendered.contains("Details"),
            "narrow grid must suppress the Details side panel; got: {rendered:?}"
        );
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn full_detail_shows_the_runtime_pipeline_when_present() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();

        let mut picker = Picker::from_fontsize((8, 16));
        picker.set_protocol_type(ProtocolType::Halfblocks);
        let mut app = App::new(None, true, picker).unwrap();
        let mut metadata = test_metadata(1216, 704);
        metadata.output_format = Some(mold_core::OutputFormat::Mp4);
        metadata.pipeline = Some(mold_core::Ltx2PipelineMode::TwoStageHq);
        app.gallery.entries = vec![GalleryEntry {
            path: PathBuf::from("runtime-pipeline.mp4"),
            metadata,
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
            title: None,
            origins: Vec::new(),
        }];

        let backend = TestBackend::new(100, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| super::render_detail(frame, &mut app, Rect::new(0, 0, 100, 40)))
            .unwrap();
        let rendered: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();

        assert!(
            rendered.contains("Pipeline"),
            "pipeline label: {rendered:?}"
        );
        assert!(
            rendered.contains("two-stage-hq"),
            "runtime pipeline value: {rendered:?}"
        );
    }

    fn render_detail_to_string(app: &mut App) -> String {
        let backend = TestBackend::new(100, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| super::render_detail(frame, &mut *app, Rect::new(0, 0, 100, 40)))
            .unwrap();
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn full_detail_shows_the_title_above_the_filename_when_present() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();

        let mut picker = Picker::from_fontsize((8, 16));
        picker.set_protocol_type(ProtocolType::Halfblocks);
        let mut app = App::new(None, true, picker).unwrap();
        app.gallery.entries = vec![GalleryEntry {
            path: PathBuf::from("titled-print.png"),
            metadata: test_metadata(64, 64),
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
            title: Some("Lighthouse study".into()),
            origins: Vec::new(),
        }];

        let rendered = render_detail_to_string(&mut app);
        let title_at = rendered.find("Lighthouse study").expect("title rendered");
        let name_at = rendered
            .find("titled-print.png")
            .expect("filename rendered");
        assert!(title_at < name_at, "title sits above the filename");

        // Untitled prints render nothing in the title's place.
        app.gallery.entries[0].title = None;
        let rendered = render_detail_to_string(&mut app);
        assert!(!rendered.contains("Lighthouse study"));
        assert!(rendered.contains("titled-print.png"));
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn full_detail_hint_names_trash_only_when_the_print_can_be_trashed() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();

        let mut picker = Picker::from_fontsize((8, 16));
        picker.set_protocol_type(ProtocolType::Halfblocks);
        let mut app = App::new(None, true, picker).unwrap();
        app.gallery.entries = vec![GalleryEntry {
            path: PathBuf::from("hinted.png"),
            metadata: test_metadata(64, 64),
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
            title: None,
            origins: Vec::new(),
        }];

        app.gallery.local_trash_available = false;
        let rendered = render_detail_to_string(&mut app);
        assert!(rendered.contains("Delete"), "{rendered:?}");
        assert!(!rendered.contains("Trash"), "{rendered:?}");

        app.gallery.local_trash_available = true;
        let rendered = render_detail_to_string(&mut app);
        assert!(rendered.contains("Trash"), "{rendered:?}");
        assert!(!rendered.contains("Delete"), "{rendered:?}");
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
    let meta_block = panel_block(theme, "Details", true, None);
    let meta_inner = meta_block.inner(layout[0]);
    frame.render_widget(meta_block, layout[0]);

    let meta = &entry.metadata;
    let removal = app.selected_removal_kind().hint_label();
    let mut lines: Vec<Line> = Vec::new();

    // Title (when the print has one), then filename
    if let Some(title) = entry
        .title
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        lines.push(Line::from(Span::styled(
            title.to_string(),
            Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
        )));
    }
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

    if let Some(pipeline) = meta.pipeline {
        lines.push(Line::from(vec![
            Span::styled(format!("{:<10}", "Pipeline"), theme.param_label()),
            Span::styled(pipeline.to_string(), theme.param_value()),
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
        ("d", removal),
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
