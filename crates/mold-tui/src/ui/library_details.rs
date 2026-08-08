//! Library details side panel — the selected print's thumbnail, prompt,
//! key metadata (including which Machine it lives on), and action hints.
//!
//! Replaces the old bottom Selected + Prompt inspector row. The panel
//! reuses the grid's fixed-protocol image pattern (`Image::new` over a
//! cached `Protocol`) with a one-slot cache keyed on
//! `(entry index, width, height)` — it renders the *thumbnail* file, never
//! a full-image decode per selection change.

use ratatui::prelude::*;
use ratatui::widgets::{Paragraph, Wrap};
use ratatui_image::{Image, Resize};

use crate::app::App;
use crate::ui::widgets::panel_block;

/// Width of the Details side panel (including its borders).
pub(crate) const DETAILS_PANEL_W: u16 = 28;

/// Rows reserved for the thumbnail at the top of the panel.
const THUMB_ROWS: u16 = 10;

/// Rows of wrapped prompt text shown at most (plus one dim `neg:` line).
const PROMPT_MAX_ROWS: u16 = 6;

/// Whether the Grid view is wide enough to show the side panel without
/// starving the grid of its first tile column (`CELL_W` + 2 borders).
pub(crate) fn show_details_panel(width: u16) -> bool {
    width >= crate::ui::gallery::CELL_W + 2 + DETAILS_PANEL_W
}

/// Render the Details panel for the currently selected print.
pub(crate) fn render(frame: &mut Frame, app: &mut App, area: Rect) {
    let block = panel_block(&app.theme, "Details", false, None);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let Some(entry) = app.gallery.entries.get(app.gallery.selected) else {
        let empty = Paragraph::new("No print selected").style(app.theme.dim());
        frame.render_widget(empty, inner);
        return;
    };

    // Snapshot everything the text content needs before the mutable
    // thumbnail-cache borrow below.
    let idx = app.gallery.selected;
    let prompt = entry.metadata.prompt.clone();
    let negative = entry.metadata.negative_prompt.clone();
    let model =
        mold_core::ModelInfoExtended::human_name_for(&entry.metadata.model, &app.models.catalog);
    let seed = entry.metadata.seed.to_string();
    let dims = format!("{}×{}", entry.metadata.width, entry.metadata.height);
    let pipeline = entry.metadata.pipeline.map(|pipeline| pipeline.to_string());
    let machine = entry.machine_label();
    let thumb_path = crate::thumbnails::thumbnail_path(&entry.path);

    // ── Thumbnail (fixed protocol, one-slot cache) ──────────────
    let mut y = inner.y;
    if inner.height >= THUMB_ROWS + 8 {
        let thumb_area = Rect {
            x: inner.x,
            y,
            width: inner.width,
            height: THUMB_ROWS,
        };
        render_thumb(frame, app, thumb_area, idx, &thumb_path);
        y += THUMB_ROWS + 1;
    }

    let theme = &app.theme;

    // ── Prompt (wrapped) + dim negative line ────────────────────
    let remaining = (inner.y + inner.height).saturating_sub(y);
    let neg_rows = u16::from(negative.is_some());
    // Four required KV rows, optional pipeline, blank, two hints, and blank.
    let kv_rows = 4 + u16::from(pipeline.is_some());
    let prompt_rows = PROMPT_MAX_ROWS
        .min(remaining.saturating_sub(neg_rows + kv_rows + 4))
        .max(1)
        .min(remaining);
    let prompt_area = Rect {
        x: inner.x,
        y,
        width: inner.width,
        height: prompt_rows,
    };
    let prompt_para = Paragraph::new(prompt)
        .style(Style::default().fg(theme.text))
        .wrap(Wrap { trim: true });
    frame.render_widget(prompt_para, prompt_area);
    y += prompt_rows;

    if let Some(neg) = negative {
        if y < inner.y + inner.height {
            let neg_area = Rect {
                x: inner.x,
                y,
                width: inner.width,
                height: 1,
            };
            frame.render_widget(
                Paragraph::new(format!("neg: {neg}")).style(theme.dim()),
                neg_area,
            );
            y += 1;
        }
    }
    y += 1; // spacer

    // ── KV rows + action hints ──────────────────────────────────
    let mut lines: Vec<Line> = vec![
        crate::ui::widgets::kv_row_line(theme, "Model", &model, 8, false),
        crate::ui::widgets::kv_row_line(theme, "Seed", &seed, 8, false),
        crate::ui::widgets::kv_row_line(theme, "Size", &dims, 8, false),
    ];
    if let Some(ref pipeline) = pipeline {
        lines.push(crate::ui::widgets::kv_row_line(
            theme, "Pipeline", pipeline, 8, false,
        ));
    }
    lines.extend([
        crate::ui::widgets::kv_row_line(theme, "Machine", &machine, 8, false),
        Line::from(""),
    ]);
    for (key, label) in [("[r]", "Recall prompt"), ("[u]", "Upscale")] {
        lines.push(Line::from(vec![
            Span::styled(key, theme.status_key()),
            Span::raw(" "),
            Span::styled(label, theme.dim()),
        ]));
    }

    if y < inner.y + inner.height {
        let text_area = Rect {
            x: inner.x,
            y,
            width: inner.width,
            height: (inner.y + inner.height).saturating_sub(y),
        };
        frame.render_widget(Paragraph::new(lines), text_area);
    }
}

/// Render the panel thumbnail via the one-slot fixed-protocol cache.
/// Cache key is `(entry index, area width, area height)` — selection or
/// geometry changes rebuild it; re-renders of the same selection reuse it.
fn render_thumb(
    frame: &mut Frame,
    app: &mut App,
    thumb_area: Rect,
    idx: usize,
    thumb_path: &std::path::Path,
) {
    let cache_valid = app
        .gallery
        .details_thumb
        .as_ref()
        .is_some_and(|(i, w, h, _)| *i == idx && *w == thumb_area.width && *h == thumb_area.height);

    if !cache_valid {
        app.gallery.details_thumb = None;
        if thumb_path.is_file() {
            if let Ok(img) = image::open(thumb_path) {
                if let Ok(protocol) = app.picker.new_protocol(img, thumb_area, Resize::Fit(None)) {
                    app.gallery.details_thumb =
                        Some((idx, thumb_area.width, thumb_area.height, protocol));
                }
            }
        }
    }

    if let Some((_, _, _, ref mut protocol)) = app.gallery.details_thumb {
        let fitted = protocol.area();
        let centered = crate::ui::gallery::center_rect(thumb_area, fitted.width, fitted.height);
        frame.render_widget(Image::new(protocol), centered);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{App, GalleryEntry, GalleryOrigin};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use ratatui_image::picker::{Picker, ProtocolType};
    use std::path::PathBuf;

    // The KV label column (8) + a short value must fit the panel's inner
    // width; if someone shrinks the panel below label+borders the build
    // breaks instead of values silently truncating to nothing.
    const _: () = assert!(DETAILS_PANEL_W >= 8 + 2 + 4);

    fn test_metadata() -> mold_core::OutputMetadata {
        mold_core::OutputMetadata {
            guidance_overrides: None,
            prompt: "a lighthouse at dusk".to_string(),
            negative_prompt: Some("blurry".to_string()),
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            model: "flux2-klein:q8".to_string(),
            seed: 771204,
            steps: 4,
            guidance: 0.0,
            width: 1024,
            height: 1024,
            generation_width: Some(1024),
            generation_height: Some(1024),
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
        }
    }

    fn test_app_with_entry(origins: Vec<GalleryOrigin>) -> App {
        let mut picker = Picker::from_fontsize((8, 16));
        picker.set_protocol_type(ProtocolType::Halfblocks);
        let mut app = App::new(None, true, picker).unwrap();
        app.gallery.entries = vec![GalleryEntry {
            path: PathBuf::from("details-panel-test.png"),
            metadata: test_metadata(),
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
            origins,
        }];
        app.gallery.thumbnail_states = vec![None];
        app.gallery.thumb_dimensions = vec![None];
        app.gallery.thumb_fixed_cache = vec![None];
        app.gallery.refresh_filter();
        app
    }

    fn render_to_string(app: &mut App) -> String {
        let backend = TestBackend::new(DETAILS_PANEL_W, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render(frame, app, Rect::new(0, 0, DETAILS_PANEL_W, 30));
            })
            .unwrap();
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    #[test]
    fn show_details_panel_hidden_below_min_width() {
        let min = crate::ui::gallery::CELL_W + 2 + DETAILS_PANEL_W;
        assert!(!show_details_panel(0));
        assert!(!show_details_panel(min - 1));
        assert!(show_details_panel(min));
        assert!(show_details_panel(min + 40));
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn panel_shows_prompt_kv_rows_and_machine_this_mac_for_local() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();

        let mut app = test_app_with_entry(Vec::new());
        let rendered = render_to_string(&mut app);

        assert!(rendered.contains("lighthouse"), "prompt: {rendered:?}");
        assert!(rendered.contains("neg: blurry"), "negative: {rendered:?}");
        assert!(rendered.contains("771204"), "seed: {rendered:?}");
        assert!(rendered.contains("Machine"), "machine row: {rendered:?}");
        assert!(
            rendered.contains(crate::hosts::local_display_name()),
            "local entries show the local machine name: {rendered:?}"
        );
        assert!(
            rendered.contains("[r] Recall prompt") && rendered.contains("[u] Upscale"),
            "action hints: {rendered:?}"
        );
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn panel_machine_row_names_the_remote_host() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();

        let mut app = test_app_with_entry(vec![GalleryOrigin {
            host_id: "hal9000-7680".into(),
            url: Some("http://hal9000:7680".into()),
            name: "hal9000".into(),
        }]);
        let rendered = render_to_string(&mut app);
        assert!(
            rendered.contains("hal9000"),
            "remote entries name their host: {rendered:?}"
        );
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn panel_shows_the_runtime_pipeline_when_present() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();

        let mut app = test_app_with_entry(Vec::new());
        app.gallery.entries[0].metadata.output_format = Some(mold_core::OutputFormat::Mp4);
        app.gallery.entries[0].metadata.pipeline = Some(mold_core::Ltx2PipelineMode::TwoStageHq);

        let rendered = render_to_string(&mut app);

        assert!(
            rendered.contains("Pipeline"),
            "pipeline label: {rendered:?}"
        );
        assert!(
            rendered.contains("two-stage-hq"),
            "runtime pipeline value: {rendered:?}"
        );
    }

    #[test]
    fn machine_label_counts_multi_host_prints() {
        let entry = GalleryEntry {
            path: PathBuf::from("multi.png"),
            metadata: test_metadata(),
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
            origins: vec![
                GalleryOrigin {
                    host_id: crate::hosts::LOCAL_HOST_ID.into(),
                    url: None,
                    name: crate::hosts::local_display_name().into(),
                },
                GalleryOrigin {
                    host_id: "hal9000-7680".into(),
                    url: Some("http://hal9000:7680".into()),
                    name: "hal9000".into(),
                },
            ],
        };
        assert_eq!(entry.machine_label(), "2 machines");
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn details_thumb_cache_is_keyed_by_selection_and_geometry() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();

        // Real thumbnail on disk so the cache populates.
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let entry_path = PathBuf::from(format!("details-thumb-cache-{unique}.png"));
        let thumb_path = crate::thumbnails::thumbnail_path(&entry_path);
        if let Some(parent) = thumb_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            64,
            64,
            image::Rgba([0, 255, 0, 255]),
        ))
        .save(&thumb_path)
        .unwrap();

        let mut app = test_app_with_entry(Vec::new());
        app.gallery.entries[0].path = entry_path;

        let _ = render_to_string(&mut app);
        let cached = app
            .gallery
            .details_thumb
            .as_ref()
            .map(|(i, w, h, _)| (*i, *w, *h));
        assert_eq!(
            cached,
            Some((0, DETAILS_PANEL_W - 2, THUMB_ROWS)),
            "one-slot cache holds the selected entry at the panel geometry"
        );

        std::fs::remove_file(&thumb_path).ok();
    }
}
