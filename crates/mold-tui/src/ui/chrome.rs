//! Studio chrome — the tab strip, activity strip, and status bar.
//!
//! All three bars render purely into the frame buffer (no direct terminal
//! writes), matching the Mold Studio TUI mockup: a two-row tab strip
//! (workspace labels + an accent underline under the active tab, host chip
//! right-aligned), a one-line activity strip carrying live job · throughput
//! · queue depth, and the key-hint status bar. Geometry for mouse
//! hit-testing is single-sourced in [`tab_spans`].

use std::ops::Range;

use ratatui::prelude::*;

use crate::action::View;
use crate::app::{App, InferenceMode};

/// Braille spinner frames — the shipping glyph vocabulary (no emoji).
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Milliseconds per spinner frame (the mockup animates at ~90ms).
const SPINNER_FRAME_MS: u128 = 90;

/// Current spinner frame, derived from the generation start time so no
/// extra ticking state is needed — the event loop already redraws every
/// ≤16ms while a job runs.
pub fn spinner_glyph(app: &App) -> &'static str {
    let elapsed = app
        .generate
        .progress
        .generation_started_at
        .map(|t| t.elapsed().as_millis())
        .unwrap_or(0);
    SPINNER[((elapsed / SPINNER_FRAME_MS) % SPINNER.len() as u128) as usize]
}

// ── Tab strip ───────────────────────────────────────────────────────

/// Click zones for every tab, as half-open `Range<u16>` column offsets
/// relative to the strip's left edge. Mirrors exactly what
/// [`render_tab_strip`] draws: one column of leading padding, then per tab
/// `pad(1) + " N Label "(label+4) + pad(1)`, with a one-column divider
/// folded into the preceding tab's zone (no dead pixels between tabs).
pub fn tab_spans() -> Vec<(View, Range<u16>)> {
    let mut spans = Vec::with_capacity(View::ALL.len());
    let mut offset: u16 = 1; // leading strip padding
    let last = View::ALL.len() - 1;
    for (i, view) in View::ALL.iter().enumerate() {
        let tab_width = view.label().len() as u16 + 6;
        let zone = if i == last { tab_width } else { tab_width + 1 };
        spans.push((*view, offset..offset + zone));
        offset += zone;
    }
    spans
}

/// Pure hit-test for the tab strip. Given an absolute column and the
/// strip's left edge, return the tab the click lands on. Clicks left of
/// the first tab select it; clicks past the last tab (host chip, version,
/// blank) return `None`.
pub fn tab_at_column(col: u16, strip_x: u16) -> Option<View> {
    let x = col.saturating_sub(strip_x);
    if x == 0 {
        return Some(View::ALL[0]);
    }
    tab_spans()
        .into_iter()
        .find(|(_, zone)| zone.contains(&x))
        .map(|(view, _)| view)
}

/// Connection state of the host chip in the tab strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChipState {
    Ready,
    Connecting,
    Offline,
}

/// The host chip: `● {host} · {backend}` on the right of the tab strip.
pub fn host_chip(app: &App) -> (ChipState, String) {
    if app.generate.params.inference_mode == InferenceMode::Local {
        let host = if cfg!(target_os = "macos") {
            "This Mac"
        } else {
            "This machine"
        };
        return (ChipState::Ready, format!("{host} · {}", local_backend()));
    }
    if let Some(status) = &app.resource_info.server_status {
        let host = status.hostname.as_deref().unwrap_or("remote");
        let backend = remote_backend(status);
        let text = match backend {
            Some(b) => format!("{host} · {b}"),
            None => host.to_string(),
        };
        return (ChipState::Ready, text);
    }
    if app.connecting {
        return (ChipState::Connecting, "connecting…".into());
    }
    (ChipState::Offline, "offline".into())
}

/// Backend label for the local engine, from compile-time features.
pub(crate) fn local_backend() -> &'static str {
    if cfg!(feature = "metal") {
        "Metal"
    } else if cfg!(feature = "cuda") {
        "CUDA"
    } else {
        "CPU"
    }
}

/// Backend label for a remote server — prefers the additive
/// `gpu_info.backend` field, falling back to GPU-name inference for older
/// servers (the same rule the desktop app uses). Shared with the
/// Machines workspace's host detail lines.
pub(crate) fn remote_backend(status: &mold_core::ServerStatus) -> Option<&'static str> {
    let backend = status.gpu_info.as_ref().and_then(|g| g.backend);
    if let Some(b) = backend {
        return Some(match b {
            mold_core::GpuBackend::Cuda => "CUDA",
            mold_core::GpuBackend::Metal => "Metal",
        });
    }
    let name = status
        .gpu_info
        .as_ref()
        .map(|g| g.name.to_ascii_lowercase())
        .or_else(|| {
            status
                .gpus
                .as_ref()
                .and_then(|gpus| gpus.first())
                .map(|gpu| gpu.name.to_ascii_lowercase())
        })
        .unwrap_or_default();
    if name.contains("nvidia")
        || name.contains("rtx")
        || name.contains("gtx")
        || name.contains("tesla")
        || name.contains("a100")
        || name.contains("h100")
    {
        Some("CUDA")
    } else if name.contains("apple") {
        Some("Metal")
    } else {
        None
    }
}

/// Render the two-row tab strip: workspace tabs + underline row.
pub fn render_tab_strip(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;
    if area.height == 0 || area.width == 0 {
        return;
    }

    // Row 0 — tabs on the frame plane.
    let tabs_row = Rect { height: 1, ..area };
    frame.render_widget(
        ratatui::widgets::Block::default().style(Style::default().bg(theme.frame)),
        area,
    );

    let mut spans: Vec<Span> = vec![Span::styled(" ", Style::default().bg(theme.frame))];
    for (i, view) in View::ALL.iter().enumerate() {
        let active = *view == app.active_view;
        let num_style = Style::default().bg(theme.frame).fg(if active {
            theme.tab_active
        } else {
            theme.faint
        });
        let label_style = Style::default()
            .bg(theme.frame)
            .fg(if active {
                theme.tab_active
            } else {
                theme.tab_inactive
            })
            .add_modifier(Modifier::BOLD);
        // pad(1) + " N Label " + pad(1) — must mirror `tab_spans`.
        spans.push(Span::styled("  ", num_style));
        spans.push(Span::styled(format!("{} ", view.index() + 1), num_style));
        spans.push(Span::styled(view.label().to_string(), label_style));
        spans.push(Span::styled("  ", num_style));
        if i != View::ALL.len() - 1 {
            spans.push(Span::styled(" ", num_style));
        }
    }
    frame.render_widget(
        ratatui::widgets::Paragraph::new(Line::from(spans)),
        tabs_row,
    );

    // Right side of row 0 — host chip (+ version when roomy).
    let (chip_state, chip_text) = host_chip(app);
    let dot_color = match chip_state {
        ChipState::Ready => theme.success,
        ChipState::Connecting => theme.warning,
        ChipState::Offline => theme.error,
    };
    let mut right_spans: Vec<Span> = Vec::new();
    if area.width >= 100 {
        right_spans.push(Span::styled(
            format!("mold {} ", mold_core::build_info::version_string()),
            Style::default().bg(theme.frame).fg(theme.faint),
        ));
    }
    right_spans.push(Span::styled(
        "● ",
        Style::default().bg(theme.frame).fg(dot_color),
    ));
    right_spans.push(Span::styled(
        format!("{chip_text} "),
        Style::default().bg(theme.frame).fg(theme.text_dim),
    ));
    frame.render_widget(
        ratatui::widgets::Paragraph::new(Line::from(right_spans).right_aligned()),
        tabs_row,
    );

    // Row 1 — border with an accent underline beneath the active label.
    if area.height >= 2 {
        let underline_row = Rect {
            y: area.y + 1,
            height: 1,
            ..area
        };
        let mut line = vec![
            Span::styled(
                "─".repeat(area.width as usize),
                Style::default().bg(theme.frame).fg(theme.border),
            );
            1
        ];
        frame.render_widget(
            ratatui::widgets::Paragraph::new(Line::from(std::mem::take(&mut line))),
            underline_row,
        );
        if let Some((_, zone)) = tab_spans()
            .into_iter()
            .find(|(view, _)| *view == app.active_view)
        {
            // Underline the title portion (skip the per-tab pads).
            let start = area.x + zone.start + 1;
            let width = (zone.end - zone.start).saturating_sub(
                if app.active_view == *View::ALL.last().unwrap() {
                    2
                } else {
                    3
                },
            );
            let ul = Rect {
                x: start,
                y: area.y + 1,
                width: width.min(area.width.saturating_sub(zone.start + 1)),
                height: 1,
            };
            frame.render_widget(
                ratatui::widgets::Paragraph::new(Span::styled(
                    "━".repeat(ul.width as usize),
                    Style::default().bg(theme.frame).fg(theme.accent),
                )),
                ul,
            );
        }
    }
}

// ── Activity strip ──────────────────────────────────────────────────

/// Which state the activity strip is reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityKind {
    Generating,
    Done,
    Error,
    Idle,
}

/// Compose the activity strip's line. Pure so it is testable; segments
/// with unknown values are omitted rather than faked (queue depth only
/// appears when a server reports it).
pub fn activity_line(app: &App) -> (ActivityKind, String) {
    let model_name = mold_core::ModelInfoExtended::human_name_for(
        &app.generate.params.model,
        &app.models.catalog,
    );
    let queue_seg = app
        .resource_info
        .server_status
        .as_ref()
        .and_then(|s| s.queue_depth)
        .map(|n| format!(" · queue {n}"))
        .unwrap_or_default();

    if app.generate.generating {
        let p = &app.generate.progress;
        if p.is_downloading() {
            return (
                ActivityKind::Generating,
                format!("pulling · {}", p.download_status_text()),
            );
        }
        let mut s = format!("developing · {model_name}");
        if p.denoise_total > 0 {
            s.push_str(&format!(" · {}/{}", p.denoise_step, p.denoise_total));
        }
        if p.denoise_step > 0 && p.denoise_elapsed_ms > 0 {
            let rate = p.denoise_step as f64 / (p.denoise_elapsed_ms as f64 / 1000.0);
            s.push_str(&format!(" · {rate:.1} it/s"));
        }
        s.push_str(&queue_seg);
        return (ActivityKind::Generating, s);
    }

    if let Some(msg) = &app.generate.error_message {
        let first = msg.lines().next().unwrap_or_default();
        return (
            ActivityKind::Error,
            format!(
                "error · {}",
                super::widgets::truncate_with_ellipsis(first, 100)
            ),
        );
    }

    if let Some(ms) = app.generate.last_generation_time_ms {
        let mut s = format!("done · {:.1}s", ms as f64 / 1000.0);
        if let Some(dir) = app
            .generate
            .last_output_path
            .as_ref()
            .and_then(|p| p.parent())
        {
            s.push_str(&format!(" · saved to {}", tildify(dir)));
        }
        s.push_str(&queue_seg);
        return (ActivityKind::Done, s);
    }

    let mut s = format!("idle · {model_name}");
    let (_, chip) = host_chip(app);
    s.push_str(&format!(" · {chip}"));
    if let Some(mem) = app
        .resource_info
        .server_status
        .as_ref()
        .and_then(|st| st.memory_status.as_deref())
        .or(app.resource_info.memory_line.as_deref())
    {
        s.push_str(&format!(" · {mem}"));
    }
    (ActivityKind::Idle, s)
}

/// Shorten a path by replacing the home-directory prefix with `~`.
fn tildify(path: &std::path::Path) -> String {
    if let Some(home) = dirs::home_dir() {
        if let Ok(rest) = path.strip_prefix(&home) {
            return format!("~/{}", rest.display());
        }
    }
    path.display().to_string()
}

/// Render the one-line activity strip.
pub fn render_activity_strip(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;
    if area.height == 0 || area.width == 0 {
        return;
    }
    let (kind, text) = activity_line(app);
    let (glyph, glyph_color): (&str, Color) = match kind {
        ActivityKind::Generating => (spinner_glyph(app), theme.accent),
        ActivityKind::Done => ("✓", theme.success),
        ActivityKind::Error => ("•", theme.error),
        ActivityKind::Idle => ("•", theme.info),
    };
    let line = Line::from(vec![
        Span::styled(" ", theme.chrome()),
        Span::styled(glyph, Style::default().bg(theme.frame).fg(glyph_color)),
        Span::styled(" ", theme.chrome()),
        Span::styled(text, theme.chrome()),
    ]);
    frame.render_widget(
        ratatui::widgets::Paragraph::new(line).style(theme.chrome()),
        area,
    );
}

// ── Status bar ──────────────────────────────────────────────────────

/// Render the key-hint status bar. The hint tables live in
/// [`crate::ui::status_shortcuts`] beside their contract tests.
pub fn render_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;
    let shortcuts = super::status_shortcuts(app);

    let mut spans = Vec::new();
    spans.push(Span::styled(" ", theme.chrome()));
    for (i, (key, desc)) in shortcuts.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  ", theme.chrome()));
        }
        if !key.is_empty() {
            spans.push(Span::styled(key.to_string(), theme.chrome_key()));
            spans.push(Span::styled(" ", theme.chrome()));
        }
        spans.push(Span::styled(desc.to_string(), theme.chrome()));
    }

    let bar = ratatui::widgets::Paragraph::new(Line::from(spans)).style(theme.chrome());
    frame.render_widget(bar, area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tab_spans_cover_contiguous_zones() {
        let spans = tab_spans();
        assert_eq!(spans.len(), 5);
        // Zones tile the strip with no gaps: each zone starts where the
        // previous ended; the first starts at column 1 (leading pad).
        let mut expected_start = 1u16;
        for (_, zone) in &spans {
            assert_eq!(zone.start, expected_start);
            expected_start = zone.end;
        }
    }

    #[test]
    fn tab_at_column_matches_spans_property() {
        // Every column inside a span hit-tests to that view; columns past
        // the last tab return None; column 0 folds into the first tab.
        assert_eq!(tab_at_column(0, 0), Some(View::Create));
        for (view, zone) in tab_spans() {
            for col in zone.clone() {
                assert_eq!(tab_at_column(col, 0), Some(view), "col {col}");
            }
        }
        let end = tab_spans().last().unwrap().1.end;
        assert_eq!(tab_at_column(end, 0), None);
        assert_eq!(tab_at_column(end + 40, 0), None);
        // Offset strips shift with their left edge.
        assert_eq!(tab_at_column(5 + 3, 5), tab_at_column(3, 0));
    }

    #[test]
    fn spinner_frames_are_braille_only() {
        for f in SPINNER {
            assert_eq!(f.chars().count(), 1);
            let c = f.chars().next().unwrap();
            assert!(('\u{2800}'..='\u{28FF}').contains(&c), "{f} not braille");
        }
    }

    #[test]
    fn remote_backend_uses_worker_names_when_legacy_gpu_info_is_absent() {
        let status = mold_core::ServerStatus {
            version: "0.20.2".into(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: false,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 0,
            hostname: None,
            memory_status: None,
            gpus: Some(vec![mold_core::GpuWorkerStatus {
                ordinal: 7,
                name: "NVIDIA B200".into(),
                vram_total_bytes: 80 * 1024_u64.pow(3),
                vram_used_bytes: 0,
                loaded_model: None,
                state: mold_core::GpuWorkerState::Idle,
            }]),
            queue_depth: None,
            queue_capacity: None,
            queue_paused: None,
            instance_id: None,
            models_disk: None,
            host_memory: None,
        };

        assert_eq!(remote_backend(&status), Some("CUDA"));
    }
}
