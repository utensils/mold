use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::app::App;

/// Render the info panel showing model details and system resources.
pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border())
        .title(" Info ")
        .title_style(theme.title())
        .style(Style::default().bg(theme.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    let ri = &app.resource_info;

    // Model family
    if !app.generate.model_description.is_empty() {
        let desc = &app.generate.model_description;
        let truncated = if desc.len() > inner.width as usize {
            format!("{}..", &desc[..inner.width as usize - 2])
        } else {
            desc.clone()
        };
        lines.push(Line::from(Span::styled(truncated, theme.dim())));
    }

    // Memory line
    if let Some(ref mem) = ri.memory_line {
        lines.push(Line::from(Span::styled(mem.as_str(), theme.dim())));
    }

    // Process RAM
    if ri.process_ram_mb > 0 {
        lines.push(Line::from(Span::styled(
            format!("Mold: {} MB RAM", ri.process_ram_mb),
            theme.dim(),
        )));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled("No info available", theme.dim())));
    }

    // Only show lines that fit
    let visible = lines
        .into_iter()
        .take(inner.height as usize)
        .collect::<Vec<_>>();
    let paragraph = Paragraph::new(visible);
    frame.render_widget(paragraph, inner);
}

/// System resource snapshot, refreshed periodically.
#[derive(Debug, Default)]
pub struct ResourceInfo {
    /// Human-readable memory status (e.g., "VRAM: 16.2 GB free" or "Memory: 24.0 GB available")
    pub memory_line: Option<String>,
    /// Total RAM used by all mold processes in MB.
    pub process_ram_mb: u64,
}

impl ResourceInfo {
    /// Refresh resource info. Called every few seconds from the event loop.
    pub fn refresh(&mut self) {
        // System memory / VRAM
        self.memory_line = mold_inference::device::memory_status_string();

        // Total RAM across all mold processes (TUI + server + any workers)
        use sysinfo::{ProcessesToUpdate, System};
        let mut sys = System::new();
        sys.refresh_processes(ProcessesToUpdate::All, true);
        let mut total_bytes: u64 = 0;
        for proc in sys.processes().values() {
            let name = proc.name().to_string_lossy();
            if name == "mold" || name.starts_with("mold") {
                total_bytes += proc.memory();
            }
        }
        self.process_ram_mb = total_bytes / (1024 * 1024);
    }
}
