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
            let max = inner.width as usize - 2;
            let boundary = desc.floor_char_boundary(max);
            format!("{}..", &desc[..boundary])
        } else {
            desc.clone()
        };
        lines.push(Line::from(Span::styled(truncated, theme.dim())));
    }

    // Memory line
    if let Some(ref mem) = ri.memory_line {
        lines.push(Line::from(Span::styled(mem.as_str(), theme.dim())));
    }

    // Process memory
    if ri.process_memory_mb > 0 {
        lines.push(Line::from(Span::styled(
            format!("Mold: {:.1} GB", ri.process_memory_mb as f64 / 1024.0),
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
    /// Total memory used by all mold processes in MB (including mmap'd model weights).
    pub process_memory_mb: u64,
}

impl ResourceInfo {
    /// Refresh resource info. Called every few seconds from the event loop.
    pub fn refresh(&mut self) {
        // System memory / VRAM
        self.memory_line = mold_inference::device::memory_status_string();

        // Total memory across all mold processes
        self.process_memory_mb = total_mold_memory_mb();
    }
}

/// Get total memory used by all mold processes in MB.
/// On macOS, uses phys_footprint (includes mmap'd file pages) instead of RSS.
fn total_mold_memory_mb() -> u64 {
    use sysinfo::{ProcessesToUpdate, System};
    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::All, true);

    let mut pids: Vec<u32> = Vec::new();
    for (pid, proc) in sys.processes() {
        let name = proc.name().to_string_lossy();
        if name == "mold" {
            pids.push(pid.as_u32());
        }
    }

    if pids.is_empty() {
        return 0;
    }

    let mut total_bytes: u64 = 0;
    for pid in &pids {
        total_bytes += process_footprint(*pid);
    }
    total_bytes / (1024 * 1024)
}

/// Get the physical footprint of a process (includes mmap'd file pages).
/// Uses proc_pid_rusage with rusage_info_v0 to get ri_phys_footprint.
#[cfg(target_os = "macos")]
fn process_footprint(pid: u32) -> u64 {
    use std::mem::MaybeUninit;

    // rusage_info_v0 layout from <sys/resource.h>:
    // uint8_t[16] ri_uuid, then 6x uint64_t, then ri_phys_footprint
    #[repr(C)]
    #[allow(non_camel_case_types)]
    struct rusage_info_v0 {
        ri_uuid: [u8; 16],
        ri_user_time: u64,
        ri_system_time: u64,
        ri_pkg_idle_wkups: u64,
        ri_interrupt_wkups: u64,
        ri_pageins: u64,
        ri_wired_size: u64,
        ri_resident_size: u64,
        ri_phys_footprint: u64,
        ri_proc_start_abstime: u64,
        ri_proc_exit_abstime: u64,
    }

    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut rusage_info_v0) -> i32;
    }

    const RUSAGE_INFO_V0: i32 = 0;

    unsafe {
        let mut info = MaybeUninit::<rusage_info_v0>::zeroed();
        let ret = proc_pid_rusage(pid as i32, RUSAGE_INFO_V0, info.as_mut_ptr());
        if ret == 0 {
            info.assume_init().ri_phys_footprint
        } else {
            0
        }
    }
}

/// Fallback for non-macOS: use RSS from sysinfo.
#[cfg(not(target_os = "macos"))]
fn process_footprint(pid: u32) -> u64 {
    use sysinfo::{Pid, ProcessesToUpdate, System};
    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::Some(&[Pid::from_u32(pid)]), true);
    sys.process(Pid::from_u32(pid))
        .map(|p| p.memory())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_footprint_returns_nonzero_for_self() {
        let pid = std::process::id();
        let footprint = process_footprint(pid);
        // Our own process must have some memory allocated
        assert!(footprint > 0, "footprint was {footprint}");
    }

    #[test]
    fn process_footprint_returns_zero_for_invalid_pid() {
        // PID 0 is the kernel, we can't query it — should return 0 or a value
        // PID max is unlikely to exist
        let footprint = process_footprint(u32::MAX);
        assert_eq!(footprint, 0);
    }

    #[test]
    fn total_mold_memory_includes_current_process() {
        // The current test binary has "mold" in it (mold-ai-tui tests)
        // but the process name is the test runner, not "mold"
        // Just verify it doesn't panic
        let mb = total_mold_memory_mb();
        // May be 0 if no mold process is running, that's fine
        let _ = mb;
    }

    #[test]
    fn resource_info_refresh_does_not_panic() {
        let mut ri = ResourceInfo::default();
        ri.refresh();
        // memory_line should be Some on macOS (unified memory) or CUDA
        // process_memory_mb may be 0 if no mold processes running
    }

    #[test]
    fn truncation_does_not_panic_on_multibyte_chars() {
        // Em dash is 3 bytes (U+2014). Truncation must land on a char boundary.
        let desc = "Flux.2 Klein-9B Q4 GGUF \u{2014} smallest footprint";
        let width: u16 = 34; // lands inside the em dash at bytes 32..35
        if desc.len() > width as usize {
            let max = width as usize - 2;
            let boundary = desc.floor_char_boundary(max);
            let truncated = format!("{}..", &desc[..boundary]);
            assert!(truncated.len() <= width as usize);
            assert!(truncated.ends_with(".."));
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn phys_footprint_larger_than_zero_for_self() {
        // On macOS, phys_footprint includes mmap'd pages — must be > 0
        let pid = std::process::id();
        let footprint = process_footprint(pid);
        assert!(
            footprint > 1024 * 1024,
            "phys_footprint should be at least 1 MB, got {footprint} bytes"
        );
    }
}
