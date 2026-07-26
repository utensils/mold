//! Machines workspace — the multi-host registry view.
//!
//! Left pane: one row per machine, the local machine always first, each
//! with a status dot (success ready / warning connecting / error
//! offline), display name, and a dim hardware detail line. Right pane:
//! the selected machine's telemetry key-values plus its queue lanes
//! (running `▶`, queued `●`), fed by `MoldClient::list_queue`. The
//! local row composes its lane from the in-flight generation and recent
//! prompt history — this view absorbed the old read-only Queue tab.

use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

use crate::app::App;
use crate::hosts::{GenTarget, HostHealth, MachineRowId, MachinesFocus, MachinesState};

use super::widgets::{panel_block, truncate_with_ellipsis};

/// Left host-list column width (mock geometry: `[Min(34), Min(30)]`).
const HOST_LIST_MIN_W: u16 = 34;
const DETAIL_MIN_W: u16 = 30;
/// Label column width for the detail pane's KV rows.
const KV_LABEL_W: usize = 10;

/// Owned-value variant of `widgets::kv_row_line` — the detail pane
/// formats its values on the fly, so the line must own them.
fn kv_owned(
    theme: &crate::ui::theme::Theme,
    label: &str,
    value: String,
    muted: bool,
) -> Line<'static> {
    let value_style = if muted {
        theme.dim()
    } else {
        theme.param_value()
    };
    Line::from(vec![
        Span::styled(
            format!("{:<w$}", label, w = KV_LABEL_W),
            theme.param_label(),
        ),
        Span::styled(value, value_style),
    ])
}

/// Render the Machines workspace.
pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    let [list_area, detail_area] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(HOST_LIST_MIN_W),
            Constraint::Min(DETAIL_MIN_W),
        ])
        .areas(area);

    render_host_list(frame, app, list_area);
    render_detail(frame, app, detail_area);
}

/// Row ids in display order — the local machine is always first.
pub(crate) fn row_ids(state: &MachinesState) -> Vec<MachineRowId> {
    let mut rows = Vec::with_capacity(state.row_count());
    rows.push(MachineRowId::Local);
    rows.extend(
        state
            .registry
            .hosts
            .iter()
            .map(|h| MachineRowId::Host(h.id.clone())),
    );
    rows
}

fn render_host_list(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;
    let st = &app.machines;
    let hint = format!("{} machines · c Connect", st.row_count());
    let block = panel_block(
        theme,
        "Machines",
        st.focus == MachinesFocus::HostList,
        Some(&hint),
    );
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    for (i, row) in row_ids(st).into_iter().enumerate() {
        let selected = i == st.selected;
        let (dot_color, name, right, detail) = match &row {
            MachineRowId::Local => (
                theme.success,
                crate::hosts::local_display_name().to_string(),
                "ready".to_string(),
                local_detail_line(
                    app.resource_info.memory_line.as_deref(),
                    &local_server_label(app.server_url.as_deref(), app.server_process.is_some()),
                ),
            ),
            MachineRowId::Host(id) => {
                let entry = st.registry.get(id);
                let name = entry
                    .map(|e| e.display_name())
                    .unwrap_or_else(|| id.clone());
                let (color, label) = match st.statuses.get(id).map(|s| &s.health) {
                    Some(HostHealth::Ready) => (theme.success, "ready"),
                    Some(HostHealth::Offline(_)) => (theme.error, "offline"),
                    _ => (theme.warning, "connecting…"),
                };
                let detail = host_detail_line(
                    entry.map(|e| e.url.as_str()).unwrap_or(""),
                    st.statuses.get(id).and_then(|s| s.status.as_deref()),
                );
                (color, name, label.to_string(), detail)
            }
        };

        let is_target = match (&row, &app.target) {
            (MachineRowId::Local, GenTarget::Local) => true,
            (MachineRowId::Host(id), GenTarget::Host(t)) => id == t,
            _ => false,
        };

        let name_style = if selected {
            theme.list_selected()
        } else {
            Style::default().fg(theme.text)
        };
        let mut spans = vec![
            Span::styled("● ", Style::default().fg(dot_color)),
            Span::styled(name.clone(), name_style.add_modifier(Modifier::BOLD)),
        ];
        if is_target {
            spans.push(Span::styled(" · target", Style::default().fg(theme.accent)));
        }
        // Right-aligned status text.
        let used: usize = 2 + name.chars().count() + if is_target { " · target".len() } else { 0 };
        let pad = (inner.width as usize).saturating_sub(used + right.chars().count() + 1);
        spans.push(Span::raw(" ".repeat(pad)));
        spans.push(Span::styled(right, theme.dim()));
        lines.push(Line::from(spans));
        lines.push(Line::from(Span::styled(
            format!(
                "  {}",
                truncate_with_ellipsis(&detail, inner.width.saturating_sub(2) as usize)
            ),
            theme.dim(),
        )));
        lines.push(Line::default());
    }

    lines.push(Line::from(Span::styled(
        "+ Connect a machine  (c)",
        theme.dim(),
    )));

    frame.render_widget(
        Paragraph::new(
            lines
                .into_iter()
                .take(inner.height as usize)
                .collect::<Vec<_>>(),
        ),
        inner,
    );
}

fn render_detail(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;
    let st = &app.machines;
    let title = match st.selected_row() {
        MachineRowId::Local => crate::hosts::local_display_name().to_string(),
        MachineRowId::Host(id) => st.registry.get(&id).map(|e| e.display_name()).unwrap_or(id),
    };
    let block = panel_block(theme, &title, st.focus == MachinesFocus::Detail, None);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    match st.selected_row() {
        MachineRowId::Local => build_local_detail(app, &mut lines),
        MachineRowId::Host(id) => build_host_detail(app, &id, &mut lines),
    }

    frame.render_widget(
        Paragraph::new(
            lines
                .into_iter()
                .take(inner.height as usize)
                .collect::<Vec<_>>(),
        ),
        inner,
    );
}

fn build_local_detail(app: &App, lines: &mut Vec<Line>) {
    let theme = &app.theme;
    let ri = &app.resource_info;

    lines.push(kv_owned(
        theme,
        "Backend",
        local_backend_label().to_string(),
        false,
    ));
    if let Some(mem) = ri.memory_line.clone() {
        lines.push(kv_owned(theme, "Memory", mem, false));
    }
    if ri.process_memory_mb > 0 {
        let mold_mem = format!("{:.1} GB", ri.process_memory_mb as f64 / 1024.0);
        lines.push(kv_owned(theme, "Mold", mold_mem, false));
    }
    let server = local_server_label(app.server_url.as_deref(), app.server_process.is_some());
    lines.push(kv_owned(theme, "Server", server, false));
    lines.push(kv_owned(
        theme,
        "Version",
        mold_core::build_info::version_string(),
        true,
    ));
    render_devices(app, crate::hosts::LOCAL_HOST_ID, lines);

    // Lane: the in-flight generation plus recent prompt history.
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        "Queue",
        app.theme.dim().add_modifier(Modifier::BOLD),
    )));
    let mut any = false;
    if app.generate.generating {
        any = true;
        let first_line = app
            .generate
            .prompt
            .lines()
            .iter()
            .map(|s| s.trim())
            .find(|s| !s.is_empty())
            .unwrap_or("");
        lines.push(Line::from(vec![
            Span::styled("▶ ", theme.warning()),
            Span::styled(
                format!(
                    "{} · {}",
                    mold_core::ModelInfoExtended::human_name_for(
                        &app.generate.params.model,
                        &app.models.catalog,
                    ),
                    running_time_label(
                        app.generate.progress.denoise_step,
                        app.generate.progress.denoise_total,
                    )
                ),
                Style::default().fg(app.theme.text),
            ),
            Span::styled(format!("  {}", prompt_preview(first_line)), theme.dim()),
        ]));
    }
    for entry in app.history.recent(5) {
        any = true;
        let model_name = if entry.model.is_empty() {
            "—".to_string()
        } else {
            mold_core::ModelInfoExtended::human_name_for(&entry.model, &app.models.catalog)
        };
        lines.push(Line::from(vec![
            Span::styled("✓ ", theme.success()),
            Span::styled(
                format!("{} · {}", model_name, relative_time(entry.timestamp)),
                theme.dim(),
            ),
            Span::styled(format!("  {}", prompt_preview(&entry.prompt)), theme.dim()),
        ]));
    }
    if !any {
        lines.push(Line::from(Span::styled(
            "— idle. no runs this session.",
            theme.dim(),
        )));
    }
}

fn build_host_detail(app: &App, host_id: &str, lines: &mut Vec<Line>) {
    let theme = &app.theme;
    let st = &app.machines;
    let health = st.statuses.get(host_id).map(|s| &s.health);
    let status = st.statuses.get(host_id).and_then(|s| s.status.as_deref());

    match (health, status) {
        (Some(HostHealth::Offline(reason)), _) => {
            lines.push(Line::from(Span::styled(
                format!("offline — {reason}"),
                theme.error(),
            )));
            if let Some(entry) = st.registry.get(host_id) {
                lines.push(Line::from(Span::styled(
                    host_port_hint(&entry.url),
                    theme.dim(),
                )));
            }
            lines.push(Line::default());
            lines.push(Line::from(Span::styled(
                "r Refresh · d Forget",
                theme.dim(),
            )));
            return;
        }
        (_, None) => {
            lines.push(Line::from(Span::styled("connecting…", theme.dim())));
            return;
        }
        (_, Some(status)) => {
            if st.devices.contains_key(host_id) {
                render_devices(app, host_id, lines);
            } else if let Some(gpus) = status.gpus.as_ref().filter(|gpus| !gpus.is_empty()) {
                for gpu in gpus {
                    lines.push(kv_owned(
                        theme,
                        &format!("GPU {}", gpu.ordinal),
                        gpu.name.clone(),
                        false,
                    ));
                    let vram = vram_label(
                        gpu.vram_used_bytes / 1024_u64.pow(2),
                        gpu.vram_total_bytes / 1024_u64.pow(2),
                    );
                    lines.push(kv_owned(
                        theme,
                        &format!("VRAM {}", gpu.ordinal),
                        vram,
                        false,
                    ));
                }
            } else if let Some(gpu) = &status.gpu_info {
                lines.push(kv_owned(theme, "GPU", gpu.name.clone(), false));
                let vram = vram_label(gpu.vram_used_mb, gpu.vram_total_mb);
                lines.push(kv_owned(theme, "VRAM", vram, false));
            }
            if let Some(disk) = &status.models_disk {
                let d = disk_label(disk.free_bytes, disk.total_bytes);
                lines.push(kv_owned(theme, "Models", d, false));
            }
            let loaded = if status.models_loaded.is_empty() {
                "none".to_string()
            } else {
                status
                    .models_loaded
                    .iter()
                    .map(|name| {
                        mold_core::ModelInfoExtended::human_name_for(name, &app.models.catalog)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            lines.push(kv_owned(theme, "Loaded", loaded, false));
            lines.push(kv_owned(
                theme,
                "Uptime",
                format_uptime(status.uptime_secs),
                false,
            ));
            lines.push(kv_owned(theme, "Version", status.version.clone(), true));
        }
    }

    // Queue lanes for the selected host.
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        "Queue",
        theme.dim().add_modifier(Modifier::BOLD),
    )));
    let listing = st.queue.as_ref().filter(|(id, _)| id == host_id);
    match listing {
        Some((_, listing)) if !listing.entries.is_empty() => {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            for (i, job) in listing.entries.iter().enumerate() {
                let selected = st.focus == MachinesFocus::Detail && i == st.queue_selected;
                let text = if job.state == "running" {
                    running_lane(
                        &mold_core::ModelInfoExtended::human_name_for(
                            &job.model,
                            &app.models.catalog,
                        ),
                        &queue_elapsed_label(job.started_at_unix_ms, now_ms),
                        job.gpu,
                    )
                } else {
                    queued_lane(
                        &mold_core::ModelInfoExtended::human_name_for(
                            &job.model,
                            &app.models.catalog,
                        ),
                        job.position,
                    )
                };
                let style = if selected {
                    theme.list_selected()
                } else if job.state == "running" {
                    theme.warning()
                } else {
                    theme.dim()
                };
                lines.push(Line::from(Span::styled(text, style)));
            }
            lines.push(Line::default());
            lines.push(Line::from(Span::styled(
                "Tab Focus lanes · x Cancel queued",
                theme.dim(),
            )));
        }
        _ => {
            lines.push(Line::from(Span::styled("— queue empty.", theme.dim())));
        }
    }
}

fn render_devices(app: &App, host_id: &str, lines: &mut Vec<Line>) {
    let Some(state) = app.machines.devices.get(host_id) else {
        return;
    };
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        "GPU devices",
        app.theme.dim().add_modifier(Modifier::BOLD),
    )));
    if state.devices.is_empty() {
        lines.push(Line::from(Span::styled("— none visible.", app.theme.dim())));
        return;
    }
    for (index, device) in state.devices.iter().enumerate() {
        let selected =
            app.machines.focus == MachinesFocus::Detail && index == app.machines.device_selected;
        let marker = if selected { "›" } else { " " };
        let ordinal = device
            .ordinal
            .map(|ordinal| format!("GPU {ordinal}"))
            .unwrap_or_else(|| match device.backend {
                mold_core::GpuBackend::Cuda => "CUDA".to_string(),
                mold_core::GpuBackend::Metal => "METAL".to_string(),
            });
        let state = device_state_label(device);
        let style = if selected {
            app.theme.list_selected()
        } else if device.admin_state == mold_core::DeviceAdminState::Draining {
            app.theme.warning()
        } else if device.health != mold_core::DeviceHealth::Healthy {
            app.theme.error()
        } else {
            Style::default().fg(app.theme.text)
        };
        lines.push(Line::from(Span::styled(
            format!("{marker} {ordinal} · {} · {state}", device.name),
            style,
        )));
        let vram = match (device.memory.used_bytes, device.memory.total_bytes) {
            (Some(used), Some(total)) => {
                vram_label(used / 1024_u64.pow(2), total / 1024_u64.pow(2))
            }
            (_, Some(total)) => format!("{:.1} GB total", total as f64 / 1_073_741_824.0),
            _ => "VRAM unavailable".to_string(),
        };
        let utilization = device
            .telemetry
            .utilization_percent
            .map(|value| format!(" · {value:.0}%"))
            .unwrap_or_default();
        lines.push(Line::from(Span::styled(
            format!("  {vram}{utilization} · {}", short_device_id(&device.id)),
            app.theme.dim(),
        )));
        if selected {
            if let Some(active) = &device.active_work_id {
                lines.push(Line::from(Span::styled(
                    format!("  active {active}"),
                    app.theme.warning(),
                )));
            }
            if !device.loaded_models.is_empty() {
                lines.push(Line::from(Span::styled(
                    format!("  loaded {}", device.loaded_models.join(", ")),
                    app.theme.dim(),
                )));
            }
        }
    }
    lines.push(Line::from(Span::styled(
        "g Next GPU · e Enable/disable",
        app.theme.dim(),
    )));
}

fn device_state_label(device: &mold_core::DeviceInfo) -> &'static str {
    use mold_core::{DeviceAdminState, DeviceHealth};
    match device.admin_state {
        DeviceAdminState::StartupExcluded => "startup excluded",
        DeviceAdminState::Starting => "starting",
        DeviceAdminState::Draining => "finishing current work",
        DeviceAdminState::Disabled => "disabled",
        DeviceAdminState::Enabled => match device.health {
            DeviceHealth::Healthy => "enabled",
            DeviceHealth::Degraded => "degraded",
            DeviceHealth::Unavailable => "unavailable",
            DeviceHealth::Poisoned => "poisoned",
        },
    }
}

fn short_device_id(id: &str) -> String {
    if id.len() <= 18 {
        id.to_string()
    } else {
        format!("{}…{}", &id[..11], &id[id.len() - 6..])
    }
}

// ── Pure formatting helpers (contract-tested) ───────────────────────

/// Backend label for the local engine — single-sourced with the chrome
/// host chip so the two surfaces can never disagree.
pub(crate) fn local_backend_label() -> &'static str {
    super::chrome::local_backend()
}

/// Detail line for a remote host row:
/// `"{gpu} · {vram} GB · {backend} · {host:port}"`, degrading gracefully
/// when the status (or its GPU info) is missing.
pub(crate) fn host_detail_line(url: &str, status: Option<&mold_core::ServerStatus>) -> String {
    let hostport = crate::hosts::host_port_label(url);
    let Some(status) = status else {
        return hostport;
    };
    let mut segs: Vec<String> = Vec::new();
    if let Some(gpus) = status.gpus.as_ref().filter(|gpus| !gpus.is_empty()) {
        let mut groups: Vec<(&str, usize)> = Vec::new();
        for gpu in gpus {
            if let Some((_, count)) = groups.iter_mut().find(|(name, _)| *name == gpu.name) {
                *count += 1;
            } else {
                groups.push((&gpu.name, 1));
            }
        }
        segs.push(
            groups
                .into_iter()
                .map(|(name, count)| {
                    if count > 1 {
                        format!("{count}× {name}")
                    } else {
                        name.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(" + "),
        );
        let total = gpus.iter().map(|gpu| gpu.vram_total_bytes).sum::<u64>();
        segs.push(format!("{:.0} GB", total as f64 / 1_073_741_824.0));
    } else if let Some(gpu) = &status.gpu_info {
        segs.push(gpu.name.clone());
        segs.push(format!("{:.0} GB", gpu.vram_total_mb as f64 / 1024.0));
    }
    if let Some(backend) = super::chrome::remote_backend(status) {
        segs.push(backend.to_string());
    }
    segs.push(hostport);
    segs.join(" · ")
}

/// Detail line for the local row: memory + backend + server route.
pub(crate) fn local_detail_line(memory_line: Option<&str>, server_label: &str) -> String {
    let mut segs: Vec<String> = Vec::new();
    if let Some(mem) = memory_line {
        segs.push(mem.to_string());
    }
    segs.push(local_backend_label().to_string());
    segs.push(server_label.to_string());
    segs.join(" · ")
}

/// How the local machine serves requests: the auto-spawned/loopback
/// server's `host:port` when one is attached, `"in-process"` otherwise.
/// A remote `server_url` (`--host`, `MOLD_HOST`) is NOT this machine's
/// server and must not appear on the local row.
pub(crate) fn local_server_label(server_url: Option<&str>, spawned: bool) -> String {
    match server_url {
        Some(url) if spawned || is_loopback_url(url) => crate::hosts::host_port_label(url),
        _ => "in-process".to_string(),
    }
}

fn is_loopback_url(url: &str) -> bool {
    let hp = crate::hosts::host_port_label(url);
    let host = hp.split(':').next().unwrap_or("");
    matches!(host, "localhost" | "127.0.0.1" | "::1" | "[::1]")
}

fn host_port_hint(url: &str) -> String {
    format!(
        "Check the host at {} in Machines.",
        crate::hosts::host_port_label(url)
    )
}

/// `"{used} / {total} GB"` from MB counts.
pub(crate) fn vram_label(used_mb: u64, total_mb: u64) -> String {
    format!(
        "{:.1} / {:.1} GB",
        used_mb as f64 / 1024.0,
        total_mb as f64 / 1024.0
    )
}

/// `"{free} GB free of {total} GB"` from byte counts.
pub(crate) fn disk_label(free_bytes: u64, total_bytes: u64) -> String {
    format!(
        "{:.1} GB free of {:.1} GB",
        free_bytes as f64 / 1_073_741_824.0,
        total_bytes as f64 / 1_073_741_824.0
    )
}

/// Compact uptime: `"42s"`, `"12m"`, `"3h 24m"`, `"2d 3h"`.
pub(crate) fn format_uptime(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m", secs / 60),
        3600..=86_399 => format!("{}h {}m", secs / 3600, (secs % 3600) / 60),
        _ => format!("{}d {}h", secs / 86_400, (secs % 86_400) / 3600),
    }
}

/// Running queue lane: `"▶ {model} · {elapsed} · gpu{N}"` (GPU segment
/// omitted when the server didn't report one).
pub(crate) fn running_lane(model: &str, elapsed: &str, gpu: Option<usize>) -> String {
    match gpu {
        Some(n) => format!("▶ {model} · {elapsed} · gpu{n}"),
        None => format!("▶ {model} · {elapsed}"),
    }
}

/// Queued lane: `"● {model} · #{pos}"` (1-based display position).
pub(crate) fn queued_lane(model: &str, position: usize) -> String {
    format!("● {model} · #{}", position + 1)
}

/// Elapsed label for a running job from its accepted-at timestamp.
pub(crate) fn queue_elapsed_label(started_at_unix_ms: u64, now_ms: u64) -> String {
    let secs = now_ms.saturating_sub(started_at_unix_ms) / 1000;
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m", secs / 60),
        _ => format!("{}h", secs / 3600),
    }
}

// ── Helpers absorbed from the retired ui/queue.rs ───────────────────

/// First line of a prompt, truncated for a table/lane cell.
pub(crate) fn prompt_preview(prompt: &str) -> String {
    const MAX: usize = 80;
    let first_line = prompt.lines().next().unwrap_or("").trim();
    if first_line.is_empty() {
        return "(empty)".to_string();
    }
    truncate_with_ellipsis(first_line, MAX)
}

/// Short status for a running job — the current denoise step if known.
pub(crate) fn running_time_label(step: usize, total: usize) -> String {
    if total > 0 {
        format!("{step}/{total}")
    } else {
        "…".to_string()
    }
}

/// Convert a UNIX-seconds timestamp into a compact relative label.
pub(crate) fn relative_time(ts: u64) -> String {
    if ts == 0 {
        return "—".to_string();
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let delta = now.saturating_sub(ts);
    match delta {
        0..=59 => "now".to_string(),
        60..=3599 => format!("{}m", delta / 60),
        3600..=86_399 => format!("{}h", delta / 3600),
        _ => format!("{}d", delta / 86_400),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hosts::HostRegistry;

    fn status(gpu: Option<mold_core::GpuInfo>) -> mold_core::ServerStatus {
        mold_core::ServerStatus {
            version: "0.20.2".into(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: false,
            current_generation: None,
            gpu_info: gpu,
            uptime_secs: 0,
            hostname: Some("hal9000".into()),
            memory_status: None,
            gpus: None,
            queue_depth: None,
            queue_capacity: None,
            queue_paused: None,
            instance_id: None,
            models_disk: None,
        }
    }

    // ── row ordering ────────────────────────────────────────────

    #[test]
    fn local_row_is_always_first() {
        let mut st = MachinesState::default();
        st.registry
            .add(crate::hosts::HostEntry {
                id: "hal9000-7680".into(),
                url: "http://hal9000:7680".into(),
                name: None,
                instance_id: None,
            })
            .unwrap();
        let rows = row_ids(&st);
        assert_eq!(rows[0], MachineRowId::Local);
        assert_eq!(rows[1], MachineRowId::Host("hal9000-7680".into()));

        // Even with no registered hosts, the local row is present.
        let empty = MachinesState {
            registry: HostRegistry::default(),
            ..Default::default()
        };
        assert_eq!(row_ids(&empty), vec![MachineRowId::Local]);
    }

    // ── detail-line formatting ──────────────────────────────────

    #[test]
    fn host_detail_line_formats_gpu_vram_backend_hostport() {
        let s = status(Some(mold_core::GpuInfo {
            name: "NVIDIA GeForce RTX 4090".into(),
            vram_total_mb: 24564,
            vram_used_mb: 8192,
            backend: Some(mold_core::GpuBackend::Cuda),
        }));
        assert_eq!(
            host_detail_line("http://hal9000:7680", Some(&s)),
            "NVIDIA GeForce RTX 4090 · 24 GB · CUDA · hal9000:7680"
        );
    }

    #[test]
    fn host_detail_line_infers_backend_for_older_servers() {
        // No additive `backend` field — GPU-name inference (desktop rule).
        let s = status(Some(mold_core::GpuInfo {
            name: "Apple M3 Max".into(),
            vram_total_mb: 49152,
            vram_used_mb: 0,
            backend: None,
        }));
        let line = host_detail_line("http://bender:7680", Some(&s));
        assert!(line.contains("Metal"), "{line}");
        assert!(line.ends_with("bender:7680"), "{line}");
    }

    #[test]
    fn host_detail_line_summarizes_every_gpu_worker() {
        let mut s = status(Some(mold_core::GpuInfo {
            name: "NVIDIA RTX 3090".into(),
            vram_total_mb: 24576,
            vram_used_mb: 8192,
            backend: Some(mold_core::GpuBackend::Cuda),
        }));
        s.gpus = Some(vec![
            mold_core::GpuWorkerStatus {
                ordinal: 0,
                name: "NVIDIA RTX 3090".into(),
                vram_total_bytes: 24 * 1024_u64.pow(3),
                vram_used_bytes: 8 * 1024_u64.pow(3),
                loaded_model: Some("flux-dev:q4".into()),
                state: mold_core::GpuWorkerState::Generating,
            },
            mold_core::GpuWorkerStatus {
                ordinal: 1,
                name: "NVIDIA RTX 3090".into(),
                vram_total_bytes: 24 * 1024_u64.pow(3),
                vram_used_bytes: 4 * 1024_u64.pow(3),
                loaded_model: None,
                state: mold_core::GpuWorkerState::Idle,
            },
        ]);

        assert_eq!(
            host_detail_line("http://hal9000:7680", Some(&s)),
            "2× NVIDIA RTX 3090 · 48 GB · CUDA · hal9000:7680"
        );
    }

    #[test]
    fn host_detail_line_without_status_is_hostport_only() {
        assert_eq!(
            host_detail_line("http://hal9000:7680", None),
            "hal9000:7680"
        );
    }

    #[test]
    fn local_detail_line_joins_memory_backend_server() {
        let line = local_detail_line(Some("VRAM: 32.0 GB free"), "in-process");
        assert!(line.starts_with("VRAM: 32.0 GB free · "));
        assert!(line.ends_with(" · in-process"));
        // Without a memory reading the line still names backend + server.
        let bare = local_detail_line(None, "localhost:7680");
        assert!(bare.ends_with(" · localhost:7680"));
    }

    #[test]
    fn local_server_label_prefers_loopback_and_spawned_servers() {
        assert_eq!(
            local_server_label(Some("http://localhost:7680"), false),
            "localhost:7680"
        );
        assert_eq!(
            local_server_label(Some("http://127.0.0.1:7680"), false),
            "127.0.0.1:7680"
        );
        // A remote MOLD_HOST is not this machine's server.
        assert_eq!(
            local_server_label(Some("http://hal9000:7680"), false),
            "in-process"
        );
        // …unless the TUI spawned it (spawned servers are always local).
        assert_eq!(
            local_server_label(Some("http://localhost:9999"), true),
            "localhost:9999"
        );
        assert_eq!(local_server_label(None, false), "in-process");
    }

    // ── lane labels ─────────────────────────────────────────────

    #[test]
    fn running_lane_includes_gpu_ordinal_when_known() {
        assert_eq!(
            running_lane("flux2-klein:q8", "12s", Some(0)),
            "▶ flux2-klein:q8 · 12s · gpu0"
        );
        assert_eq!(
            running_lane("flux2-klein:q8", "12s", None),
            "▶ flux2-klein:q8 · 12s"
        );
    }

    #[test]
    fn queued_lane_positions_are_one_based() {
        assert_eq!(queued_lane("sdxl:fp16", 0), "● sdxl:fp16 · #1");
        assert_eq!(queued_lane("sdxl:fp16", 2), "● sdxl:fp16 · #3");
    }

    #[test]
    fn queue_elapsed_label_scales_units() {
        assert_eq!(queue_elapsed_label(0, 12_000), "12s");
        assert_eq!(queue_elapsed_label(0, 90_000), "1m");
        assert_eq!(queue_elapsed_label(0, 7_200_000), "2h");
        // Clock skew (start in the future) must not underflow.
        assert_eq!(queue_elapsed_label(10_000, 5_000), "0s");
    }

    #[test]
    fn vram_and_disk_labels() {
        assert_eq!(vram_label(8192, 24564), "8.0 / 24.0 GB");
        assert_eq!(
            disk_label(213_909_504_000, 994_662_584_320),
            "199.2 GB free of 926.4 GB"
        );
    }

    #[test]
    fn format_uptime_scales_units() {
        assert_eq!(format_uptime(42), "42s");
        assert_eq!(format_uptime(720), "12m");
        assert_eq!(format_uptime(12_240), "3h 24m");
        assert_eq!(format_uptime(183_600), "2d 3h");
    }

    // ── helpers moved from ui/queue.rs (tests preserved) ────────

    #[test]
    fn prompt_preview_keeps_short_prompts_intact() {
        assert_eq!(prompt_preview("a cat"), "a cat");
    }

    #[test]
    fn prompt_preview_truncates_long_prompts_with_ellipsis() {
        let long = "x".repeat(200);
        let result = prompt_preview(&long);
        assert!(result.ends_with('…'));
        assert!(result.chars().count() <= 80);
    }

    #[test]
    fn prompt_preview_empty_renders_placeholder() {
        assert_eq!(prompt_preview(""), "(empty)");
        assert_eq!(prompt_preview("   "), "(empty)");
    }

    #[test]
    fn relative_time_handles_zero_as_unknown() {
        assert_eq!(relative_time(0), "—");
    }

    #[test]
    fn running_time_label_uses_fraction_when_total_known() {
        assert_eq!(running_time_label(3, 10), "3/10");
        assert_eq!(running_time_label(0, 0), "…");
    }
}
