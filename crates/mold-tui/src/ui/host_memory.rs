//! Host-RAM pressure for the terminal surface.
//!
//! The Rust mirror of `studio/lib/hostMemory.ts` — web, desktop, iPhone, and
//! the TUI must colour the same host the same way. The level comes from the
//! scheduler's own headroom measured against its safety floor, never from
//! used/total: bytes committed to a reservation that has not allocated yet
//! still park a queue while the OS keeps reporting them free, and that gap is
//! the whole point of the reading.
//!
//! Absent means unknown. A host that reports no snapshot renders exactly what
//! it rendered before this field existed; the terminal never fills the gap
//! with zeros, which would read as a machine under total pressure.

use ratatui::prelude::*;

use super::theme::Theme;

/// Mirrors the shared `HostMemoryLevel` vocabulary. `None` at the call site
/// means "the host did not say".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostMemoryLevel {
    Ok,
    Warn,
    Critical,
}

impl HostMemoryLevel {
    /// The word a surface prints beside the meter. Pressure is always named
    /// in text — colour alone is not a signal in a terminal that may be
    /// monochrome, and the repo's badge rule says the same.
    pub fn as_str(self) -> &'static str {
        match self {
            HostMemoryLevel::Ok => "ok",
            HostMemoryLevel::Warn => "tight",
            HostMemoryLevel::Critical => "critical",
        }
    }
}

/// Pressure level from the ledger's own headroom. Thresholds are pinned to
/// `hostMemoryLevel` in `studio/lib/hostMemory.ts`:
///
/// - no headroom left → `Critical` (nothing more can be admitted)
/// - less than one safety floor of headroom → `Warn`
/// - otherwise → `Ok`
pub fn host_memory_level(
    snapshot: Option<&mold_core::HostMemorySnapshot>,
) -> Option<HostMemoryLevel> {
    let snapshot = snapshot?;
    if snapshot.headroom_bytes == 0 {
        return Some(HostMemoryLevel::Critical);
    }
    if snapshot.safety_floor_bytes > 0 && snapshot.headroom_bytes < snapshot.safety_floor_bytes {
        return Some(HostMemoryLevel::Warn);
    }
    Some(HostMemoryLevel::Ok)
}

/// `"45.4 GB"` — GiB-scaled, matching the VRAM and disk labels beside it.
pub(crate) fn format_gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
}

/// Detail-pane value: what the scheduler can still spend, out of what the
/// machine has, with the pressure named whenever it is not `ok`.
///
/// A ZFS host's headroom already counts its evictable ARC (#1439); the value
/// says so whenever the credit is positive, so the number a user reads names
/// what it includes.
pub fn host_memory_detail_value(snapshot: &mold_core::HostMemorySnapshot) -> String {
    let arc = match snapshot.reclaimable_zfs_arc_bytes {
        Some(credit) if credit > 0 => {
            format!(" (incl. {} evictable ZFS ARC)", format_gb(credit))
        }
        _ => String::new(),
    };
    let base = format!(
        "{} schedulable of {}{arc}",
        format_gb(snapshot.headroom_bytes),
        format_gb(snapshot.total_bytes)
    );
    match host_memory_level(Some(snapshot)) {
        Some(HostMemoryLevel::Ok) | None => base,
        Some(level) => format!("{base} · {}", level.as_str()),
    }
}

/// Style for a host-RAM reading — the same warning/error vocabulary the
/// device rows already use, dim when the host is comfortable.
pub fn host_memory_style(level: Option<HostMemoryLevel>, theme: &Theme) -> Style {
    match level {
        Some(HostMemoryLevel::Critical) => theme.error(),
        Some(HostMemoryLevel::Warn) => theme.warning(),
        _ => theme.param_value(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOOR: u64 = 10 * 1024_u64.pow(3);

    fn snapshot(headroom_bytes: u64, safety_floor_bytes: u64) -> mold_core::HostMemorySnapshot {
        mold_core::HostMemorySnapshot {
            total_bytes: 64 * 1024_u64.pow(3),
            available_bytes: 48 * 1024_u64.pow(3),
            headroom_bytes,
            safety_floor_bytes,
            reclaimable_zfs_arc_bytes: None,
        }
    }

    /// Pinned to `hostMemoryLevel` in `studio/lib/hostMemory.ts`. A terminal
    /// that draws its own thresholds tells a different story about the same
    /// host than the browser sitting next to it.
    #[test]
    fn level_mirrors_the_shared_studio_thresholds() {
        assert_eq!(
            host_memory_level(Some(&snapshot(0, FLOOR))),
            Some(HostMemoryLevel::Critical),
            "no headroom means nothing more can be admitted"
        );
        assert_eq!(
            host_memory_level(Some(&snapshot(FLOOR - 1, FLOOR))),
            Some(HostMemoryLevel::Warn)
        );
        assert_eq!(
            host_memory_level(Some(&snapshot(FLOOR, FLOOR))),
            Some(HostMemoryLevel::Ok),
            "exactly one floor of headroom is not yet pressure"
        );
        assert_eq!(
            host_memory_level(Some(&snapshot(40 * 1024_u64.pow(3), FLOOR))),
            Some(HostMemoryLevel::Ok)
        );
        // A host with no floor still reports critical at zero headroom, and
        // never warns — there is no reserve to compare against.
        assert_eq!(
            host_memory_level(Some(&snapshot(1, 0))),
            Some(HostMemoryLevel::Ok)
        );
        assert_eq!(
            host_memory_level(Some(&snapshot(0, 0))),
            Some(HostMemoryLevel::Critical)
        );
    }

    #[test]
    fn absent_snapshot_has_no_level() {
        assert_eq!(host_memory_level(None), None);
    }

    #[test]
    fn detail_value_names_pressure_in_words() {
        assert_eq!(
            host_memory_detail_value(&snapshot(40 * 1024_u64.pow(3), FLOOR)),
            "40.0 GB schedulable of 64.0 GB",
            "a comfortable host reads as plain telemetry"
        );
        assert_eq!(
            host_memory_detail_value(&snapshot(3 * 1024_u64.pow(3), FLOOR)),
            "3.0 GB schedulable of 64.0 GB · tight"
        );
        assert_eq!(
            host_memory_detail_value(&snapshot(0, FLOOR)),
            "0.0 GB schedulable of 64.0 GB · critical",
            "pressure is always spelled out; colour alone is not a signal"
        );
    }

    /// Mirrors `hostMemoryScheduleLabel` in `studio/lib/hostMemory.ts`: the
    /// credit is named only when positive, and never changes the level.
    #[test]
    fn detail_value_names_the_evictable_arc_when_present() {
        let zfs = mold_core::HostMemorySnapshot {
            reclaimable_zfs_arc_bytes: Some(14 * 1024_u64.pow(3)),
            ..snapshot(40 * 1024_u64.pow(3), FLOOR)
        };
        assert_eq!(
            host_memory_detail_value(&zfs),
            "40.0 GB schedulable of 64.0 GB (incl. 14.0 GB evictable ZFS ARC)"
        );
        let tight = mold_core::HostMemorySnapshot {
            reclaimable_zfs_arc_bytes: Some(2 * 1024_u64.pow(3)),
            ..snapshot(3 * 1024_u64.pow(3), FLOOR)
        };
        assert_eq!(
            host_memory_detail_value(&tight),
            "3.0 GB schedulable of 64.0 GB (incl. 2.0 GB evictable ZFS ARC) · tight"
        );
        let cold = mold_core::HostMemorySnapshot {
            reclaimable_zfs_arc_bytes: Some(0),
            ..snapshot(40 * 1024_u64.pow(3), FLOOR)
        };
        assert_eq!(
            host_memory_detail_value(&cold),
            "40.0 GB schedulable of 64.0 GB",
            "a cold ARC on a ZFS host reads like any other host"
        );
    }

    #[test]
    fn style_reuses_the_device_row_vocabulary() {
        let theme = Theme::default();
        assert_eq!(
            host_memory_style(Some(HostMemoryLevel::Critical), &theme),
            theme.error()
        );
        assert_eq!(
            host_memory_style(Some(HostMemoryLevel::Warn), &theme),
            theme.warning()
        );
        assert_eq!(
            host_memory_style(Some(HostMemoryLevel::Ok), &theme),
            theme.param_value()
        );
        assert_eq!(host_memory_style(None, &theme), theme.param_value());
    }

    #[test]
    fn format_gb_matches_the_neighbouring_vram_and_disk_labels() {
        assert_eq!(format_gb(0), "0.0 GB");
        assert_eq!(format_gb(64 * 1024_u64.pow(3)), "64.0 GB");
    }
}
