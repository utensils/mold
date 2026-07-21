//! Progress formatting + row-model helpers.
//!
//! The Create view's progress *rendering* moved in the redesign: the
//! glyph-styled session log lives in `ui::timeline`, and denoise progress
//! lives in `ui::preview`. This module keeps the shared pieces — the
//! download/weight formatting helpers (also used by the Library upscale
//! overlay) and the [`timeline_rows`] predicate the Timeline uses to
//! decide which pull gauges to pin.

pub(crate) fn format_bytes(bytes: u64) -> String {
    mold_core::format::human_bytes_compact(bytes)
}

pub(crate) fn format_eta(seconds: u64) -> String {
    match seconds {
        0..=59 => format!("{seconds}s"),
        60..=3599 => format!("{}m{:02}s", seconds / 60, seconds % 60),
        _ => format!("{}h{:02}m", seconds / 3600, (seconds % 3600) / 60),
    }
}

/// 2-decimal binary-unit flavor for f64 transfer rates. Presentational
/// one-off — general flavors live in `mold_core::format`.
pub(crate) fn format_bytes_binary(bytes: f64) -> String {
    if bytes >= 1_073_741_824.0 {
        format!("{:.2}GiB", bytes / 1_073_741_824.0)
    } else if bytes >= 1_048_576.0 {
        format!("{:.2}MiB", bytes / 1_048_576.0)
    } else if bytes >= 1024.0 {
        format!("{:.2}KiB", bytes / 1024.0)
    } else {
        format!("{:.0}B", bytes)
    }
}

/// Whether the Timeline panel has anything to draw in its "active bars"
/// region for the given progress snapshot. Exposed as a pure predicate
/// so the "downloading-but-no-bytes-yet" placeholder behaviour is unit
/// testable without spinning up a real frame.
///
/// `has_download` reflects the full gauge (filename/bytes/eta).
/// `has_placeholder` reflects the indeterminate "Preparing download…"
/// row that should appear whenever `downloading` is true but no concrete
/// bytes have arrived yet *and* no spinner stage is set — without it the
/// Timeline stays blank during the `hf-hub` handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TimelineRows {
    pub overall: bool,
    pub spinner: bool,
    pub download: bool,
    pub placeholder: bool,
    pub weight: bool,
    pub denoise: bool,
}

impl TimelineRows {
    pub fn total(self) -> u16 {
        self.overall as u16
            + self.spinner as u16
            + self.download as u16
            + self.placeholder as u16
            + self.weight as u16
            + self.denoise as u16
    }
}

pub(crate) fn timeline_rows(progress: &crate::app::ProgressState) -> TimelineRows {
    let has_denoise = progress.denoise_total > 0 && progress.denoise_step < progress.denoise_total;
    let has_weight = progress.weight_total > 0 && progress.weight_loaded < progress.weight_total;
    let has_download = progress.download_total > 0;
    let has_spinner = progress.current_stage.is_some();
    // Downloading is set on the very first `hf-hub` event, before the
    // file-size resolver has populated `download_total`. Show an
    // indeterminate "preparing" row so the Timeline is never empty
    // while a pull is actually in flight.
    let has_placeholder = progress.is_downloading() && !has_download && !has_spinner;
    // The Overall row is the "you are still generating" heartbeat. It
    // renders whenever a generation is in flight *and* we're not purely
    // downloading (in which case the pull rows already tell the story).
    let has_overall = progress.generation_started_at.is_some() && !progress.is_downloading();
    TimelineRows {
        overall: has_overall,
        spinner: has_spinner,
        download: has_download,
        placeholder: has_placeholder,
        weight: has_weight,
        denoise: has_denoise,
    }
}

/// Format `d` as a compact elapsed timer, e.g. `4.3s`, `1m12s`, `1h02m`.
pub(crate) fn format_elapsed(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        let total_ms = d.as_millis();
        format!("{:.1}s", total_ms as f64 / 1000.0)
    } else if secs < 3600 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::ProgressState;

    #[test]
    fn format_eta_short_values() {
        assert_eq!(format_eta(7), "7s");
        assert_eq!(format_eta(65), "1m05s");
        assert_eq!(format_eta(3665), "1h01m");
    }

    #[test]
    fn format_bytes_binary_uses_cli_style_units() {
        assert_eq!(format_bytes_binary(512.0), "512B");
        assert_eq!(format_bytes_binary(2_048.0), "2.00KiB");
        assert_eq!(format_bytes_binary(3.5 * 1_048_576.0), "3.50MiB");
    }

    #[test]
    fn timeline_shows_placeholder_when_downloading_has_no_bytes_yet() {
        // Codex-adjacent bug: during the hf-hub pre-flight the TUI set
        // `progress.downloading = true`, cleared `current_stage`, and had
        // `download_total = 0`. The Timeline then had nothing to render,
        // leaving the pane blank while the status bar said "Preparing…".
        let mut progress = ProgressState::default();
        progress.downloading = true;
        progress.download_total = 0;
        progress.current_stage = None;

        let rows = timeline_rows(&progress);
        assert!(
            rows.placeholder,
            "expected an indeterminate placeholder row while waiting on hf-hub"
        );
        assert!(!rows.download);
        assert!(!rows.spinner);
        assert_eq!(rows.total(), 1);
    }

    #[test]
    fn timeline_skips_placeholder_once_download_bar_is_live() {
        // Once real byte counts arrive, the full download gauge takes
        // over — the placeholder must not double up with it.
        let mut progress = ProgressState::default();
        progress.downloading = true;
        progress.download_total = 100;
        progress.download_bytes = 10;
        progress.current_stage = None;

        let rows = timeline_rows(&progress);
        assert!(rows.download);
        assert!(!rows.placeholder);
    }

    #[test]
    fn timeline_skips_placeholder_when_spinner_stage_set() {
        // A visible spinner/stage line is already telling the user what's
        // happening, so the placeholder would be redundant.
        let mut progress = ProgressState::default();
        progress.downloading = true;
        progress.current_stage = Some("Verifying weights".into());

        let rows = timeline_rows(&progress);
        assert!(rows.spinner);
        assert!(!rows.placeholder);
    }

    #[test]
    fn timeline_idle_when_not_downloading() {
        let progress = ProgressState::default();
        let rows = timeline_rows(&progress);
        assert_eq!(rows.total(), 0);
    }

    #[test]
    fn timeline_shows_overall_row_while_generating_even_without_gauges() {
        // User-reported: during the model-loading phase of a local run the
        // Timeline went silent between StageStart events — no gauge, no
        // spinner. The Overall row is the heartbeat that's always visible
        // while a generation is in flight, so the user can tell the
        // pipeline is still progressing.
        let mut progress = ProgressState::default();
        progress.mark_generation_start();
        assert!(progress.generation_started_at.is_some());

        let rows = timeline_rows(&progress);
        assert!(
            rows.overall,
            "Overall row must render for the duration of any generation"
        );
    }

    #[test]
    fn timeline_overall_hides_when_only_downloading() {
        // Pure pull (no subsequent generation) already has the download
        // gauge/placeholder telling the story — the Overall heartbeat
        // would just duplicate it.
        let mut progress = ProgressState::default();
        progress.downloading = true;
        progress.download_total = 100;
        progress.download_bytes = 10;
        // No generation started — we're only pulling.
        let rows = timeline_rows(&progress);
        assert!(!rows.overall);
    }

    #[test]
    fn timeline_overall_row_coexists_with_stage_spinner() {
        // During a real generation: Overall heartbeat on top, active
        // spinner row beneath it, plus whatever gauge applies.
        let mut progress = ProgressState::default();
        progress.mark_generation_start();
        progress.current_stage = Some("Loading T5 encoder".into());

        let rows = timeline_rows(&progress);
        assert!(rows.overall);
        assert!(rows.spinner);
        assert_eq!(rows.total(), 2);
    }

    #[test]
    fn format_elapsed_sub_minute_has_decimal() {
        assert_eq!(
            format_elapsed(std::time::Duration::from_millis(250)),
            "0.2s"
        );
        assert_eq!(
            format_elapsed(std::time::Duration::from_millis(4_300)),
            "4.3s"
        );
    }

    #[test]
    fn format_elapsed_rolls_into_minutes_and_hours() {
        assert_eq!(format_elapsed(std::time::Duration::from_secs(75)), "1m15s");
        assert_eq!(
            format_elapsed(std::time::Duration::from_secs(3_725)),
            "1h02m"
        );
    }
}
