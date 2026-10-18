//! Observational no-progress diagnostics. This module never releases a device.

use mold_core::{stall::StallDiagnostic, SseProgressEvent};
use std::time::Instant;

#[derive(Debug, Clone)]
pub(crate) struct ProgressWatchdog {
    last_work: Instant,
    last_work_unix_ms: u64,
    stage: String,
    mark: Option<String>,
}

impl ProgressWatchdog {
    pub(crate) fn new(now: Instant, unix_ms: u64) -> Self {
        Self {
            last_work: now,
            last_work_unix_ms: unix_ms,
            stage: "Starting generation".into(),
            mark: None,
        }
    }

    /// Timers, repeated counters, previews and explanatory text are not work.
    pub(crate) fn observe(&mut self, event: &SseProgressEvent, now: Instant, unix_ms: u64) -> bool {
        if let SseProgressEvent::StageStart { name } = event {
            // Counters belong to a phase, even when the next phase reuses
            // its name. Starting it still does not complete any work.
            self.mark = None;
            self.stage = name.clone();
            return false;
        }
        let (stage, mark) = match event {
            SseProgressEvent::StageStart { .. } => unreachable!(),
            SseProgressEvent::StageDone { name, elapsed_ms } => {
                (name.clone(), format!("done:{name}:{elapsed_ms}"))
            }
            SseProgressEvent::StageProgress { current: 0, .. }
            | SseProgressEvent::DenoiseStep { step: 0, .. }
            | SseProgressEvent::WeightLoad {
                bytes_loaded: 0, ..
            }
            | SseProgressEvent::DownloadProgress {
                bytes_downloaded: 0,
                ..
            } => return false,
            SseProgressEvent::StageProgress {
                name,
                current,
                total,
            } => (name.clone(), format!("work:{name}:{current}:{total}")),
            SseProgressEvent::DenoiseStep { step, total, .. } => {
                ("Denoising".into(), format!("step:{step}:{total}"))
            }
            SseProgressEvent::WeightLoad {
                component,
                bytes_loaded,
                bytes_total,
            } => (
                format!("Loading {component}"),
                format!("load:{component}:{bytes_loaded}:{bytes_total}"),
            ),
            SseProgressEvent::DownloadProgress {
                filename,
                bytes_downloaded,
                bytes_total,
                ..
            } => (
                format!("Downloading {filename}"),
                format!("download:{filename}:{bytes_downloaded}:{bytes_total}"),
            ),
            _ => return false,
        };
        if self.mark.as_ref() == Some(&mark) {
            return false;
        }
        self.mark = Some(mark);
        self.stage = stage;
        self.last_work = now;
        self.last_work_unix_ms = unix_ms;
        true
    }

    pub(crate) fn diagnostic(
        &self,
        now: Instant,
        threshold_seconds: u32,
        cancelled: bool,
    ) -> Option<StallDiagnostic> {
        let elapsed = now.saturating_duration_since(self.last_work);
        if threshold_seconds == 0 || elapsed.as_secs() < u64::from(threshold_seconds) {
            return None;
        }
        Some(StallDiagnostic {
            stage: self.stage.clone(),
            last_completed_work_unix_ms: self.last_work_unix_ms,
            elapsed_stall_ms: elapsed.as_millis().try_into().unwrap_or(u64::MAX),
            cancel_requested: cancelled,
            device_still_held: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::SseProgressEvent;
    use std::time::{Duration, Instant};

    #[test]
    fn reports_stall_without_claiming_cancellation_or_releasing_owner() {
        let start = Instant::now();
        let mut clock = ProgressWatchdog::new(start, 1000);
        clock.observe(
            &SseProgressEvent::StageStart {
                name: "Decoding video frames".into(),
            },
            start,
            1000,
        );
        assert!(clock
            .diagnostic(start + Duration::from_secs(29), 30, false)
            .is_none());
        let stalled = clock
            .diagnostic(start + Duration::from_secs(30), 30, true)
            .unwrap();
        assert_eq!(stalled.stage, "Decoding video frames");
        assert_eq!(stalled.last_completed_work_unix_ms, 1000);
        assert_eq!(stalled.elapsed_stall_ms, 30_000);
        assert!(stalled.cancel_requested);
        assert!(stalled.device_still_held);
        assert!(clock
            .diagnostic(start + Duration::from_secs(300), 0, true)
            .is_none());
    }

    #[test]
    fn duplicate_work_previews_and_info_do_not_refresh_completed_work() {
        let start = Instant::now();
        let mut clock = ProgressWatchdog::new(start, 1000);
        let work = SseProgressEvent::StageProgress {
            name: "Decode block".into(),
            current: 1,
            total: 8,
        };
        clock.observe(&work, start, 1000);
        for event in [
            work.clone(),
            SseProgressEvent::Info {
                message: "Still working".into(),
            },
            SseProgressEvent::Preview {
                image: "preview".into(),
                step: 1,
                total: 8,
            },
        ] {
            clock.observe(&event, start + Duration::from_secs(29), 30_000);
        }
        assert!(clock
            .diagnostic(start + Duration::from_secs(30), 30, false)
            .is_some());
        clock.observe(
            &SseProgressEvent::StageProgress {
                name: "Decode block".into(),
                current: 2,
                total: 8,
            },
            start + Duration::from_secs(31),
            32_000,
        );
        assert!(clock
            .diagnostic(start + Duration::from_secs(32), 30, false)
            .is_none());
    }

    #[test]
    fn stage_text_does_not_claim_a_completed_work_boundary() {
        let start = Instant::now();
        let mut clock = ProgressWatchdog::new(start, 1000);
        clock.observe(
            &SseProgressEvent::StageStart {
                name: "Attention kernel".into(),
            },
            start + Duration::from_secs(29),
            30_000,
        );
        let stall = clock
            .diagnostic(start + Duration::from_secs(30), 30, false)
            .unwrap();
        assert_eq!(stall.stage, "Attention kernel");
        assert_eq!(stall.last_completed_work_unix_ms, 1000);
    }

    #[test]
    fn elapsed_uses_monotonic_time_even_if_wall_clock_jumps() {
        let start = Instant::now();
        let mut clock = ProgressWatchdog::new(start, 1000);
        clock.observe(
            &SseProgressEvent::Info {
                message: "clock jumped".into(),
            },
            start,
            u64::MAX,
        );
        assert!(clock
            .diagnostic(start + Duration::from_secs(1), 30, false)
            .is_none());
    }
    #[test]
    fn repeated_stage_name_gets_a_new_counter_scope_without_claiming_started_work() {
        let start = Instant::now();
        let mut clock = ProgressWatchdog::new(start, 1000);
        let work = SseProgressEvent::StageProgress {
            name: "Decode chunk".into(),
            current: 1,
            total: 1,
        };
        assert!(clock.observe(&work, start, 1000));
        assert!(!clock.observe(
            &SseProgressEvent::StageStart {
                name: "Decode chunk".into()
            },
            start + Duration::from_secs(29),
            30000
        ));
        assert!(clock
            .diagnostic(start + Duration::from_secs(30), 30, false)
            .is_some());
        assert!(clock.observe(&work, start + Duration::from_secs(31), 32000));
        assert!(clock
            .diagnostic(start + Duration::from_secs(32), 30, false)
            .is_none());
    }

    #[test]
    fn zero_byte_reports_do_not_claim_completed_loading_or_download_work() {
        let start = Instant::now();
        let mut clock = ProgressWatchdog::new(start, 1000);
        for event in [
            SseProgressEvent::WeightLoad {
                component: "VAE".into(),
                bytes_loaded: 0,
                bytes_total: 100,
            },
            SseProgressEvent::DownloadProgress {
                filename: "weights".into(),
                file_index: 0,
                total_files: 1,
                bytes_downloaded: 0,
                bytes_total: 100,
                batch_bytes_downloaded: 0,
                batch_bytes_total: 100,
                batch_elapsed_ms: 0,
            },
        ] {
            assert!(!clock.observe(&event, start + Duration::from_secs(29), 30000));
        }
        assert!(clock
            .diagnostic(start + Duration::from_secs(30), 30, false)
            .is_some());
    }
}
