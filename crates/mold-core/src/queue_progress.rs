//! One progress snapshot per queued generation, and the two directions it
//! travels.
//!
//! A durable child publishes no progress frames of its own: its authority is
//! the batch child state, and the only live signal a client can read is
//! `GET /api/queue/{id}/preview`. That endpoint used to carry a denoise
//! preview alone, so a host running with `MOLD_STEP_PREVIEW=0` reported no
//! step counter at all and every non-browser surface — CLI `--batch N`, the
//! TUI batch pane, MCP async jobs, RunPod — went silent for the whole render.
//!
//! [`QueueJobProgress`] is the fix, and it is one mechanism rather than a
//! second emitter: the server FOLDS every [`SseProgressEvent`] the existing
//! `progress_tx` fan-out already carries into this snapshot ([`apply`]), and
//! a polling client UNFOLDS the difference between two snapshots back into
//! the same events ([`events_since`]) so every surface renders progress with
//! exactly the code the attached singleton path already has.
//!
//! [`apply`]: QueueJobProgress::apply
//! [`events_since`]: QueueJobProgress::events_since

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::SseProgressEvent;

/// Weight-loading progress for the component a job is currently reading.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct QueueJobWeightLoad {
    pub bytes_loaded: u64,
    pub bytes_total: u64,
    pub component: String,
}

/// Download progress for the file a job's model pull is currently fetching.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct QueueJobDownload {
    pub filename: String,
    pub file_index: usize,
    pub total_files: usize,
    pub bytes_downloaded: u64,
    pub bytes_total: u64,
}

/// Everything a live queue row knows about its own progress right now.
///
/// This is STATE, not a message log: a client polls it, so two snapshots are
/// all any consumer ever sees and anything that happened strictly between
/// them is gone. Each field therefore holds the latest value of something
/// that persists, and the one-shot message variants (`Info`, `CacheHit`,
/// `DependencyWait`) land on [`stage`](Self::stage), which is the
/// human-facing "what is happening now" line every surface already renders.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct QueueJobProgress {
    /// Latest denoise step, once one has been reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<usize>,
    /// Total denoise steps for this render.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<usize>,
    /// Latest activity line: a stage name, or the message from a dependency
    /// wait, cache hit, or info event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    /// Completed work inside the current named stage. This stays separate
    /// from denoise `step` because setup, paint, and export stages also have
    /// bounded progress and must round-trip through durable polling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_current: Option<usize>,
    /// Total work inside the current named stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_total: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_load: Option<QueueJobWeightLoad>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub download: Option<QueueJobDownload>,
    /// Position in line as the server last reported it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_position: Option<usize>,
    /// Base64 PNG of the latest denoise preview. Absent for the whole render
    /// on a host with `MOLD_STEP_PREVIEW=0` — which is exactly why the step
    /// counter above is a separate field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview_image: Option<String>,
    /// Wall clock of the last folded event, so a consumer can price a
    /// per-step interval without the server keeping a timer per job.
    pub updated_at_ms: u64,
}

impl QueueJobProgress {
    /// Fold one progress event into the snapshot.
    ///
    /// Called from the single `progress_tx` fan-out the attached observer
    /// already reads — there is deliberately no second emitter.
    pub fn apply(&mut self, event: &SseProgressEvent, now_ms: u64) {
        match event {
            SseProgressEvent::DenoiseStep { step, total, .. } => {
                self.step = Some(*step);
                self.total = Some(*total);
            }
            SseProgressEvent::Preview { image, step, total } => {
                self.preview_image = Some(image.clone());
                self.step = Some(*step);
                self.total = Some(*total);
            }
            SseProgressEvent::StageStart { name } => {
                self.stage = Some(name.clone());
                self.step = None;
                self.total = None;
                self.stage_current = None;
                self.stage_total = None;
            }
            SseProgressEvent::StageProgress {
                name,
                current,
                total,
            } => {
                self.stage = Some(name.clone());
                self.stage_current = Some(*current);
                self.stage_total = Some(*total);
            }
            SseProgressEvent::StageDone { name, .. } => {
                if self.stage.as_deref() == Some(name.as_str()) {
                    self.stage = None;
                    self.stage_current = None;
                    self.stage_total = None;
                }
            }
            SseProgressEvent::DependencyWait { dependency, reason } => {
                self.stage = Some(format!("waiting for {dependency}: {reason}"));
            }
            SseProgressEvent::CacheHit { resource } => {
                self.stage = Some(format!("cache hit: {resource}"));
            }
            SseProgressEvent::Info { message } => self.stage = Some(message.clone()),
            SseProgressEvent::WeightLoad {
                bytes_loaded,
                bytes_total,
                component,
            } => {
                self.weight_load = Some(QueueJobWeightLoad {
                    bytes_loaded: *bytes_loaded,
                    bytes_total: *bytes_total,
                    component: component.clone(),
                });
            }
            SseProgressEvent::DownloadProgress {
                filename,
                file_index,
                total_files,
                bytes_downloaded,
                bytes_total,
                ..
            } => {
                self.download = Some(QueueJobDownload {
                    filename: filename.clone(),
                    file_index: *file_index,
                    total_files: *total_files,
                    bytes_downloaded: *bytes_downloaded,
                    bytes_total: *bytes_total,
                });
            }
            SseProgressEvent::DownloadDone {
                filename,
                file_index,
                total_files,
                ..
            } => {
                let bytes_total = self
                    .download
                    .as_ref()
                    .filter(|current| current.file_index == *file_index)
                    .map(|current| current.bytes_total)
                    .unwrap_or_default();
                self.download = Some(QueueJobDownload {
                    filename: filename.clone(),
                    file_index: *file_index,
                    total_files: *total_files,
                    bytes_downloaded: bytes_total,
                    bytes_total,
                });
            }
            SseProgressEvent::PullComplete { .. } => self.download = None,
            SseProgressEvent::Queued { position, .. } => self.queue_position = Some(*position),
        }
        self.updated_at_ms = now_ms;
    }

    /// Unfold the difference against the previously observed snapshot back
    /// into the events a surface already knows how to render.
    ///
    /// Only what CHANGED is reported, so a client that polls twice as fast as
    /// the server renders does not redraw an unchanged bar, and a consumer
    /// that has never seen this job gets everything the snapshot holds.
    pub fn events_since(&self, previous: Option<&Self>) -> Vec<SseProgressEvent> {
        let mut events = Vec::new();
        if self.queue_position != previous.and_then(|p| p.queue_position) {
            if let Some(position) = self.queue_position {
                events.push(SseProgressEvent::Queued {
                    position,
                    id: String::new(),
                });
            }
        }
        if self.stage != previous.and_then(|p| p.stage.clone()) {
            if let Some(name) = self.stage.clone() {
                events.push(SseProgressEvent::StageStart { name });
            }
        }
        let stage_progress_changed = self.stage_current != previous.and_then(|p| p.stage_current)
            || self.stage_total != previous.and_then(|p| p.stage_total);
        if stage_progress_changed {
            if let (Some(name), Some(current), Some(total)) =
                (self.stage.clone(), self.stage_current, self.stage_total)
            {
                events.push(SseProgressEvent::StageProgress {
                    name,
                    current,
                    total,
                });
            }
        }
        if self.weight_load != previous.and_then(|p| p.weight_load.clone()) {
            if let Some(weights) = self.weight_load.clone() {
                events.push(SseProgressEvent::WeightLoad {
                    bytes_loaded: weights.bytes_loaded,
                    bytes_total: weights.bytes_total,
                    component: weights.component,
                });
            }
        }
        if self.download != previous.and_then(|p| p.download.clone()) {
            if let Some(download) = self.download.clone() {
                events.push(SseProgressEvent::DownloadProgress {
                    filename: download.filename,
                    file_index: download.file_index,
                    total_files: download.total_files,
                    bytes_downloaded: download.bytes_downloaded,
                    bytes_total: download.bytes_total,
                    batch_bytes_downloaded: download.bytes_downloaded,
                    batch_bytes_total: download.bytes_total,
                    batch_elapsed_ms: 0,
                });
            }
        }
        let step_changed = self.step != previous.and_then(|p| p.step);
        if let (Some(step), Some(total)) = (self.step, self.total) {
            if step_changed {
                events.push(SseProgressEvent::DenoiseStep {
                    step,
                    total,
                    elapsed_ms: self.step_interval_ms(previous),
                });
            }
        }
        let preview_changed = self.preview_image != previous.and_then(|p| p.preview_image.clone());
        if preview_changed {
            if let (Some(image), Some(step), Some(total)) =
                (self.preview_image.clone(), self.step, self.total)
            {
                events.push(SseProgressEvent::Preview { image, step, total });
            }
        }
        events
    }

    /// Wall-clock milliseconds per denoise step between two snapshots, which
    /// is what a rendered `it/s` figure is priced from. `0` whenever the
    /// interval cannot be derived — the renderers already treat that as
    /// "unknown" rather than dividing by it.
    fn step_interval_ms(&self, previous: Option<&Self>) -> u64 {
        let Some(previous) = previous else {
            return 0;
        };
        let steps = self
            .step
            .unwrap_or_default()
            .saturating_sub(previous.step.unwrap_or_default());
        let elapsed = self.updated_at_ms.saturating_sub(previous.updated_at_ms);
        if steps == 0 {
            return 0;
        }
        elapsed / steps as u64
    }
}

/// The client side of the fold: remembers the last snapshot seen per durable
/// job so every polling surface derives the same events from the same rule
/// instead of keeping its own diff.
#[derive(Debug, Default)]
pub struct ProgressEventStream {
    last: HashMap<String, QueueJobProgress>,
}

impl ProgressEventStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// Events this snapshot adds for `job_id`, and remember it as the new
    /// baseline.
    pub fn events(&mut self, job_id: &str, progress: &QueueJobProgress) -> Vec<SseProgressEvent> {
        let events = progress.events_since(self.last.get(job_id));
        self.last.insert(job_id.to_string(), progress.clone());
        events
    }

    /// Forget a job that has settled, so a replayed id never diffs against a
    /// previous render's snapshot.
    pub fn forget(&mut self, job_id: &str) {
        self.last.remove(job_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folded(events: &[SseProgressEvent]) -> QueueJobProgress {
        let mut progress = QueueJobProgress::default();
        for (index, event) in events.iter().enumerate() {
            progress.apply(event, 1_000 + index as u64);
        }
        progress
    }

    #[test]
    fn a_denoise_step_carries_the_counter_without_any_preview() {
        let progress = folded(&[SseProgressEvent::DenoiseStep {
            step: 7,
            total: 20,
            elapsed_ms: 120,
        }]);
        assert_eq!(progress.step, Some(7));
        assert_eq!(progress.total, Some(20));
        assert_eq!(progress.preview_image, None);
        assert_eq!(progress.updated_at_ms, 1_000);
    }

    #[test]
    fn a_named_stage_keeps_and_unfolds_its_own_counter() {
        let start = folded(&[SseProgressEvent::StageStart {
            name: "Generating PBR views".into(),
        }]);
        let current = folded(&[
            SseProgressEvent::StageStart {
                name: "Generating PBR views".into(),
            },
            SseProgressEvent::StageProgress {
                name: "Generating PBR views".into(),
                current: 7,
                total: 15,
            },
        ]);

        assert_eq!(current.stage_current, Some(7));
        assert_eq!(current.stage_total, Some(15));
        assert!(matches!(
            current.events_since(Some(&start)).as_slice(),
            [SseProgressEvent::StageProgress {
                name,
                current: 7,
                total: 15,
            }] if name == "Generating PBR views"
        ));
    }

    #[test]
    fn every_progress_variant_folds_into_the_snapshot() {
        let progress = folded(&[
            SseProgressEvent::Queued {
                position: 2,
                id: "job".into(),
            },
            SseProgressEvent::DependencyWait {
                dependency: "flux-dev:q8".into(),
                reason: "pulling".into(),
            },
            SseProgressEvent::DownloadProgress {
                filename: "t5.safetensors".into(),
                file_index: 1,
                total_files: 3,
                bytes_downloaded: 10,
                bytes_total: 100,
                batch_bytes_downloaded: 10,
                batch_bytes_total: 300,
                batch_elapsed_ms: 5,
            },
            SseProgressEvent::DownloadDone {
                filename: "t5.safetensors".into(),
                file_index: 1,
                total_files: 3,
                batch_bytes_downloaded: 100,
                batch_bytes_total: 300,
                batch_elapsed_ms: 9,
            },
            SseProgressEvent::CacheHit {
                resource: "tokenizer".into(),
            },
            SseProgressEvent::WeightLoad {
                bytes_loaded: 5,
                bytes_total: 50,
                component: "transformer".into(),
            },
            SseProgressEvent::StageStart {
                name: "Denoising".into(),
            },
            SseProgressEvent::StageProgress {
                name: "Denoising".into(),
                current: 1,
                total: 4,
            },
            SseProgressEvent::Preview {
                image: "UFJFVklFVw==".into(),
                step: 3,
                total: 8,
            },
            SseProgressEvent::Info {
                message: "auto pipeline".into(),
            },
        ]);
        assert_eq!(progress.queue_position, Some(2));
        assert_eq!(progress.stage.as_deref(), Some("auto pipeline"));
        assert_eq!(
            progress.weight_load,
            Some(QueueJobWeightLoad {
                bytes_loaded: 5,
                bytes_total: 50,
                component: "transformer".into(),
            })
        );
        assert_eq!(
            progress.download,
            Some(QueueJobDownload {
                filename: "t5.safetensors".into(),
                file_index: 1,
                total_files: 3,
                bytes_downloaded: 100,
                bytes_total: 100,
            })
        );
        assert_eq!(progress.step, Some(3));
        assert_eq!(progress.total, Some(8));
        assert_eq!(progress.preview_image.as_deref(), Some("UFJFVklFVw=="));
    }

    #[test]
    fn a_finished_stage_clears_only_its_own_name() {
        let mut progress = folded(&[SseProgressEvent::StageStart {
            name: "Loading".into(),
        }]);
        progress.apply(
            &SseProgressEvent::StageDone {
                name: "Encoding".into(),
                elapsed_ms: 4,
            },
            2_000,
        );
        assert_eq!(progress.stage.as_deref(), Some("Loading"));
        progress.apply(
            &SseProgressEvent::StageDone {
                name: "Loading".into(),
                elapsed_ms: 4,
            },
            2_001,
        );
        assert_eq!(progress.stage, None);
    }

    #[test]
    fn a_completed_pull_drops_the_download_row() {
        let progress = folded(&[
            SseProgressEvent::DownloadProgress {
                filename: "a".into(),
                file_index: 0,
                total_files: 1,
                bytes_downloaded: 1,
                bytes_total: 2,
                batch_bytes_downloaded: 1,
                batch_bytes_total: 2,
                batch_elapsed_ms: 0,
            },
            SseProgressEvent::PullComplete {
                model: "flux-dev:q8".into(),
            },
        ]);
        assert_eq!(progress.download, None);
    }

    #[test]
    fn the_first_snapshot_unfolds_into_everything_it_holds() {
        let progress = folded(&[
            SseProgressEvent::Queued {
                position: 1,
                id: "x".into(),
            },
            SseProgressEvent::StageStart {
                name: "Denoising".into(),
            },
            SseProgressEvent::DenoiseStep {
                step: 2,
                total: 8,
                elapsed_ms: 50,
            },
            SseProgressEvent::Preview {
                image: "aW1n".into(),
                step: 2,
                total: 8,
            },
        ]);
        let events = progress.events_since(None);
        assert_eq!(
            events,
            vec![
                SseProgressEvent::Queued {
                    position: 1,
                    id: String::new(),
                },
                SseProgressEvent::StageStart {
                    name: "Denoising".into(),
                },
                SseProgressEvent::DenoiseStep {
                    step: 2,
                    total: 8,
                    elapsed_ms: 0,
                },
                SseProgressEvent::Preview {
                    image: "aW1n".into(),
                    step: 2,
                    total: 8,
                },
            ]
        );
    }

    #[test]
    fn an_unchanged_snapshot_unfolds_into_nothing() {
        let progress = folded(&[SseProgressEvent::DenoiseStep {
            step: 2,
            total: 8,
            elapsed_ms: 50,
        }]);
        assert_eq!(progress.events_since(Some(&progress.clone())), Vec::new());
    }

    #[test]
    fn a_step_advance_prices_its_own_interval() {
        let previous = folded(&[SseProgressEvent::DenoiseStep {
            step: 2,
            total: 8,
            elapsed_ms: 0,
        }]);
        let mut next = previous.clone();
        next.apply(
            &SseProgressEvent::DenoiseStep {
                step: 4,
                total: 8,
                elapsed_ms: 0,
            },
            previous.updated_at_ms + 500,
        );
        assert_eq!(
            next.events_since(Some(&previous)),
            vec![SseProgressEvent::DenoiseStep {
                step: 4,
                total: 8,
                elapsed_ms: 250,
            }]
        );
    }

    #[test]
    fn the_stream_diffs_each_job_against_its_own_baseline() {
        let mut stream = ProgressEventStream::new();
        let first = folded(&[SseProgressEvent::DenoiseStep {
            step: 1,
            total: 4,
            elapsed_ms: 0,
        }]);
        assert_eq!(stream.events("a", &first).len(), 1);
        assert_eq!(stream.events("a", &first), Vec::new());
        // A different job has never been seen, so it reports in full.
        assert_eq!(stream.events("b", &first).len(), 1);
        stream.forget("a");
        assert_eq!(stream.events("a", &first).len(), 1);
    }
}
