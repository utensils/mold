//! How a durable duration parent reads as an ordinary Queue row.
//!
//! One wording on every surface. `docs/contracts/video-job-rows.json` is the
//! cross-language fixture; `studio/lib/videoJobRow.ts` and MoldClient's
//! `VideoJobRow.swift` are pinned to the same file.

use crate::video_job::{VideoJob, VideoJobProgressPhase, VideoJobState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoJobRowSymbol {
    Waiting,
    Rendering,
    Joining,
    Stopping,
    Paused,
    Interrupted,
    Failed,
    Stopped,
    Done,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VideoJobRow {
    pub caption: String,
    pub detail: String,
    pub symbol: VideoJobRowSymbol,
    /// Completed fraction for a determinate meter; only multi-clip work in
    /// progress has one.
    pub progress: Option<f64>,
}

/// The facts a row is drawn from. `current` is false when the client can only
/// show its last cached snapshot (its owner check refused live controls).
#[derive(Debug, Clone, Copy)]
pub struct VideoJobRowInput {
    pub state: VideoJobState,
    pub progress_phase: Option<VideoJobProgressPhase>,
    pub completed_clips: u64,
    pub clip_count: u64,
    pub total_duration_ms: u64,
    pub can_cancel: bool,
    pub current: bool,
}

impl VideoJobRowInput {
    pub fn from_job(job: &VideoJob, current: bool) -> Self {
        Self {
            state: job.state,
            progress_phase: job.progress_phase,
            completed_clips: job.completed_clips,
            clip_count: job.plan.clip_count,
            total_duration_ms: job.plan.total_duration_ms,
            can_cancel: job.can_cancel,
            current,
        }
    }
}

pub fn video_job_row(input: VideoJobRowInput) -> VideoJobRow {
    use VideoJobRowSymbol as S;
    let clips = input.clip_count.max(1);
    let done = input.completed_clips.min(clips);
    let multi = clips > 1;
    let tally = |verb: &str| {
        if multi {
            format!("{verb} · {done} of {clips} clips done")
        } else {
            verb.to_owned()
        }
    };
    let meter = multi.then(|| done as f64 / clips as f64);
    let (caption, symbol, progress) = match input.state {
        VideoJobState::Queued => ("Waiting".to_owned(), S::Waiting, None),
        VideoJobState::Running if !input.can_cancel => ("Stopping".to_owned(), S::Stopping, None),
        VideoJobState::Running => match input.progress_phase {
            Some(VideoJobProgressPhase::Waiting) if done == 0 => {
                ("Waiting".to_owned(), S::Waiting, None)
            }
            Some(VideoJobProgressPhase::Joining) => (
                if multi { "Joining clips" } else { "Finishing" }.to_owned(),
                S::Joining,
                meter,
            ),
            // Rendering, a waiting later clip, or an older server's bare
            // `running` all read as the movie being rendered.
            _ if multi => (
                format!("Rendering clip {} of {clips}", (done + 1).min(clips)),
                S::Rendering,
                meter,
            ),
            _ => ("Rendering".to_owned(), S::Rendering, None),
        },
        VideoJobState::Paused => (tally("Paused"), S::Paused, None),
        VideoJobState::Interrupted => (tally("Interrupted"), S::Interrupted, None),
        VideoJobState::Failed => ("Failed".to_owned(), S::Failed, None),
        VideoJobState::Cancelled => ("Stopped".to_owned(), S::Stopped, None),
        VideoJobState::Completed => ("Done".to_owned(), S::Done, None),
    };
    let caption = if input.current {
        caption
    } else {
        format!("Last known: {caption}")
    };
    let detail = if multi {
        format!("{} video · {clips} clips", seconds(input.total_duration_ms))
    } else {
        format!("{} video", seconds(input.total_duration_ms))
    };
    VideoJobRow {
        caption,
        detail,
        symbol,
        progress,
    }
}

/// `14000` → `14 s`, `14375` → `14.4 s` (one decimal, trailing zero dropped).
fn seconds(milliseconds: u64) -> String {
    let tenths = (milliseconds + 50) / 100;
    if tenths.is_multiple_of(10) {
        format!("{} s", tenths / 10)
    } else {
        format!("{}.{} s", tenths / 10, tenths % 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Input {
        state: VideoJobState,
        #[serde(default)]
        progress_phase: Option<VideoJobProgressPhase>,
        completed_clips: String,
        clip_count: String,
        total_duration_ms: String,
        can_cancel: bool,
        current: bool,
    }

    #[derive(serde::Deserialize)]
    struct Case {
        input: Input,
        row: VideoJobRow,
    }

    #[test]
    fn cross_surface_contract() {
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("../../../docs/contracts/video-job-rows.json"))
                .unwrap();
        assert!(cases.len() >= 10);
        for case in cases {
            let input = VideoJobRowInput {
                state: case.input.state,
                progress_phase: case.input.progress_phase,
                completed_clips: case.input.completed_clips.parse().unwrap(),
                clip_count: case.input.clip_count.parse().unwrap(),
                total_duration_ms: case.input.total_duration_ms.parse().unwrap(),
                can_cancel: case.input.can_cancel,
                current: case.input.current,
            };
            assert_eq!(video_job_row(input), case.row);
        }
    }

    #[test]
    fn never_says_one_clips() {
        for state in [
            VideoJobState::Running,
            VideoJobState::Paused,
            VideoJobState::Queued,
        ] {
            let row = video_job_row(VideoJobRowInput {
                state,
                progress_phase: None,
                completed_clips: 0,
                clip_count: 1,
                total_duration_ms: 5_000,
                can_cancel: true,
                current: true,
            });
            assert!(
                !row.caption.contains("clips") && !row.detail.contains("clips"),
                "{row:?}"
            );
        }
    }
}
