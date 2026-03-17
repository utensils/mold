use std::time::Duration;

/// Progress events emitted during model loading and inference.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// A named stage has started (e.g. "Loading T5 encoder (CPU)")
    StageStart { name: String },
    /// The most recent stage completed, with its elapsed time
    StageDone { name: String, elapsed: Duration },
    /// Informational message (e.g. "CUDA detected, using GPU")
    Info { message: String },
    /// A single denoising step completed.
    DenoiseStep {
        step: usize,
        total: usize,
        elapsed: Duration,
    },
}

/// Callback type for receiving progress events.
pub type ProgressCallback = Box<dyn Fn(ProgressEvent) + Send + Sync>;

/// Wrapper around an optional progress callback with convenience methods.
///
/// Stored as a field in each engine so progress reporting can be borrowed
/// independently from the engine's mutable model state.
#[derive(Default)]
pub struct ProgressReporter {
    callback: Option<ProgressCallback>,
}

impl ProgressReporter {
    pub fn emit(&self, event: ProgressEvent) {
        if let Some(cb) = &self.callback {
            cb(event);
        }
    }

    pub fn stage_start(&self, name: &str) {
        self.emit(ProgressEvent::StageStart {
            name: name.to_string(),
        });
    }

    pub fn stage_done(&self, name: &str, elapsed: Duration) {
        self.emit(ProgressEvent::StageDone {
            name: name.to_string(),
            elapsed,
        });
    }

    pub fn info(&self, message: &str) {
        self.emit(ProgressEvent::Info {
            message: message.to_string(),
        });
    }

    pub fn set_callback(&mut self, callback: ProgressCallback) {
        self.callback = Some(callback);
    }
}
