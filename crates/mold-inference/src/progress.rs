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
}

/// Callback type for receiving progress events.
pub type ProgressCallback = Box<dyn Fn(ProgressEvent) + Send + Sync>;
