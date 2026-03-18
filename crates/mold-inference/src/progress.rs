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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Helper: create a callback that pushes debug-formatted events into a shared vec.
    fn capturing_callback() -> (ProgressCallback, Arc<Mutex<Vec<String>>>) {
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log_clone = Arc::clone(&log);
        let cb: ProgressCallback = Box::new(move |event: ProgressEvent| {
            log_clone.lock().unwrap().push(format!("{event:?}"));
        });
        (cb, log)
    }

    #[test]
    fn test_default_no_callback_no_panic() {
        let reporter = ProgressReporter::default();
        // All convenience methods should be callable without a callback set.
        reporter.stage_start("Loading model");
        reporter.stage_done("Loading model", Duration::from_millis(42));
        reporter.info("hello");
        reporter.emit(ProgressEvent::DenoiseStep {
            step: 1,
            total: 10,
            elapsed: Duration::from_millis(5),
        });
        // Reaching this point without panic is the assertion.
    }

    #[test]
    fn test_callback_receives_stage_start() {
        let mut reporter = ProgressReporter::default();
        let (cb, log) = capturing_callback();
        reporter.set_callback(cb);

        reporter.stage_start("Encoding prompt");

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(
            entries[0].contains("StageStart"),
            "expected StageStart, got: {}",
            entries[0]
        );
        assert!(
            entries[0].contains("Encoding prompt"),
            "expected stage name in event, got: {}",
            entries[0]
        );
    }

    #[test]
    fn test_callback_receives_denoise_step() {
        let mut reporter = ProgressReporter::default();
        let (cb, log) = capturing_callback();
        reporter.set_callback(cb);

        reporter.emit(ProgressEvent::DenoiseStep {
            step: 3,
            total: 20,
            elapsed: Duration::from_millis(100),
        });

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(
            entries[0].contains("DenoiseStep"),
            "expected DenoiseStep, got: {}",
            entries[0]
        );
        assert!(
            entries[0].contains("step: 3"),
            "expected step: 3, got: {}",
            entries[0]
        );
        assert!(
            entries[0].contains("total: 20"),
            "expected total: 20, got: {}",
            entries[0]
        );
    }

    #[test]
    fn test_stage_done_includes_elapsed() {
        let mut reporter = ProgressReporter::default();
        let (cb, log) = capturing_callback();
        reporter.set_callback(cb);

        let dur = Duration::from_secs(2) + Duration::from_millis(500);
        reporter.stage_done("VAE decode", dur);

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(
            entries[0].contains("StageDone"),
            "expected StageDone, got: {}",
            entries[0]
        );
        assert!(
            entries[0].contains("VAE decode"),
            "expected stage name, got: {}",
            entries[0]
        );
        // Duration debug format is "2.5s"
        assert!(
            entries[0].contains("2.5"),
            "expected elapsed ~2.5s, got: {}",
            entries[0]
        );
    }

    #[test]
    fn test_set_callback_replaces_previous() {
        let mut reporter = ProgressReporter::default();

        // Install first callback.
        let (cb1, log1) = capturing_callback();
        reporter.set_callback(cb1);
        reporter.info("first");
        assert_eq!(log1.lock().unwrap().len(), 1);

        // Replace with second callback.
        let (cb2, log2) = capturing_callback();
        reporter.set_callback(cb2);
        reporter.info("second");

        // Old callback must NOT have received the new event.
        assert_eq!(
            log1.lock().unwrap().len(),
            1,
            "old callback should not receive events after replacement"
        );
        // New callback must have received exactly one event.
        let entries2 = log2.lock().unwrap();
        assert_eq!(
            entries2.len(),
            1,
            "new callback should receive events after replacement"
        );
        assert!(
            entries2[0].contains("second"),
            "new callback got wrong event: {}",
            entries2[0]
        );
    }
}
