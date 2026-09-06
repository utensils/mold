use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

/// Attempt-scoped cooperative cancellation shared between a caller and one
/// inference invocation.
///
/// Cancellation never interrupts a CUDA kernel. Engines poll the token only at
/// explicit safe points, so Candle and CUDA objects remain owned and dropped by
/// the same worker thread that started the invocation.
#[derive(Clone, Debug, Default)]
pub struct InferenceCancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl InferenceCancellationToken {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub fn checkpoint(&self) -> Result<(), InferenceCancelled> {
        if self.is_cancelled() {
            Err(InferenceCancelled)
        } else {
            Ok(())
        }
    }

    pub(crate) fn flag(&self) -> &AtomicBool {
        &self.cancelled
    }
}

/// Read-only view of one attempt-scoped cancellation token.
///
/// Cached runtimes may retain this view only while an explicit attempt guard
/// is active. The view deliberately exposes no cancellation mutator, so model
/// code cannot cancel its caller or replace the current attempt authority.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
#[derive(Clone, Debug)]
pub(crate) struct InferenceCancellationObserver {
    token: InferenceCancellationToken,
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
impl InferenceCancellationObserver {
    pub(crate) fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }
}

/// Typed non-fatal result returned when an inference attempt observes its
/// cooperative cancellation token at a safe point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InferenceCancelled;

impl std::fmt::Display for InferenceCancelled {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("inference cancelled")
    }
}

impl std::error::Error for InferenceCancelled {}

pub fn is_inference_cancelled(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| cause.is::<InferenceCancelled>())
}

/// Machine-readable inference phase used by the scheduler's learned timing
/// model. Display names remain independently evolvable client copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressPhase {
    ModelLoad,
    /// Face-identity extraction: the detector, the recognizer, the BiSeNet
    /// parser, the EVA02-CLIP tower, and the IDFormer, run on the render's own
    /// leased device before anything else touches it (#1227 phase 2,
    /// `docs/architecture/pulid-perf.md` §5).
    ///
    /// Emitted only by the ONE sibling of a parent request that actually
    /// performs the once-per-parent extraction, and not at all for a request
    /// that does not condition on a face or whose effective `id_weight` is
    /// zero. Every other sibling's `identity_extract_ms` stays `None`, exactly
    /// as `cold_load_ms` already does on a warm reuse.
    IdentityExtract,
    PromptEncode,
    /// Legacy family-neutral VAE work. Existing image/video families retain
    /// this bucket while synchronized A/V families can report their output
    /// decoders independently.
    Vae,
    /// Visual output decode, including streaming into a family-owned encoder
    /// sink when decode and publication are one bounded pipeline phase.
    VisualDecode,
    /// Audio latent decode into the publication waveform.
    AudioDecode,
    /// Final synchronized media-container mux.
    Mux,
    Upscale,
}

/// Progress events emitted during model loading and inference.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// A named stage has started (e.g. "Loading T5 encoder (CPU)")
    StageStart { name: String },
    /// The most recent stage completed, with its elapsed time
    StageDone { name: String, elapsed: Duration },
    /// Bounded work inside a stage, such as one streaming transformer layer.
    /// This remains separate from `DenoiseStep` so setup and inner evaluations
    /// never masquerade as completed sampler steps.
    StageProgress {
        name: String,
        current: usize,
        total: usize,
    },
    /// A scheduler-observable phase completed. This maps to the same
    /// StageDone SSE shape as a display-only stage but retains typed identity
    /// inside the inference/server boundary.
    PhaseDone {
        phase: ProgressPhase,
        name: String,
        elapsed: Duration,
    },
    /// Informational message (e.g. "CUDA detected, using GPU")
    Info { message: String },
    /// A cached artifact was reused instead of recomputed.
    CacheHit { resource: String },
    /// A single denoising step completed.
    DenoiseStep {
        step: usize,
        total: usize,
        elapsed: Duration,
    },
    /// Progress loading model weights from disk.
    WeightLoad {
        bytes_loaded: u64,
        bytes_total: u64,
        component: String,
    },
    /// A low-fidelity live preview of the denoising latent (small PNG at
    /// latent resolution; clients upscale). Arc keeps Clone cheap.
    Preview {
        image_png: std::sync::Arc<Vec<u8>>,
        step: usize,
        total: usize,
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
    cancellation: Option<InferenceCancellationToken>,
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

    pub fn stage_progress(&self, name: &str, current: usize, total: usize) {
        self.emit(ProgressEvent::StageProgress {
            name: name.to_string(),
            current,
            total,
        });
    }

    pub fn phase_done(&self, phase: ProgressPhase, name: &str, elapsed: Duration) {
        self.emit(ProgressEvent::PhaseDone {
            phase,
            name: name.to_string(),
            elapsed,
        });
    }

    /// Finish a named stage that is ALSO a typed scheduler phase.
    ///
    /// One event, not two. [`ProgressEvent::PhaseDone`] already converts to
    /// the wire's `StageDone`, so emitting `stage_done` beside `phase_done`
    /// with the same label sends the SAME completion twice and every client
    /// prints the line twice — which is exactly what the Hunyuan3D log did
    /// for encode, decode, and extraction. Use this wherever a stage carries
    /// a phase; keep the bare [`Self::stage_done`] for display-only stages.
    pub fn stage_complete(&self, phase: ProgressPhase, name: &str, elapsed: Duration) {
        self.phase_done(phase, name, elapsed);
    }

    pub fn info(&self, message: &str) {
        self.emit(ProgressEvent::Info {
            message: message.to_string(),
        });
    }

    pub fn cache_hit(&self, resource: &str) {
        self.emit(ProgressEvent::CacheHit {
            resource: resource.to_string(),
        });
    }

    pub fn weight_load(&self, component: &str, bytes_loaded: u64, bytes_total: u64) {
        self.emit(ProgressEvent::WeightLoad {
            bytes_loaded,
            bytes_total,
            component: component.to_string(),
        });
    }

    pub fn set_callback(&mut self, callback: ProgressCallback) {
        self.callback = Some(callback);
    }

    pub fn clear_callback(&mut self) {
        self.callback = None;
    }

    pub fn set_cancellation_token(&mut self, token: InferenceCancellationToken) {
        self.cancellation = Some(token);
    }

    pub fn clear_cancellation_token(&mut self) {
        self.cancellation = None;
    }

    /// Run one private attempt with exactly one authoritative cancellation
    /// token installed for every progress checkpoint. Any previously installed
    /// token is restored even when the attempt panics; callers cannot supply a
    /// second token for the reporter and diverge the two cancellation paths.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) fn with_attempt_cancellation_token<T>(
        &mut self,
        token: InferenceCancellationToken,
        run: impl FnOnce(&ProgressReporter) -> T,
    ) -> T {
        let previous = self.cancellation.replace(token);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(self)));
        self.cancellation = previous;
        match outcome {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Snapshot a read-only view for one explicitly bounded runtime attempt.
    /// Callers must not retain it across `clear_cancellation_token` or a later
    /// `set_cancellation_token`; cached runtimes should install it behind an
    /// attempt-scoped guard.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) fn cancellation_observer(&self) -> Option<InferenceCancellationObserver> {
        self.cancellation
            .clone()
            .map(|token| InferenceCancellationObserver { token })
    }

    /// Return a typed cancellation result only at a caller-selected safe point.
    pub fn checkpoint(&self) -> Result<(), InferenceCancelled> {
        match &self.cancellation {
            Some(token) => token.checkpoint(),
            None => Ok(()),
        }
    }

    pub(crate) fn cancellation_token(&self) -> InferenceCancellationToken {
        self.cancellation.clone().unwrap_or_default()
    }
}

impl From<ProgressEvent> for mold_core::SseProgressEvent {
    fn from(event: ProgressEvent) -> Self {
        match event {
            ProgressEvent::StageStart { name } => mold_core::SseProgressEvent::StageStart { name },
            ProgressEvent::StageDone { name, elapsed } => mold_core::SseProgressEvent::StageDone {
                name,
                elapsed_ms: elapsed.as_millis() as u64,
            },
            ProgressEvent::StageProgress {
                name,
                current,
                total,
            } => mold_core::SseProgressEvent::StageProgress {
                name,
                current,
                total,
            },
            ProgressEvent::PhaseDone {
                name,
                elapsed,
                phase: _,
            } => mold_core::SseProgressEvent::StageDone {
                name,
                elapsed_ms: elapsed.as_millis() as u64,
            },
            ProgressEvent::Info { message } => mold_core::SseProgressEvent::Info { message },
            ProgressEvent::CacheHit { resource } => {
                mold_core::SseProgressEvent::CacheHit { resource }
            }
            ProgressEvent::DenoiseStep {
                step,
                total,
                elapsed,
            } => mold_core::SseProgressEvent::DenoiseStep {
                step,
                total,
                elapsed_ms: elapsed.as_millis() as u64,
            },
            ProgressEvent::WeightLoad {
                bytes_loaded,
                bytes_total,
                component,
            } => mold_core::SseProgressEvent::WeightLoad {
                bytes_loaded,
                bytes_total,
                component,
            },
            ProgressEvent::Preview {
                image_png,
                step,
                total,
            } => mold_core::SseProgressEvent::Preview {
                image: {
                    use base64::Engine as _;
                    base64::engine::general_purpose::STANDARD.encode(image_png.as_slice())
                },
                step,
                total,
            },
        }
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
        reporter.cache_hit("prompt conditioning");
        reporter.emit(ProgressEvent::DenoiseStep {
            step: 1,
            total: 10,
            elapsed: Duration::from_millis(5),
        });
        // Reaching this point without panic is the assertion.
    }

    #[test]
    fn cancellation_token_is_shared_and_attempt_scoped() {
        let first_attempt = InferenceCancellationToken::default();
        let first_attempt_worker = first_attempt.clone();
        let second_attempt = InferenceCancellationToken::default();

        assert!(first_attempt_worker.checkpoint().is_ok());
        first_attempt.cancel();

        assert_eq!(first_attempt_worker.checkpoint(), Err(InferenceCancelled));
        assert!(second_attempt.checkpoint().is_ok());
    }

    #[test]
    fn reporter_observes_and_clears_cooperative_cancellation() {
        let token = InferenceCancellationToken::default();
        let mut reporter = ProgressReporter::default();
        reporter.set_cancellation_token(token.clone());

        assert!(reporter.checkpoint().is_ok());
        token.cancel();
        let error = anyhow::Error::from(reporter.checkpoint().unwrap_err());
        assert!(is_inference_cancelled(&error));

        reporter.clear_cancellation_token();
        assert!(reporter.checkpoint().is_ok());
    }

    #[cfg(feature = "h3-private-uat")]
    #[test]
    fn read_only_observer_tracks_only_its_attempt_token() {
        let first = InferenceCancellationToken::default();
        let mut reporter = ProgressReporter::default();
        reporter.set_cancellation_token(first.clone());
        let observer = reporter.cancellation_observer().unwrap();

        assert!(!observer.is_cancelled());
        first.cancel();
        assert!(observer.is_cancelled());

        reporter.clear_cancellation_token();
        assert!(reporter.cancellation_observer().is_none());
    }

    #[test]
    fn denoise_loop_stops_at_the_next_safe_checkpoint_and_can_be_reused() {
        let token = InferenceCancellationToken::default();
        let mut reporter = ProgressReporter::default();
        reporter.set_cancellation_token(token.clone());
        let mut completed_steps = 0;

        let result = (|| {
            for step in 0..10 {
                reporter.checkpoint()?;
                completed_steps += 1;
                if step == 3 {
                    token.cancel();
                }
            }
            Ok::<(), InferenceCancelled>(())
        })();

        assert_eq!(result, Err(InferenceCancelled));
        assert_eq!(completed_steps, 4);
        reporter.clear_cancellation_token();
        assert!(reporter.checkpoint().is_ok());
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
    fn typed_phase_done_preserves_the_existing_sse_wire_shape() {
        for phase in [
            ProgressPhase::Vae,
            ProgressPhase::VisualDecode,
            ProgressPhase::AudioDecode,
            ProgressPhase::Mux,
        ] {
            let wire: mold_core::SseProgressEvent = ProgressEvent::PhaseDone {
                phase,
                name: "Finishing output".to_string(),
                elapsed: Duration::from_millis(37),
            }
            .into();

            assert!(matches!(
                wire,
                mold_core::SseProgressEvent::StageDone {
                    name,
                    elapsed_ms: 37
                } if name == "Finishing output"
            ));
        }
    }

    /// A stage that carries a typed phase completes ONCE. `PhaseDone` maps
    /// to the wire's `StageDone`, so emitting `stage_done` beside it printed
    /// every Hunyuan3D stage line twice on every client.
    #[test]
    fn a_typed_stage_completes_exactly_once_on_the_wire() {
        let mut reporter = ProgressReporter::default();
        let (cb, log) = capturing_callback();
        reporter.set_callback(cb);

        reporter.stage_start("Decoding volume");
        reporter.stage_complete(
            ProgressPhase::Vae,
            "Decoding volume",
            Duration::from_millis(1200),
        );

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 2, "{entries:?}");
        assert!(entries[0].contains("StageStart"), "{}", entries[0]);
        assert_eq!(
            entries
                .iter()
                .filter(|entry| entry.contains("Decoding volume")
                    && (entry.contains("StageDone") || entry.contains("PhaseDone")))
                .count(),
            1,
            "a stage must report one completion, got {entries:?}"
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

    #[test]
    fn test_clear_callback_stops_future_events() {
        let mut reporter = ProgressReporter::default();
        let (cb, log) = capturing_callback();
        reporter.set_callback(cb);
        reporter.info("before-clear");
        reporter.clear_callback();
        reporter.info("after-clear");

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].contains("before-clear"));
    }

    #[test]
    fn test_weight_load_emits_structured_event() {
        let mut reporter = ProgressReporter::default();
        let (cb, log) = capturing_callback();
        reporter.set_callback(cb);

        reporter.weight_load("FLUX transformer", 500_000_000, 1_000_000_000);

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].contains("WeightLoad"));
        assert!(entries[0].contains("FLUX transformer"));
        assert!(entries[0].contains("500000000"));
    }

    #[test]
    fn test_cache_hit_emits_structured_event() {
        let mut reporter = ProgressReporter::default();
        let (cb, log) = capturing_callback();
        reporter.set_callback(cb);

        reporter.cache_hit("prompt conditioning");

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].contains("CacheHit"));
        assert!(entries[0].contains("prompt conditioning"));
    }
}
