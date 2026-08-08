//! Authorization-bound adapters for the private MiniMax H3 CUDA runtime.
//!
//! This module is compiled only by the developer-only `h3-private-uat`
//! feature. It has no registry, capability, catalog, download, server, or CLI
//! integration. The public family remains fail-closed; the adapters merely
//! connect already-authenticated Candle objects to the runtime-neutral block
//! streaming contract used by private qualification.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{bail, Result};
use mold_candle::minimax_h3::{
    H3BlockProgress, H3BlockStack, H3ComfyInt8BlockLoader, H3ComfyInt8Cancellation, H3ForwardInput,
    H3FrozenPackedLayout, H3LoadedTransformerBlock, H3StreamedTransformer, H3TransformerOutput,
    H3TransformerStep, H3TransformerTask,
};

use crate::progress::{InferenceCancellationObserver, InferenceCancelled, ProgressReporter};

use super::engine::H3StreamedTransformerExecutor;
use super::offload::{FrozenH3BlockStreamingPlan, H3BlockLoader};
use super::pipeline::{H3PipelineCheckpoint, H3PipelineEvent, H3PipelinePhase};

const H3_MAIN_BLOCK_COUNT: usize = 50;

#[derive(Default)]
struct H3PrivateComfyCancellationState {
    next_attempt_id: u64,
    active_attempt_id: Option<u64>,
    attempt_guard_alive: bool,
    observer: Option<InferenceCancellationObserver>,
    active_operation_attempt_id: Option<u64>,
    candle_was_polled: bool,
    candle_observed_cancellation: bool,
}

/// One cached Candle loader can serve multiple serialized inference attempts.
/// This slot refreshes its read-only cancellation authority for each attempt
/// instead of retaining the token that happened to load the model initially.
#[derive(Clone, Default)]
pub(crate) struct H3PrivateComfyCancellationSlot {
    state: Arc<Mutex<H3PrivateComfyCancellationState>>,
}

impl H3PrivateComfyCancellationSlot {
    pub(crate) fn install(
        &self,
        progress: &ProgressReporter,
    ) -> Result<H3PrivateComfyCancellationGuard> {
        let mut state = lock_cancellation_state(&self.state);
        if state.active_attempt_id.is_some() {
            bail!("private H3 Comfy cancellation authority already has an active attempt");
        }
        state.next_attempt_id = state
            .next_attempt_id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("private H3 Comfy attempt identity overflow"))?;
        let attempt_id = state.next_attempt_id;
        state.active_attempt_id = Some(attempt_id);
        state.attempt_guard_alive = true;
        state.observer = progress.cancellation_observer();
        state.active_operation_attempt_id = None;
        state.candle_was_polled = false;
        state.candle_observed_cancellation = false;
        Ok(H3PrivateComfyCancellationGuard {
            slot: self.clone(),
            attempt_id,
        })
    }

    /// Run one Candle load operation and prove that it polled this exact slot.
    /// This closes the otherwise-untyped trait-object seam between the cached
    /// Candle loader and the attempt authority retained by this adapter.
    pub(crate) fn run_candle_operation<T>(
        &self,
        operation: impl FnOnce() -> candle_core::Result<T>,
    ) -> Result<T> {
        let operation_guard = H3PrivateComfyOperationGuard::begin(self)?;
        let result = operation();
        operation_guard.finish(result)
    }

    #[cfg(test)]
    fn has_active_attempt(&self) -> bool {
        lock_cancellation_state(&self.state)
            .active_attempt_id
            .is_some()
    }
}

impl H3ComfyInt8Cancellation for H3PrivateComfyCancellationSlot {
    fn is_cancelled(&self) -> bool {
        let mut state = lock_cancellation_state(&self.state);
        if state.active_attempt_id.is_none()
            || state.active_operation_attempt_id != state.active_attempt_id
        {
            return false;
        }
        let cancelled = state
            .observer
            .as_ref()
            .is_some_and(InferenceCancellationObserver::is_cancelled);
        state.candle_was_polled = true;
        state.candle_observed_cancellation |= cancelled;
        cancelled
    }
}

#[must_use = "dropping the guard clears the cached runtime's attempt authority"]
pub(crate) struct H3PrivateComfyCancellationGuard {
    slot: H3PrivateComfyCancellationSlot,
    attempt_id: u64,
}

impl Drop for H3PrivateComfyCancellationGuard {
    fn drop(&mut self) {
        let mut state = lock_cancellation_state(&self.slot.state);
        if state.active_attempt_id == Some(self.attempt_id) {
            state.attempt_guard_alive = false;
            if state.active_operation_attempt_id != Some(self.attempt_id) {
                clear_cancellation_attempt(&mut state, self.attempt_id);
            }
        }
    }
}

struct H3PrivateComfyOperationGuard {
    slot: H3PrivateComfyCancellationSlot,
    attempt_id: u64,
    finished: bool,
}

impl H3PrivateComfyOperationGuard {
    fn begin(slot: &H3PrivateComfyCancellationSlot) -> Result<Self> {
        let mut state = lock_cancellation_state(&slot.state);
        let attempt_id = state
            .active_attempt_id
            .filter(|_| state.attempt_guard_alive)
            .ok_or_else(|| {
                anyhow::anyhow!("private H3 Comfy load has no active cancellation attempt")
            })?;
        if state.active_operation_attempt_id.is_some() {
            bail!("private H3 Comfy cancellation authority already has an active load");
        }
        state.active_operation_attempt_id = Some(attempt_id);
        state.candle_was_polled = false;
        state.candle_observed_cancellation = false;
        Ok(Self {
            slot: slot.clone(),
            attempt_id,
            finished: false,
        })
    }

    fn finish<T>(mut self, result: candle_core::Result<T>) -> Result<T> {
        let (identity_matches, guard_alive, polled, observed) = {
            let mut state = lock_cancellation_state(&self.slot.state);
            let identity_matches = state.active_attempt_id == Some(self.attempt_id)
                && state.active_operation_attempt_id == Some(self.attempt_id);
            let guard_alive = identity_matches && state.attempt_guard_alive;
            let polled = identity_matches && std::mem::take(&mut state.candle_was_polled);
            let observed =
                identity_matches && std::mem::take(&mut state.candle_observed_cancellation);
            if identity_matches {
                state.active_operation_attempt_id = None;
                if !guard_alive {
                    clear_cancellation_attempt(&mut state, self.attempt_id);
                }
            }
            (identity_matches, guard_alive, polled, observed)
        };
        self.finished = true;
        if !identity_matches {
            bail!("private H3 Comfy cancellation attempt changed during a load");
        }
        if !polled {
            bail!("private H3 Comfy loader did not poll its bound cancellation authority");
        }
        if observed {
            return Err(InferenceCancelled.into());
        }
        if !guard_alive {
            bail!("private H3 Comfy cancellation authority ended during a load");
        }
        result.map_err(Into::into)
    }
}

impl Drop for H3PrivateComfyOperationGuard {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let mut state = lock_cancellation_state(&self.slot.state);
        if state.active_attempt_id == Some(self.attempt_id)
            && state.active_operation_attempt_id == Some(self.attempt_id)
        {
            state.active_operation_attempt_id = None;
            state.candle_was_polled = false;
            state.candle_observed_cancellation = false;
            if !state.attempt_guard_alive {
                clear_cancellation_attempt(&mut state, self.attempt_id);
            }
        }
    }
}

fn clear_cancellation_attempt(state: &mut H3PrivateComfyCancellationState, attempt_id: u64) {
    if state.active_attempt_id == Some(attempt_id) {
        state.active_attempt_id = None;
        state.attempt_guard_alive = false;
        state.observer = None;
        state.active_operation_attempt_id = None;
        state.candle_was_polled = false;
        state.candle_observed_cancellation = false;
    }
}

fn lock_cancellation_state(
    state: &Mutex<H3PrivateComfyCancellationState>,
) -> MutexGuard<'_, H3PrivateComfyCancellationState> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Route-fenced adapter over the exact Comfy block loader returned alongside
/// one resident streamed transformer.
pub(crate) struct H3PrivateComfyBlockLoader {
    inner: H3ComfyInt8BlockLoader,
    device_id: String,
    execution_fingerprint: String,
    cancellation: H3PrivateComfyCancellationSlot,
}

impl H3PrivateComfyBlockLoader {
    fn new(
        inner: H3ComfyInt8BlockLoader,
        plan: &FrozenH3BlockStreamingPlan,
        cancellation: H3PrivateComfyCancellationSlot,
    ) -> Result<Self> {
        plan.validate()?;
        if plan.resident_block_count != 0 || plan.prefetch_depth != 0 {
            bail!(
                "private H3 Comfy streaming requires zero resident blocks and zero prefetch depth"
            );
        }
        if inner.block_count() != H3_MAIN_BLOCK_COUNT {
            bail!(
                "private H3 Comfy loader exposes {} blocks, expected {H3_MAIN_BLOCK_COUNT}",
                inner.block_count()
            );
        }
        Ok(Self {
            inner,
            device_id: plan.device_id.clone(),
            execution_fingerprint: plan.execution_fingerprint.clone(),
            cancellation,
        })
    }

    pub(crate) fn bytes_read(&self) -> u64 {
        self.inner.bytes_read()
    }

    pub(crate) fn live_block_count(&self) -> usize {
        self.inner.live_block_count()
    }
}

impl H3BlockLoader for H3PrivateComfyBlockLoader {
    type Block = H3LoadedTransformerBlock;

    fn load_block(
        &mut self,
        index: usize,
        device_id: &str,
        execution_fingerprint: &str,
    ) -> Result<Self::Block> {
        if device_id != self.device_id || execution_fingerprint != self.execution_fingerprint {
            bail!("private H3 Comfy block load differs from the frozen execution route");
        }
        self.cancellation
            .run_candle_operation(|| self.inner.load_block(index))
    }
}

/// Per-step executor over the resident projections, refiners, MM-RoPE, and
/// output heads. Main blocks are supplied one at a time by the paired loader.
pub(crate) struct H3PrivateComfyTransformerExecutor {
    transformer: H3StreamedTransformer,
    step: Option<H3TransformerStep>,
}

impl H3PrivateComfyTransformerExecutor {
    fn new(transformer: H3StreamedTransformer) -> Self {
        Self {
            transformer,
            step: None,
        }
    }
}

impl H3StreamedTransformerExecutor<H3LoadedTransformerBlock> for H3PrivateComfyTransformerExecutor {
    fn begin_step(
        &mut self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        if self.step.is_some() {
            bail!("private H3 streamed transformer already has an active denoise step");
        }
        let error_bridge = H3TransformerCallbackErrorBridge::default();
        let result = self
            .transformer
            .begin_step_with_observer(input, layout, |event| {
                private_transformer_checkpoint(checkpoint, event)
                    .map_err(|error| error_bridge.capture(error))
            });
        let step = error_bridge.finish(result)?;
        self.step = Some(step);
        Ok(())
    }

    fn forward_block(
        &mut self,
        index: usize,
        block: &H3LoadedTransformerBlock,
        _checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        let step = self
            .step
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("private H3 streamed step was not started"))?;
        Ok(step.forward_block(index, block)?)
    }

    fn finish_step(
        &mut self,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TransformerOutput> {
        let step = self
            .step
            .take()
            .ok_or_else(|| anyhow::anyhow!("private H3 streamed step was not started"))?;
        let error_bridge = H3TransformerCallbackErrorBridge::default();
        let result = self.transformer.finish_step_with_observer(step, |event| {
            private_transformer_checkpoint(checkpoint, event)
                .map_err(|error| error_bridge.capture(error))
        });
        error_bridge.finish(result)
    }

    fn abort_step(&mut self) {
        self.step = None;
    }
}

/// Consume only a transformer/loader pair returned by the same authenticated
/// checkpoint open. Pointer identity is still checked again by Candle for
/// every streamed block.
pub(crate) fn pair_private_comfy_stream(
    transformer: H3StreamedTransformer,
    loader: H3ComfyInt8BlockLoader,
    plan: &FrozenH3BlockStreamingPlan,
    expected_task: H3TransformerTask,
    cancellation: H3PrivateComfyCancellationSlot,
) -> Result<(H3PrivateComfyBlockLoader, H3PrivateComfyTransformerExecutor)> {
    let exact_open_pair = loader.is_exact_pair_for(&transformer);
    validate_stream_pair(
        plan,
        expected_task,
        transformer.task(),
        transformer.block_count(),
        transformer.checkpoint_identity_sha256(),
        loader.task(),
        loader.block_count(),
        loader.checkpoint_identity_sha256(),
        exact_open_pair,
    )?;
    Ok((
        H3PrivateComfyBlockLoader::new(loader, plan, cancellation)?,
        H3PrivateComfyTransformerExecutor::new(transformer),
    ))
}

#[allow(clippy::too_many_arguments)]
fn validate_stream_pair(
    plan: &FrozenH3BlockStreamingPlan,
    expected_task: H3TransformerTask,
    transformer_task: H3TransformerTask,
    transformer_blocks: usize,
    transformer_checkpoint: Option<&str>,
    loader_task: H3TransformerTask,
    loader_blocks: usize,
    loader_checkpoint: &str,
    exact_open_pair: bool,
) -> Result<()> {
    plan.validate()?;
    if plan.resident_block_count != 0 || plan.prefetch_depth != 0 {
        bail!("private H3 Comfy streaming requires zero resident blocks and zero prefetch depth");
    }
    if transformer_task != expected_task || loader_task != expected_task {
        bail!("private H3 Comfy transformer/loader task authority differs");
    }
    if transformer_blocks != H3_MAIN_BLOCK_COUNT || loader_blocks != H3_MAIN_BLOCK_COUNT {
        bail!("private H3 Comfy transformer/loader block count differs from production");
    }
    if transformer_checkpoint != Some(loader_checkpoint) {
        bail!("private H3 Comfy transformer and loader came from different checkpoints");
    }
    if !exact_open_pair {
        bail!("private H3 Comfy transformer and loader came from different checkpoint opens");
    }
    Ok(())
}

fn private_transformer_checkpoint(
    checkpoint: &mut dyn H3PipelineCheckpoint,
    event: H3BlockProgress,
) -> Result<()> {
    let completed = match event.stack {
        // The block stream owns 0..50. Refiners occur before block zero and
        // output projection occurs after block fifty, so repeated boundary
        // checkpoints preserve cancellation without inventing another scale.
        H3BlockStack::TokenRefiner => 0,
        H3BlockStack::Transformer => event.completed.min(H3_MAIN_BLOCK_COUNT),
        H3BlockStack::Output => H3_MAIN_BLOCK_COUNT,
    };
    checkpoint.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::TransformerBlock,
        completed,
        total: H3_MAIN_BLOCK_COUNT,
    })
}

/// Candle observers can return only `candle_core::Error`. Retain the original
/// pipeline error beside that transport value so cancellation and future typed
/// failures survive the transformer boundary unchanged.
#[derive(Clone, Default)]
struct H3TransformerCallbackErrorBridge {
    original: Rc<RefCell<Option<anyhow::Error>>>,
}

impl H3TransformerCallbackErrorBridge {
    fn capture(&self, error: anyhow::Error) -> candle_core::Error {
        let message = error.to_string();
        let mut original = self.original.borrow_mut();
        if original.is_none() {
            *original = Some(error);
        }
        candle_core::Error::Msg(message)
    }

    fn finish<T>(&self, result: candle_core::Result<T>) -> Result<T> {
        let original = self.original.borrow_mut().take();
        match (result, original) {
            (Ok(value), None) => Ok(value),
            (Ok(_), Some(error)) | (Err(_), Some(error)) => Err(error),
            (Err(error), None) => Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::{is_inference_cancelled, InferenceCancellationToken};

    const FINGERPRINT: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const CHECKPOINT: &str = "2222222222222222222222222222222222222222222222222222222222222222";

    fn plan() -> FrozenH3BlockStreamingPlan {
        FrozenH3BlockStreamingPlan::new("gpu-0", FINGERPRINT, 0, 0).unwrap()
    }

    #[test]
    fn exact_comfy_stream_pair_is_accepted() {
        validate_stream_pair(
            &plan(),
            H3TransformerTask::T2VaFl2Va,
            H3TransformerTask::T2VaFl2Va,
            50,
            Some(CHECKPOINT),
            H3TransformerTask::T2VaFl2Va,
            50,
            CHECKPOINT,
            true,
        )
        .unwrap();
    }

    #[test]
    fn mixed_task_or_checkpoint_pair_is_rejected() {
        let error = validate_stream_pair(
            &plan(),
            H3TransformerTask::T2VaFl2Va,
            H3TransformerTask::Ref2Va,
            50,
            Some(CHECKPOINT),
            H3TransformerTask::T2VaFl2Va,
            50,
            CHECKPOINT,
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("task authority"));

        let error = validate_stream_pair(
            &plan(),
            H3TransformerTask::T2VaFl2Va,
            H3TransformerTask::T2VaFl2Va,
            50,
            Some(CHECKPOINT),
            H3TransformerTask::T2VaFl2Va,
            50,
            FINGERPRINT,
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("different checkpoints"));

        let error = validate_stream_pair(
            &plan(),
            H3TransformerTask::T2VaFl2Va,
            H3TransformerTask::T2VaFl2Va,
            50,
            Some(CHECKPOINT),
            H3TransformerTask::T2VaFl2Va,
            50,
            CHECKPOINT,
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("different checkpoint opens"));
    }

    #[test]
    fn multi_block_residency_is_rejected_for_single_live_loader() {
        let plan = FrozenH3BlockStreamingPlan::new("gpu-0", FINGERPRINT, 1, 0).unwrap();
        let error = validate_stream_pair(
            &plan,
            H3TransformerTask::T2VaFl2Va,
            H3TransformerTask::T2VaFl2Va,
            50,
            Some(CHECKPOINT),
            H3TransformerTask::T2VaFl2Va,
            50,
            CHECKPOINT,
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("zero resident blocks"));
    }

    #[test]
    fn comfy_cancellation_slot_refreshes_attempts_and_preserves_typed_cancel() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<H3PrivateComfyCancellationSlot>();

        let slot = H3PrivateComfyCancellationSlot::default();
        let first_token = InferenceCancellationToken::default();
        let mut progress = ProgressReporter::default();
        progress.set_cancellation_token(first_token.clone());
        let first_guard = slot.install(&progress).unwrap();

        assert!(slot.has_active_attempt());
        assert!(!H3ComfyInt8Cancellation::is_cancelled(&slot));
        first_token.cancel();
        let error = slot
            .run_candle_operation::<()>(|| {
                assert!(H3ComfyInt8Cancellation::is_cancelled(&slot));
                Err(candle_core::Error::Msg(
                    "Candle cancellation transport".into(),
                ))
            })
            .unwrap_err();
        assert!(is_inference_cancelled(&error));
        assert!(slot.install(&progress).is_err());
        drop(first_guard);
        assert!(!slot.has_active_attempt());

        let second_token = InferenceCancellationToken::default();
        progress.set_cancellation_token(second_token);
        let _second_guard = slot.install(&progress).unwrap();
        assert!(!H3ComfyInt8Cancellation::is_cancelled(&slot));
        let error = slot
            .run_candle_operation::<()>(|| {
                assert!(!H3ComfyInt8Cancellation::is_cancelled(&slot));
                Err(candle_core::Error::Msg("unrelated loader failure".into()))
            })
            .unwrap_err();
        assert!(!is_inference_cancelled(&error));
        assert!(error.to_string().contains("unrelated loader failure"));

        let error = slot
            .run_candle_operation::<()>(|| {
                Err(candle_core::Error::Msg(
                    "loader used a different authority".into(),
                ))
            })
            .unwrap_err();
        assert!(error.to_string().contains("did not poll its bound"));
    }

    #[test]
    fn comfy_cancellation_guard_clears_attempt_state_during_unwind() {
        let slot = H3PrivateComfyCancellationSlot::default();
        let progress = ProgressReporter::default();
        let guard = slot.install(&progress).unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = slot.run_candle_operation::<()>(|| {
                assert!(!H3ComfyInt8Cancellation::is_cancelled(&slot));
                panic!("synthetic load panic");
            });
        }));

        assert!(panic.is_err());
        assert!(slot.has_active_attempt());
        slot.run_candle_operation(|| {
            assert!(!H3ComfyInt8Cancellation::is_cancelled(&slot));
            Ok(())
        })
        .unwrap();
        drop(guard);
        assert!(!slot.has_active_attempt());
        assert!(slot.install(&progress).is_ok());
    }

    #[test]
    fn guard_drop_during_load_fences_replacement_attempt() {
        let slot = H3PrivateComfyCancellationSlot::default();
        let progress = ProgressReporter::default();
        let guard = slot.install(&progress).unwrap();
        let worker_slot = slot.clone();
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(0);
        let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(0);
        let worker = std::thread::spawn(move || {
            worker_slot.run_candle_operation(|| {
                assert!(!H3ComfyInt8Cancellation::is_cancelled(&worker_slot));
                started_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
                Ok(())
            })
        });

        started_rx.recv().unwrap();
        drop(guard);
        assert!(slot.install(&progress).is_err());
        resume_tx.send(()).unwrap();
        let error = worker.join().unwrap().unwrap_err();
        assert!(error.to_string().contains("ended during a load"));
        assert!(!slot.has_active_attempt());

        let _next_guard = slot.install(&progress).unwrap();
        assert!(!H3ComfyInt8Cancellation::is_cancelled(&slot));
    }

    #[test]
    fn overlapping_candle_operations_fail_before_running_the_second_load() {
        let slot = H3PrivateComfyCancellationSlot::default();
        let progress = ProgressReporter::default();
        let _guard = slot.install(&progress).unwrap();
        let worker_slot = slot.clone();
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(0);
        let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(0);
        let worker = std::thread::spawn(move || {
            worker_slot.run_candle_operation(|| {
                assert!(!H3ComfyInt8Cancellation::is_cancelled(&worker_slot));
                started_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
                Ok(())
            })
        });

        started_rx.recv().unwrap();
        let mut second_ran = false;
        let error = slot
            .run_candle_operation::<()>(|| {
                second_ran = true;
                Ok(())
            })
            .unwrap_err();
        assert!(error.to_string().contains("already has an active load"));
        assert!(!second_ran);
        resume_tx.send(()).unwrap();
        worker.join().unwrap().unwrap();
    }

    struct CancelCheckpoint;

    impl H3PipelineCheckpoint for CancelCheckpoint {
        fn checkpoint(&mut self, _event: H3PipelineEvent) -> Result<()> {
            Err(InferenceCancelled.into())
        }
    }

    #[test]
    fn transformer_callback_bridge_preserves_typed_cancellation_and_first_error() {
        let bridge = H3TransformerCallbackErrorBridge::default();
        let mut checkpoint = CancelCheckpoint;
        let transport = private_transformer_checkpoint(
            &mut checkpoint,
            H3BlockProgress {
                stack: H3BlockStack::TokenRefiner,
                completed: 0,
                total: 2,
            },
        )
        .map_err(|error| bridge.capture(error))
        .unwrap_err();
        let error = bridge.finish::<()>(Err(transport)).unwrap_err();
        assert!(is_inference_cancelled(&error));

        let bridge = H3TransformerCallbackErrorBridge::default();
        let _first_transport = bridge.capture(InferenceCancelled.into());
        let later_transport = bridge.capture(anyhow::anyhow!("later callback failure"));
        let error = bridge.finish::<()>(Err(later_transport)).unwrap_err();
        assert!(is_inference_cancelled(&error));
    }
}
