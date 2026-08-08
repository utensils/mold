//! Authorization-bound adapters for the private MiniMax H3 CUDA runtime.
//!
//! This module is compiled only by the developer-only `h3-private-uat`
//! feature. It has no registry, capability, catalog, download, server, or CLI
//! integration. The public family remains fail-closed; the adapters merely
//! connect already-authenticated Candle objects to the runtime-neutral block
//! streaming contract used by private qualification.

use anyhow::{bail, Result};
use mold_candle::minimax_h3::{
    H3BlockProgress, H3BlockStack, H3ComfyInt8BlockLoader, H3ForwardInput, H3FrozenPackedLayout,
    H3LoadedTransformerBlock, H3StreamedTransformer, H3TransformerOutput, H3TransformerStep,
    H3TransformerTask,
};

use super::engine::H3StreamedTransformerExecutor;
use super::offload::{FrozenH3BlockStreamingPlan, H3BlockLoader};
use super::pipeline::{H3PipelineCheckpoint, H3PipelineEvent, H3PipelinePhase};

const H3_MAIN_BLOCK_COUNT: usize = 50;

/// Route-fenced adapter over the exact Comfy block loader returned alongside
/// one resident streamed transformer.
pub(crate) struct H3PrivateComfyBlockLoader {
    inner: H3ComfyInt8BlockLoader,
    device_id: String,
    execution_fingerprint: String,
}

impl H3PrivateComfyBlockLoader {
    fn new(inner: H3ComfyInt8BlockLoader, plan: &FrozenH3BlockStreamingPlan) -> Result<Self> {
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
        Ok(self.inner.load_block(index)?)
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
        let step = self
            .transformer
            .begin_step_with_observer(input, layout, |event| {
                private_transformer_checkpoint(checkpoint, event)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))
            })?;
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
        Ok(self.transformer.finish_step_with_observer(step, |event| {
            private_transformer_checkpoint(checkpoint, event)
                .map_err(|error| candle_core::Error::Msg(error.to_string()))
        })?)
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
        H3PrivateComfyBlockLoader::new(loader, plan)?,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
