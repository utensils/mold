use mold_core::{LoraWeight, Ltx2SpatialUpscale, Ltx2TemporalUpscale, TimeRange};

use super::conditioning::StagedConditioning;
use super::execution::Ltx2ExecutionGraph;
use super::preset::Ltx2ModelPreset;
use super::text::gemma::EncodedPromptPair;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipelineKind {
    OneStage,
    TwoStage,
    TwoStageHq,
    Distilled,
    IcLora,
    Keyframe,
    A2Vid,
    Retake,
}

impl PipelineKind {
    pub(crate) fn requires_distilled_checkpoint(self) -> bool {
        matches!(self, Self::Distilled | Self::IcLora | Self::Retake)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Ltx2GeneratePlan {
    pub(crate) pipeline: PipelineKind,
    pub(crate) preset: Ltx2ModelPreset,
    pub(crate) checkpoint_is_distilled: bool,
    pub(crate) execution_graph: Ltx2ExecutionGraph,
    pub(crate) checkpoint_path: String,
    #[allow(dead_code)]
    pub(crate) distilled_checkpoint_path: Option<String>,
    pub(crate) distilled_lora_path: Option<String>,
    pub(crate) spatial_upsampler_path: Option<String>,
    pub(crate) temporal_upsampler_path: Option<String>,
    pub(crate) gemma_root: String,
    #[allow(dead_code)]
    pub(crate) output_path: String,
    pub(crate) prompt: String,
    pub(crate) negative_prompt: Option<String>,
    pub(crate) prompt_tokens: EncodedPromptPair,
    pub(crate) seed: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) num_frames: u32,
    pub(crate) frame_rate: u32,
    pub(crate) num_inference_steps: u32,
    pub(crate) guidance: f64,
    #[allow(dead_code)]
    pub(crate) quantization: Option<String>,
    pub(crate) streaming_prefetch_count: Option<u32>,
    pub(crate) conditioning: StagedConditioning,
    pub(crate) loras: Vec<LoraWeight>,
    pub(crate) retake_range: Option<TimeRange>,
    pub(crate) spatial_upscale: Option<Ltx2SpatialUpscale>,
    pub(crate) temporal_upscale: Option<Ltx2TemporalUpscale>,
}
