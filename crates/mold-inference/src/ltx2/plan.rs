use mold_core::{LoraWeight, TimeRange};
use serde::Serialize;

use super::conditioning::{StagedConditioning, StagedImage};
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
    pub(crate) fn module_name(self) -> &'static str {
        match self {
            Self::OneStage => "ltx_pipelines.ti2vid_one_stage",
            Self::TwoStage => "ltx_pipelines.ti2vid_two_stages",
            Self::TwoStageHq => "ltx_pipelines.ti2vid_two_stages_hq",
            Self::Distilled => "ltx_pipelines.distilled",
            Self::IcLora => "ltx_pipelines.ic_lora",
            Self::Keyframe => "ltx_pipelines.keyframe_interpolation",
            Self::A2Vid => "ltx_pipelines.a2vid_two_stage",
            Self::Retake => "ltx_pipelines.retake",
        }
    }

    pub(crate) fn requires_distilled_checkpoint(self) -> bool {
        matches!(self, Self::Distilled | Self::IcLora | Self::Retake)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Ltx2GeneratePlan {
    pub(crate) pipeline: PipelineKind,
    pub(crate) preset: Ltx2ModelPreset,
    pub(crate) execution_graph: Ltx2ExecutionGraph,
    pub(crate) checkpoint_path: String,
    pub(crate) distilled_checkpoint_path: Option<String>,
    pub(crate) distilled_lora_path: Option<String>,
    pub(crate) spatial_upsampler_path: Option<String>,
    pub(crate) gemma_root: String,
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
    pub(crate) quantization: Option<String>,
    pub(crate) streaming_prefetch_count: Option<u32>,
    pub(crate) conditioning: StagedConditioning,
    pub(crate) loras: Vec<LoraWeight>,
    pub(crate) retake_range: Option<TimeRange>,
}

impl Ltx2GeneratePlan {
    pub(crate) fn to_bridge_request(&self) -> BridgeRequest {
        BridgeRequest {
            module: self.pipeline.module_name().to_string(),
            checkpoint_path: self.checkpoint_path.clone(),
            distilled_checkpoint_path: self.distilled_checkpoint_path.clone(),
            distilled_lora_path: self.distilled_lora_path.clone(),
            spatial_upsampler_path: self.spatial_upsampler_path.clone(),
            gemma_root: self.gemma_root.clone(),
            output_path: self.output_path.clone(),
            prompt: self.prompt.clone(),
            negative_prompt: self.negative_prompt.clone(),
            seed: self.seed,
            width: self.width,
            height: self.height,
            num_frames: self.num_frames,
            frame_rate: self.frame_rate,
            num_inference_steps: self.num_inference_steps,
            quantization: self.quantization.clone(),
            streaming_prefetch_count: self.streaming_prefetch_count,
            images: self
                .conditioning
                .images
                .iter()
                .cloned()
                .map(BridgeImage::from)
                .collect(),
            loras: self.loras.clone(),
            audio_path: self.conditioning.audio_path.clone(),
            video_path: self.conditioning.video_path.clone(),
            retake_start_seconds: self.retake_range.as_ref().map(|range| range.start_seconds),
            retake_end_seconds: self.retake_range.as_ref().map(|range| range.end_seconds),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct BridgeRequest {
    module: String,
    checkpoint_path: String,
    distilled_checkpoint_path: Option<String>,
    distilled_lora_path: Option<String>,
    spatial_upsampler_path: Option<String>,
    gemma_root: String,
    output_path: String,
    prompt: String,
    negative_prompt: Option<String>,
    seed: u64,
    width: u32,
    height: u32,
    num_frames: u32,
    frame_rate: u32,
    num_inference_steps: u32,
    quantization: Option<String>,
    streaming_prefetch_count: Option<u32>,
    images: Vec<BridgeImage>,
    loras: Vec<LoraWeight>,
    audio_path: Option<String>,
    video_path: Option<String>,
    retake_start_seconds: Option<f32>,
    retake_end_seconds: Option<f32>,
}

#[derive(Serialize)]
struct BridgeImage {
    path: String,
    frame: u32,
    strength: f32,
}

impl From<StagedImage> for BridgeImage {
    fn from(image: StagedImage) -> Self {
        Self {
            path: image.path,
            frame: image.frame,
            strength: image.strength,
        }
    }
}
