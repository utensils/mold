use mold_core::{
    LoraWeight, Ltx2GuidanceOverrides, Ltx2SpatialUpscale, Ltx2TemporalUpscale, TimeRange,
};

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
    LipDub,
    /// Text-to-audio: audio-only generation with no video modality at all.
    /// Upstream's `T2AOneStagePipeline`.
    T2a,
}

impl PipelineKind {
    /// The wire mode this runtime kind corresponds to.
    ///
    /// The two enums are deliberately separate — one is the request contract,
    /// the other the resolved runtime choice — but properties that admission
    /// and execution must agree on live on the wire type, and this is the
    /// bridge. `select_pipeline` is the inverse mapping.
    pub(crate) fn wire_mode(self) -> mold_core::Ltx2PipelineMode {
        use mold_core::Ltx2PipelineMode as Mode;
        match self {
            Self::OneStage => Mode::OneStage,
            Self::TwoStage => Mode::TwoStage,
            Self::TwoStageHq => Mode::TwoStageHq,
            Self::Distilled => Mode::Distilled,
            Self::IcLora => Mode::IcLora,
            Self::Keyframe => Mode::Keyframe,
            Self::A2Vid => Mode::A2Vid,
            Self::Retake => Mode::Retake,
            Self::LipDub => Mode::LipDub,
            Self::T2a => Mode::T2a,
        }
    }

    pub(crate) fn requires_distilled_checkpoint(self) -> bool {
        matches!(
            self,
            Self::Distilled | Self::IcLora | Self::Retake | Self::LipDub
        )
    }

    /// Whether the lip-dub IC-LoRA's in-context reference video has to be
    /// re-encoded for every denoise stage.
    ///
    /// Generic IC-LoRA drops both the adapter and its reference before stage 2
    /// (upstream `ic_lora.py:99-103` builds `stage_2` with `loras=()` and
    /// `ic_lora.py:279-288` conditions it on images only). Lip dub keeps one
    /// `DiffusionStage` for both passes and rebuilds the full conditioning set
    /// each time (`lipdub.py:96-106`, `:236`, `:265`), so the mouth stays
    /// locked to the reference through the refinement.
    pub(crate) fn keeps_reference_video_in_stage_two(self) -> bool {
        matches!(self, Self::LipDub)
    }

    /// Whether this pipeline renders audio samples instead of video frames.
    /// Audio-only plans skip every spatial stage and the video VAE, and their
    /// `width`/`height`/`spatial_upscale` fields carry no meaning.
    pub(crate) fn is_audio_only(self) -> bool {
        matches!(self, Self::T2a)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Ltx2GeneratePlan {
    /// Directory for the EXR sidecar, when the request asked for one. The
    /// decode has to know before it runs: the 8-bit conversion is lossy, so
    /// HDR cannot be recovered from the frames afterwards.
    pub(crate) hdr_exr_dir: Option<String>,
    pub(crate) hdr_exr_full_float: bool,
    /// Chain-stage EXR window on the stitched timeline. `None` keeps the
    /// single-render behaviour (write every decoded frame from index 0).
    /// Stamped only by `render_chain_stage` from its sidecar argument —
    /// never derived from the request, so a stage or extend render whose
    /// request carries `hdr_exr_dir` without a window writes nothing.
    pub(crate) hdr_exr_window: Option<crate::chain::ExrStageWindow>,
    /// First reference-video frame this render conditions on. Non-zero only
    /// for chain stages, where each stage regrades its own temporal window
    /// of the shared SDR reference (upstream slices reference conditioning
    /// per temporal tile the same way: `hdr_ic_lora.py:521-541`).
    pub(crate) reference_frame_offset: u32,
    /// Pre-computed text embeddings shipped beside an IC-LoRA control's
    /// weights. Upstream's HDR pipeline requires these and never encodes a
    /// prompt, so when this is set the Gemma encode is skipped entirely and
    /// the saved `video_context` becomes the conditioning.
    pub(crate) scene_embeddings_path: Option<String>,
    pub(crate) pipeline: PipelineKind,
    pub(crate) preset: Ltx2ModelPreset,
    pub(crate) checkpoint_is_distilled: bool,
    pub(crate) execution_graph: Ltx2ExecutionGraph,
    pub(crate) checkpoint_path: String,
    pub(crate) vae_checkpoint_path: String,
    pub(crate) vae_in_checkpoint: bool,
    pub(crate) text_projection_path: Option<String>,
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
    /// Request-level overrides for the multimodal guider. `None` (and any
    /// unset field) keeps the per-(pipeline, stage) constants.
    pub(crate) guidance_overrides: Option<Ltx2GuidanceOverrides>,
    /// VRAM the scheduler admitted this job against
    /// (`predicted_vram_peak_bytes` from the frozen execution plan).
    ///
    /// The adaptive residency planner used to size itself against whatever the
    /// card happened to have free, which meant a job admitted at an 11.5 GB
    /// peak would quietly self-size to 25 GB and die at the first denoise
    /// step. When set, the planner budgets `min(grant, usable free VRAM)`.
    /// `None` keeps the legacy free-VRAM-only behaviour for callers with no
    /// scheduler authority (CLI local runs, tests).
    pub(crate) vram_grant_bytes: Option<u64>,
}
