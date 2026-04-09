use mold_core::{GenerateRequest, OutputFormat};

use super::conditioning::StagedConditioning;
use super::plan::PipelineKind;
use super::preset::{GemmaFeatureExtractorKind, Ltx2ModelPreset};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecutionBlock {
    PromptEncoder,
    TextFeatureExtractor,
    SourceImageEncoder,
    SourceVideoEncoder,
    SourceAudioEncoder,
    Stage1Denoise,
    SpatialUpsampler,
    Stage2Denoise,
    TemporalUpsampler,
    VideoDecoder,
    AudioDecoder,
    Vocoder,
    Export,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuidanceMode {
    Simple,
    Multimodal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SamplerMode {
    Euler,
    Res2S,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenoisePassPlan {
    pub(crate) block: ExecutionBlock,
    pub(crate) sampler: SamplerMode,
    pub(crate) guidance: GuidanceMode,
    pub(crate) uses_distilled_checkpoint: bool,
    pub(crate) apply_distilled_lora: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Ltx2ExecutionGraph {
    pub(crate) preset_name: &'static str,
    pub(crate) feature_extractor: GemmaFeatureExtractorKind,
    pub(crate) wants_audio_output: bool,
    pub(crate) uses_reference_video_conditioning: bool,
    pub(crate) uses_audio_conditioning: bool,
    pub(crate) uses_keyframe_conditioning: bool,
    pub(crate) uses_retake_masking: bool,
    pub(crate) stacked_lora_count: usize,
    pub(crate) blocks: Vec<ExecutionBlock>,
    pub(crate) denoise_passes: Vec<DenoisePassPlan>,
}

fn wants_audio_output(req: &GenerateRequest) -> bool {
    req.enable_audio
        .unwrap_or(req.output_format == OutputFormat::Mp4)
}

pub(crate) fn build_execution_graph(
    req: &GenerateRequest,
    pipeline: PipelineKind,
    conditioning: &StagedConditioning,
    preset: &Ltx2ModelPreset,
    stacked_lora_count: usize,
) -> Ltx2ExecutionGraph {
    let wants_audio_output = wants_audio_output(req);
    let uses_audio_conditioning = conditioning.audio_path.is_some();
    let uses_reference_video_conditioning = conditioning.video_path.is_some();
    let uses_keyframe_conditioning = conditioning.images.len() > 1;
    let uses_retake_masking = req.retake_range.is_some();

    let mut blocks = vec![
        ExecutionBlock::PromptEncoder,
        ExecutionBlock::TextFeatureExtractor,
    ];
    if !conditioning.images.is_empty() {
        blocks.push(ExecutionBlock::SourceImageEncoder);
    }
    if uses_reference_video_conditioning {
        blocks.push(ExecutionBlock::SourceVideoEncoder);
    }
    if uses_audio_conditioning {
        blocks.push(ExecutionBlock::SourceAudioEncoder);
    }

    let stage1 = DenoisePassPlan {
        block: ExecutionBlock::Stage1Denoise,
        sampler: SamplerMode::Euler,
        guidance: if matches!(pipeline, PipelineKind::OneStage) {
            GuidanceMode::Simple
        } else {
            GuidanceMode::Multimodal
        },
        uses_distilled_checkpoint: matches!(
            pipeline,
            PipelineKind::Distilled | PipelineKind::IcLora | PipelineKind::Retake
        ),
        apply_distilled_lora: false,
    };
    blocks.push(stage1.block);

    let mut denoise_passes = vec![stage1];
    if !matches!(pipeline, PipelineKind::OneStage) {
        blocks.push(ExecutionBlock::SpatialUpsampler);
        let stage2 = DenoisePassPlan {
            block: ExecutionBlock::Stage2Denoise,
            sampler: if matches!(pipeline, PipelineKind::TwoStageHq) {
                SamplerMode::Res2S
            } else {
                SamplerMode::Euler
            },
            guidance: GuidanceMode::Multimodal,
            uses_distilled_checkpoint: matches!(
                pipeline,
                PipelineKind::Distilled | PipelineKind::IcLora | PipelineKind::Retake
            ),
            apply_distilled_lora: matches!(
                pipeline,
                PipelineKind::TwoStage
                    | PipelineKind::TwoStageHq
                    | PipelineKind::A2Vid
                    | PipelineKind::Keyframe
            ),
        };
        denoise_passes.push(stage2);
        blocks.push(stage2.block);
    }

    if req.temporal_upscale.is_some() {
        blocks.push(ExecutionBlock::TemporalUpsampler);
    }
    blocks.push(ExecutionBlock::VideoDecoder);
    if wants_audio_output {
        blocks.push(ExecutionBlock::AudioDecoder);
        blocks.push(ExecutionBlock::Vocoder);
    }
    blocks.push(ExecutionBlock::Export);

    Ltx2ExecutionGraph {
        preset_name: preset.name,
        feature_extractor: preset.feature_extractor,
        wants_audio_output,
        uses_reference_video_conditioning,
        uses_audio_conditioning,
        uses_keyframe_conditioning,
        uses_retake_masking,
        stacked_lora_count,
        blocks,
        denoise_passes,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use mold_core::{GenerateRequest, ModelPaths, OutputFormat, TimeRange};

    use super::{build_execution_graph, ExecutionBlock, GuidanceMode, SamplerMode};
    use crate::{
        engine::LoadStrategy,
        ltx2::{conditioning, plan::PipelineKind, preset::preset_for_model, Ltx2Engine},
    };

    fn req(model: &str) -> GenerateRequest {
        GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: model.to_string(),
            width: 1216,
            height: 704,
            steps: 8,
            guidance: 3.0,
            seed: Some(7),
            batch_size: 1,
            output_format: OutputFormat::Mp4,
            embed_metadata: None,
            scheduler: None,
            source_image: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: Some(97),
            fps: Some(24),
            upscale_model: None,
            gif_preview: false,
            enable_audio: Some(true),
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        }
    }

    fn dummy_paths() -> ModelPaths {
        ModelPaths {
            transformer: PathBuf::from("/tmp/ltx2.safetensors"),
            transformer_shards: vec![],
            vae: PathBuf::from("/tmp/unused"),
            spatial_upscaler: Some(PathBuf::from("/tmp/spatial.safetensors")),
            temporal_upscaler: Some(PathBuf::from("/tmp/temporal.safetensors")),
            distilled_lora: Some(PathBuf::from("/tmp/distilled-lora.safetensors")),
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![PathBuf::from("/tmp/gemma/tokenizer.json")],
            text_tokenizer: None,
            decoder: None,
        }
    }

    fn dummy_paths_with_gemma_root(root: &std::path::Path) -> ModelPaths {
        let mut paths = dummy_paths();
        paths.text_encoder_files = vec![root.join("tokenizer.json")];
        paths
    }

    fn write_test_gemma_assets(root: &std::path::Path) {
        fs::write(
            root.join("tokenizer.json"),
            r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "WhitespaceSplit"
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<eos>": 7,
      "test": 11
    },
    "unk_token": "<eos>"
  }
}"#,
        )
        .unwrap();
        fs::write(
            root.join("special_tokens_map.json"),
            r#"{"eos_token":"<eos>"}"#,
        )
        .unwrap();
    }

    fn engine(model_name: &str, paths: ModelPaths) -> Ltx2Engine {
        Ltx2Engine::new(model_name.to_string(), paths, LoadStrategy::Sequential)
    }

    #[test]
    fn one_stage_graph_skips_stage_two_blocks() {
        let req = req("ltx-2-19b-dev:fp8");
        let conditioning =
            conditioning::stage_conditioning(&req, tempfile::tempdir().unwrap().path()).unwrap();
        let graph = build_execution_graph(
            &req,
            PipelineKind::OneStage,
            &conditioning,
            &preset_for_model(&req.model).unwrap(),
            0,
        );
        assert_eq!(graph.denoise_passes.len(), 1);
        assert!(!graph.blocks.contains(&ExecutionBlock::SpatialUpsampler));
        assert_eq!(graph.denoise_passes[0].guidance, GuidanceMode::Simple);
    }

    #[test]
    fn two_stage_hq_graph_uses_second_order_sampler() {
        let mut req = req("ltx-2-19b-dev:fp8");
        req.pipeline = Some(mold_core::Ltx2PipelineMode::TwoStageHq);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let graph = build_execution_graph(
            &req,
            PipelineKind::TwoStageHq,
            &conditioning,
            &preset_for_model(&req.model).unwrap(),
            1,
        );
        assert_eq!(graph.denoise_passes.len(), 2);
        assert!(graph.blocks.contains(&ExecutionBlock::SpatialUpsampler));
        assert_eq!(graph.denoise_passes[1].sampler, SamplerMode::Res2S);
        assert!(graph.denoise_passes[1].apply_distilled_lora);
        assert_eq!(graph.stacked_lora_count, 1);
    }

    #[test]
    fn a2vid_graph_tracks_audio_conditioning_and_output_blocks() {
        let mut req = req("ltx-2.3-22b-dev:fp8");
        req.audio_file = Some(b"fake".to_vec());
        req.pipeline = Some(mold_core::Ltx2PipelineMode::A2Vid);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let graph = build_execution_graph(
            &req,
            PipelineKind::A2Vid,
            &conditioning,
            &preset_for_model(&req.model).unwrap(),
            0,
        );
        assert!(graph.uses_audio_conditioning);
        assert!(graph.wants_audio_output);
        assert!(graph.blocks.contains(&ExecutionBlock::SourceAudioEncoder));
        assert!(graph.blocks.contains(&ExecutionBlock::AudioDecoder));
        assert!(graph.blocks.contains(&ExecutionBlock::Vocoder));
    }

    #[test]
    fn retake_graph_includes_source_media_and_distilled_checkpoint_usage() {
        let mut req = req("ltx-2-19b-distilled:fp8");
        req.source_video = Some(vec![0, 1, 2]);
        req.audio_file = Some(vec![3, 4, 5]);
        req.retake_range = Some(TimeRange {
            start_seconds: 0.5,
            end_seconds: 1.25,
        });
        req.pipeline = Some(mold_core::Ltx2PipelineMode::Retake);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let graph = build_execution_graph(
            &req,
            PipelineKind::Retake,
            &conditioning,
            &preset_for_model(&req.model).unwrap(),
            2,
        );
        assert!(graph.uses_reference_video_conditioning);
        assert!(graph.uses_audio_conditioning);
        assert!(graph.uses_retake_masking);
        assert_eq!(graph.denoise_passes.len(), 2);
        assert!(graph
            .denoise_passes
            .iter()
            .all(|pass| pass.uses_distilled_checkpoint));
    }

    #[test]
    fn pipeline_materialization_attaches_native_preset_and_execution_graph() {
        let gemma_dir = tempfile::tempdir().unwrap();
        write_test_gemma_assets(gemma_dir.path());
        let engine = engine(
            "ltx-2.3-22b-distilled:fp8",
            dummy_paths_with_gemma_root(gemma_dir.path()),
        );
        let req = req("ltx-2.3-22b-distilled:fp8");
        let temp_dir = tempfile::tempdir().unwrap();
        let plan = engine
            .materialize_request(&req, temp_dir.path(), &temp_dir.path().join("out.mp4"))
            .unwrap();
        assert_eq!(plan.preset.name, "ltx-2.3-22b");
        assert_eq!(plan.execution_graph.preset_name, "ltx-2.3-22b");
        assert_eq!(
            plan.execution_graph.feature_extractor,
            plan.preset.feature_extractor
        );
        assert_eq!(plan.prompt_tokens.conditional.valid_len(), 1);
        assert_eq!(plan.prompt_tokens.pad_token_id, 7);
    }
}
