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
    /// Whether the audio-video transformer's audio branch may run at all.
    /// `false` only for an explicit `video_only` request with no audio
    /// output and no audio conditioning (#1037) — upstream's
    /// `LTXVideoOnlyModelConfigurator` omits the branch structurally
    /// (`model_configurator.py:56-101` @400fd31). When `true`, the runtime
    /// still applies its own multimodal-path gate, so silent renders on
    /// checkpoints without dual-AV prompt conditioning are unchanged.
    pub(crate) run_audio_branch: bool,
    pub(crate) uses_reference_video_conditioning: bool,
    pub(crate) uses_audio_conditioning: bool,
    pub(crate) uses_keyframe_conditioning: bool,
    pub(crate) uses_retake_masking: bool,
    pub(crate) stacked_lora_count: usize,
    pub(crate) blocks: Vec<ExecutionBlock>,
    pub(crate) denoise_passes: Vec<DenoisePassPlan>,
}

pub(crate) fn wants_audio_output(req: &GenerateRequest) -> bool {
    // An audio-only pipeline has nothing else to emit, so the audio branch is
    // not optional there — `enable_audio` cannot switch it off (the validator
    // rejects `enable_audio=false` alongside `pipeline=t2a`).
    if req
        .pipeline
        .is_some_and(mold_core::Ltx2PipelineMode::is_audio_only)
    {
        return true;
    }
    // An explicit video-only request renders silent even for MP4 output —
    // the validator already rejected it beside `enable_audio=true` and the
    // audio-only pipeline, so only the MP4 default is being overridden here.
    if req.video_only == Some(true) {
        return false;
    }
    req.enable_audio
        .unwrap_or(req.resolved_output_format() == OutputFormat::Mp4)
}

pub(crate) fn build_execution_graph(
    req: &GenerateRequest,
    pipeline: PipelineKind,
    conditioning: &StagedConditioning,
    preset: &Ltx2ModelPreset,
    stacked_lora_count: usize,
) -> Ltx2ExecutionGraph {
    let wants_audio_output = wants_audio_output(req);
    // Lip dub conditions on audio without ever being handed an audio file:
    // the reference clip's own speech is decoded out of the source video and
    // appended as reference tokens (`lipdub.py:166-171`, `:238`).
    let uses_audio_conditioning = conditioning.audio_path.is_some()
        || (matches!(pipeline, PipelineKind::LipDub) && conditioning.video_path.is_some());
    let uses_reference_video_conditioning = conditioning.video_path.is_some();
    let uses_keyframe_conditioning = conditioning.images.len() > 1;
    let uses_retake_masking = req.retake_range.is_some();
    let run_audio_branch =
        wants_audio_output || uses_audio_conditioning || req.video_only != Some(true);

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

    // Audio-only: one denoise pass over the audio branch, then the audio VAE
    // and vocoder. No source encoders, no spatial upsampler, no video decoder.
    if pipeline.is_audio_only() {
        let stage1 = DenoisePassPlan {
            block: ExecutionBlock::Stage1Denoise,
            sampler: SamplerMode::Euler,
            guidance: GuidanceMode::Multimodal,
            uses_distilled_checkpoint: false,
            apply_distilled_lora: false,
        };
        return Ltx2ExecutionGraph {
            preset_name: preset.name,
            feature_extractor: preset.feature_extractor,
            wants_audio_output: true,
            run_audio_branch: true,
            uses_reference_video_conditioning: false,
            uses_audio_conditioning: false,
            uses_keyframe_conditioning: false,
            uses_retake_masking: false,
            stacked_lora_count,
            blocks: vec![
                ExecutionBlock::PromptEncoder,
                ExecutionBlock::TextFeatureExtractor,
                stage1.block,
                ExecutionBlock::AudioDecoder,
                ExecutionBlock::Vocoder,
                ExecutionBlock::Export,
            ],
            denoise_passes: vec![stage1],
        };
    }

    let stage1 = DenoisePassPlan {
        block: ExecutionBlock::Stage1Denoise,
        sampler: SamplerMode::Euler,
        guidance: if matches!(pipeline, PipelineKind::OneStage | PipelineKind::Retake) {
            GuidanceMode::Simple
        } else {
            GuidanceMode::Multimodal
        },
        uses_distilled_checkpoint: matches!(
            pipeline,
            PipelineKind::Distilled
                | PipelineKind::IcLora
                | PipelineKind::Retake
                | PipelineKind::LipDub
        ),
        apply_distilled_lora: false,
    };
    blocks.push(stage1.block);

    let mut denoise_passes = vec![stage1];
    if !matches!(pipeline, PipelineKind::OneStage | PipelineKind::Retake) {
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
                PipelineKind::Distilled
                    | PipelineKind::IcLora
                    | PipelineKind::Retake
                    | PipelineKind::LipDub
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
        run_audio_branch,
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
            video_only: None,
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            prompt: "test".to_string(),
            negative_prompt: None,
            model: model.to_string(),
            width: 1216,
            height: 704,
            steps: 8,
            guidance: 3.0,
            seed: Some(7),
            batch_size: 1,
            output_format: Some(OutputFormat::Mp4),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: Some(97),
            fps: Some(24),
            upscale_model: None,
            gif_preview: false,
            enable_audio: Some(true),
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        }
    }

    fn dummy_paths() -> ModelPaths {
        ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
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
        Ltx2Engine::new(model_name.to_string(), paths, LoadStrategy::Sequential, 0)
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
        assert_eq!(graph.denoise_passes.len(), 1);
        assert!(graph
            .denoise_passes
            .iter()
            .all(|pass| pass.uses_distilled_checkpoint));
    }

    /// The audio-only graph must not carry a single spatial stage. A stray
    /// `SpatialUpsampler` or `VideoDecoder` here would make the scheduler
    /// price VRAM for a video render that never happens, and the runtime
    /// would load a video VAE it has no latents for.
    #[test]
    fn t2a_execution_graph_skips_every_video_stage() {
        let mut request = req("ltx-2.3-22b-dev:fp8");
        request.pipeline = Some(mold_core::Ltx2PipelineMode::T2a);
        request.output_format = Some(OutputFormat::Wav);
        let work_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&request, work_dir.path()).unwrap();
        let preset = preset_for_model("ltx-2.3-22b-dev:fp8").unwrap();
        let graph = build_execution_graph(&request, PipelineKind::T2a, &conditioning, &preset, 0);

        assert_eq!(
            graph.blocks,
            vec![
                ExecutionBlock::PromptEncoder,
                ExecutionBlock::TextFeatureExtractor,
                ExecutionBlock::Stage1Denoise,
                ExecutionBlock::AudioDecoder,
                ExecutionBlock::Vocoder,
                ExecutionBlock::Export,
            ]
        );
        assert_eq!(graph.denoise_passes.len(), 1);
        assert_eq!(graph.denoise_passes[0].guidance, GuidanceMode::Multimodal);
        assert_eq!(graph.denoise_passes[0].sampler, SamplerMode::Euler);
        assert!(!graph.denoise_passes[0].uses_distilled_checkpoint);
        assert!(!graph.denoise_passes[0].apply_distilled_lora);
        assert!(graph.wants_audio_output);
    }

    /// The #1037 table: only an explicit `video_only` request with no audio
    /// output and no audio conditioning switches the branch off; everything
    /// else — the absent default included — keeps it available.
    #[test]
    fn video_only_disables_the_audio_branch_unless_audio_is_requested_or_conditioned() {
        let temp_dir = tempfile::tempdir().unwrap();
        let preset = preset_for_model("ltx-2.3-22b-distilled:fp8").unwrap();
        let graph = |configure: &dyn Fn(&mut GenerateRequest)| {
            let mut request = req("ltx-2.3-22b-distilled:fp8");
            request.enable_audio = None;
            configure(&mut request);
            let conditioning = conditioning::stage_conditioning(&request, temp_dir.path()).unwrap();
            build_execution_graph(&request, PipelineKind::Distilled, &conditioning, &preset, 0)
        };

        // Absent: today's behavior, branch available.
        let default = graph(&|_| {});
        assert!(default.run_audio_branch);
        // Explicit false is the same as absent.
        let explicit_false = graph(&|request| request.video_only = Some(false));
        assert!(explicit_false.run_audio_branch);
        // video_only with a silent export: the one skip.
        let silent = graph(&|request| {
            request.video_only = Some(true);
            request.output_format = Some(OutputFormat::Gif);
        });
        assert!(!silent.run_audio_branch);
        assert!(!silent.wants_audio_output);
        assert!(!silent.blocks.contains(&ExecutionBlock::AudioDecoder));
        // video_only overrides the MP4 default-on too.
        let mp4 = graph(&|request| request.video_only = Some(true));
        assert!(!mp4.run_audio_branch);
        assert!(!mp4.wants_audio_output);
        // Conditioning audio keeps the branch (the validator refuses the
        // pair; the engine-side table stays safe independently).
        let conditioned = graph(&|request| {
            request.video_only = Some(true);
            request.audio_file = Some(b"fake".to_vec());
        });
        assert!(conditioned.run_audio_branch);
    }

    /// `enable_audio=false` cannot silence an audio-only pipeline — there
    /// would be nothing left to emit. (The validator rejects the combination
    /// outright; this pins the engine-side behaviour independently.)
    #[test]
    fn t2a_wants_audio_output_regardless_of_enable_audio() {
        let mut request = req("ltx-2.3-22b-dev:fp8");
        request.pipeline = Some(mold_core::Ltx2PipelineMode::T2a);
        request.output_format = Some(OutputFormat::Wav);
        request.enable_audio = Some(false);
        assert!(super::wants_audio_output(&request));
    }

    #[test]
    fn pipeline_materialization_attaches_native_preset_and_execution_graph() {
        let gemma_dir = tempfile::tempdir().unwrap();
        write_test_gemma_assets(gemma_dir.path());
        let engine = engine(
            "ltx-2.3-22b-distilled:fp8",
            dummy_paths_with_gemma_root(gemma_dir.path()),
        );
        let mut req = req("ltx-2.3-22b-distilled:fp8");
        req.enable_audio = Some(false);
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
