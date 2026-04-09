#![allow(dead_code)]

use anyhow::Result;
use candle_core::Tensor;

use super::conditioning::retake_temporal_mask;
use super::model::{
    audio_temporal_positions, cross_modal_temporal_positions, video_token_positions,
    AudioLatentShape, AudioPatchifier, SpatioTemporalScaleFactors, VideoLatentPatchifier,
    VideoLatentShape, VideoPixelShape,
};
use super::plan::Ltx2GeneratePlan;
use super::text::prompt_encoder::{NativePromptEncoder, NativePromptEncoding};

pub const LTX2_VIDEO_LATENT_CHANNELS: usize = 128;
pub const LTX2_AUDIO_LATENT_CHANNELS: usize = 8;
pub const LTX2_AUDIO_MEL_BINS: usize = 16;
pub const LTX2_AUDIO_SAMPLE_RATE: usize = 16_000;
pub const LTX2_AUDIO_HOP_LENGTH: usize = 160;
pub const LTX2_AUDIO_LATENT_DOWNSAMPLE_FACTOR: usize = 4;

#[derive(Debug)]
pub struct NativePreparedRun {
    pub prompt: NativePromptEncoding,
    pub video_pixel_shape: VideoPixelShape,
    pub video_latent_shape: VideoLatentShape,
    pub audio_latent_shape: Option<AudioLatentShape>,
    pub video_positions: Tensor,
    pub audio_positions: Option<Tensor>,
    pub cross_modal_temporal_positions: Option<(Tensor, Tensor)>,
    pub retake_mask: Option<Vec<f32>>,
}

pub struct Ltx2RuntimeSession {
    prompt_encoder: NativePromptEncoder,
}

impl Ltx2RuntimeSession {
    pub fn new(prompt_encoder: NativePromptEncoder) -> Self {
        Self { prompt_encoder }
    }

    pub fn prepare(&mut self, plan: &Ltx2GeneratePlan) -> Result<NativePreparedRun> {
        let prompt = self
            .prompt_encoder
            .encode_prompt_pair(&plan.prompt_tokens)?;
        let pixel_shape = VideoPixelShape {
            batch: 1,
            frames: plan.num_frames as usize,
            height: plan.height as usize,
            width: plan.width as usize,
            fps: plan.frame_rate as f32,
        };
        let scale_factors = SpatioTemporalScaleFactors::default();
        let video_latent_shape = VideoLatentShape::from_pixel_shape(
            pixel_shape,
            LTX2_VIDEO_LATENT_CHANNELS,
            scale_factors,
        );
        let video_patchifier = VideoLatentPatchifier::new(1);
        let video_positions = video_token_positions(
            video_patchifier,
            video_latent_shape,
            self.prompt_encoder.device(),
        )?;

        let wants_audio_latents =
            plan.execution_graph.wants_audio_output || plan.execution_graph.uses_audio_conditioning;
        let (audio_latent_shape, audio_positions, cross_modal_temporal_positions) =
            if wants_audio_latents {
                let audio_shape = AudioLatentShape::from_video_pixel_shape(
                    pixel_shape,
                    LTX2_AUDIO_LATENT_CHANNELS,
                    LTX2_AUDIO_MEL_BINS,
                    LTX2_AUDIO_SAMPLE_RATE,
                    LTX2_AUDIO_HOP_LENGTH,
                    LTX2_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
                );
                let audio_patchifier = AudioPatchifier::new(
                    LTX2_AUDIO_SAMPLE_RATE,
                    LTX2_AUDIO_HOP_LENGTH,
                    LTX2_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
                    true,
                    0,
                );
                let audio_positions = audio_temporal_positions(
                    audio_patchifier,
                    audio_shape,
                    self.prompt_encoder.device(),
                )?;
                let cross_modal =
                    cross_modal_temporal_positions(&video_positions, &audio_positions)?;
                (Some(audio_shape), Some(audio_positions), Some(cross_modal))
            } else {
                (None, None, None)
            };

        let retake_mask = plan
            .retake_range
            .as_ref()
            .map(|range| retake_temporal_mask(range, plan.frame_rate, plan.num_frames))
            .transpose()?;

        Ok(NativePreparedRun {
            prompt,
            video_pixel_shape: pixel_shape,
            video_latent_shape,
            audio_latent_shape,
            video_positions,
            audio_positions,
            cross_modal_temporal_positions,
            retake_mask,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use mold_core::{GenerateRequest, OutputFormat, TimeRange};

    use super::{Ltx2RuntimeSession, LTX2_AUDIO_LATENT_CHANNELS, LTX2_VIDEO_LATENT_CHANNELS};
    use crate::ltx2::conditioning::{self, StagedConditioning};
    use crate::ltx2::execution::build_execution_graph;
    use crate::ltx2::plan::{Ltx2GeneratePlan, PipelineKind};
    use crate::ltx2::preset::{preset_for_model, Ltx2ModelPreset};
    use crate::ltx2::text::connectors::PaddingSide;
    use crate::ltx2::text::encoder::{GemmaConfig, GemmaHiddenStateEncoder};
    use crate::ltx2::text::gemma::{EncodedPromptPair, PromptTokens};
    use crate::ltx2::text::prompt_encoder::{
        build_embeddings_processor, ConnectorSpec, NativePromptEncoder,
    };

    fn req(model: &str, format: OutputFormat, enable_audio: Option<bool>) -> GenerateRequest {
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
            output_format: format,
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
            enable_audio,
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

    fn prompt_pair() -> EncodedPromptPair {
        EncodedPromptPair {
            conditional: PromptTokens {
                input_ids: vec![0, 0, 5],
                attention_mask: vec![0, 0, 1],
            },
            unconditional: PromptTokens {
                input_ids: vec![0, 0, 0],
                attention_mask: vec![0, 0, 0],
            },
            pad_token_id: 0,
            eos_token_id: Some(1),
            max_length: 3,
        }
    }

    fn tiny_gemma_config() -> GemmaConfig {
        GemmaConfig {
            attention_bias: false,
            head_dim: 4,
            hidden_activation: candle_nn::Activation::GeluPytorchTanh,
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_hidden_layers: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_local_base_freq: 10_000.0,
            vocab_size: 16,
            final_logit_softcapping: None,
            attn_logit_softcapping: None,
            query_pre_attn_scalar: 4,
            sliding_window: 4,
            sliding_window_pattern: 2,
            max_position_embeddings: 32,
        }
    }

    fn zero_gemma_var_builder(cfg: &GemmaConfig) -> VarBuilder<'static> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, &Device::Cpu).unwrap(),
        );
        for layer in 0..cfg.num_hidden_layers {
            for name in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                let (rows, cols) = match name {
                    "self_attn.q_proj" => (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
                    "self_attn.k_proj" | "self_attn.v_proj" => {
                        (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
                    }
                    "self_attn.o_proj" => (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
                    "mlp.gate_proj" | "mlp.up_proj" => (cfg.intermediate_size, cfg.hidden_size),
                    "mlp.down_proj" => (cfg.hidden_size, cfg.intermediate_size),
                    _ => unreachable!(),
                };
                tensors.insert(
                    format!("model.layers.{layer}.{name}.weight"),
                    Tensor::zeros((rows, cols), DType::F32, &Device::Cpu).unwrap(),
                );
            }
            for name in [
                "self_attn.q_norm",
                "self_attn.k_norm",
                "input_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "post_attention_layernorm",
            ] {
                let dim = if name.contains("q_norm") || name.contains("k_norm") {
                    cfg.head_dim
                } else {
                    cfg.hidden_size
                };
                tensors.insert(
                    format!("model.layers.{layer}.{name}.weight"),
                    Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
                );
            }
        }
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    }

    fn zero_connector_source_var_builder() -> VarBuilder<'static> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "text_embedding_projection.video_aggregate_embed.weight".to_string(),
            Tensor::zeros((8, 24), DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "text_embedding_projection.video_aggregate_embed.bias".to_string(),
            Tensor::zeros(8, DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "text_embedding_projection.audio_aggregate_embed.weight".to_string(),
            Tensor::zeros((4, 24), DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "text_embedding_projection.audio_aggregate_embed.bias".to_string(),
            Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap(),
        );
        for (prefix, dim) in [
            ("model.diffusion_model.video_embeddings_connector", 8usize),
            ("model.diffusion_model.audio_embeddings_connector", 4usize),
        ] {
            for linear_name in ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"] {
                tensors.insert(
                    format!("{prefix}.transformer_1d_blocks.0.{linear_name}.weight"),
                    Tensor::zeros((dim, dim), DType::F32, &Device::Cpu).unwrap(),
                );
                tensors.insert(
                    format!("{prefix}.transformer_1d_blocks.0.{linear_name}.bias"),
                    Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
                );
            }
            tensors.insert(
                format!("{prefix}.transformer_1d_blocks.0.ff.net.0.proj.weight"),
                Tensor::zeros((dim * 4, dim), DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.transformer_1d_blocks.0.ff.net.0.proj.bias"),
                Tensor::zeros(dim * 4, DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.transformer_1d_blocks.0.ff.net.2.weight"),
                Tensor::zeros((dim, dim * 4), DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.transformer_1d_blocks.0.ff.net.2.bias"),
                Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.learnable_registers"),
                Tensor::zeros((128, dim), DType::F32, &Device::Cpu).unwrap(),
            );
        }
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    }

    fn runtime_session() -> Ltx2RuntimeSession {
        let cfg = tiny_gemma_config();
        let gemma = GemmaHiddenStateEncoder::new(&cfg, zero_gemma_var_builder(&cfg)).unwrap();
        let prompt_encoder = NativePromptEncoder::new(
            gemma,
            build_embeddings_processor(
                zero_connector_source_var_builder(),
                crate::ltx2::preset::GemmaFeatureExtractorKind::V2DualAv,
                cfg.hidden_size,
                cfg.num_hidden_layers,
                8,
                Some(4),
                ConnectorSpec {
                    prefix: "model.diffusion_model.video_embeddings_connector.",
                    num_attention_heads: 2,
                    attention_head_dim: 4,
                    num_layers: 1,
                },
                Some(ConnectorSpec {
                    prefix: "model.diffusion_model.audio_embeddings_connector.",
                    num_attention_heads: 1,
                    attention_head_dim: 4,
                    num_layers: 1,
                }),
            )
            .unwrap(),
            PaddingSide::Left,
        );
        Ltx2RuntimeSession::new(prompt_encoder)
    }

    fn build_plan(
        req: &GenerateRequest,
        preset: Ltx2ModelPreset,
        conditioning: StagedConditioning,
    ) -> Ltx2GeneratePlan {
        let graph = build_execution_graph(req, PipelineKind::Distilled, &conditioning, &preset, 0);
        Ltx2GeneratePlan {
            pipeline: PipelineKind::Distilled,
            preset,
            execution_graph: graph,
            checkpoint_path: "/tmp/ltx2.safetensors".to_string(),
            distilled_checkpoint_path: None,
            distilled_lora_path: None,
            spatial_upsampler_path: None,
            gemma_root: "/tmp/gemma".to_string(),
            output_path: "/tmp/output.mp4".to_string(),
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone(),
            prompt_tokens: prompt_pair(),
            seed: 7,
            width: req.width,
            height: req.height,
            num_frames: req.frames.unwrap(),
            frame_rate: req.fps.unwrap(),
            num_inference_steps: req.steps,
            quantization: Some("fp8-cast".to_string()),
            streaming_prefetch_count: Some(2),
            conditioning,
            loras: vec![],
            retake_range: req.retake_range.clone(),
        }
    }

    #[test]
    fn runtime_prepare_tracks_audio_and_video_latent_shapes() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.frames, 97);
        assert_eq!(
            prepared.video_latent_shape.channels,
            LTX2_VIDEO_LATENT_CHANNELS
        );
        assert_eq!(prepared.video_latent_shape.frames, 13);
        assert_eq!(
            prepared.video_positions.dims4().unwrap(),
            (1, 3, 13 * 22 * 38, 2)
        );
        assert_eq!(
            prepared.audio_latent_shape.unwrap().channels,
            LTX2_AUDIO_LATENT_CHANNELS
        );
        assert!(prepared.audio_positions.is_some());
        assert!(prepared.cross_modal_temporal_positions.is_some());
        assert_eq!(
            prepared.prompt.conditional.video_encoding.dims3().unwrap(),
            (1, 3, 8)
        );
    }

    #[test]
    fn runtime_prepare_skips_audio_latents_for_silent_outputs() {
        let req = req("ltx-2-19b-distilled:fp8", OutputFormat::Gif, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert!(prepared.audio_latent_shape.is_none());
        assert!(prepared.audio_positions.is_none());
        assert!(prepared.cross_modal_temporal_positions.is_none());
    }

    #[test]
    fn runtime_prepare_derives_retake_mask_from_request_range() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.retake_range = Some(TimeRange {
            start_seconds: 1.0,
            end_seconds: 2.25,
        });
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();
        let mask = prepared.retake_mask.unwrap();

        assert_eq!(mask.len(), 97);
        assert!(mask[..24].iter().all(|value| *value == 0.0));
        assert!(mask[24..54].iter().all(|value| *value == 1.0));
        assert!(mask[54..].iter().all(|value| *value == 0.0));
    }
}
