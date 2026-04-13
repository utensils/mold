#![allow(clippy::type_complexity)]

use anyhow::{bail, Context, Result};
use candle_core::Device;
use mold_core::{
    GenerateRequest, GenerateResponse, Ltx2PipelineMode, ModelPaths, OutputFormat, VideoData,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::assets;
use super::backend::Ltx2Backend;
use super::conditioning;
use super::execution;
use super::lora;
use super::media::{self, ProbeMetadata};
use super::plan::{Ltx2GeneratePlan, PipelineKind};
use super::preset;
use super::runtime::{Ltx2RuntimeSession, NativeRenderedVideo};
use super::text::gemma::GemmaAssets;
use super::text::prompt_encoder::NativePromptEncoder;
use crate::engine::{gpu_dtype, rand_seed, InferenceEngine, LoadStrategy};
use crate::ltx_video::video_enc;
use crate::progress::ProgressCallback;

pub struct Ltx2Engine {
    model_name: String,
    paths: ModelPaths,
    loaded: bool,
    native_runtime: Option<Ltx2RuntimeSession>,
    on_progress: Option<ProgressCallback>,
}

impl Ltx2Engine {
    fn debug_force_cpu_prompt_encoder() -> bool {
        std::env::var_os("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER").is_some()
    }

    fn debug_timings_enabled() -> bool {
        std::env::var_os("MOLD_LTX2_DEBUG_TIMINGS").is_some()
    }

    fn log_timing(label: &str, start: Instant) {
        if Self::debug_timings_enabled() {
            eprintln!(
                "[ltx2-timing] {label} {:.3}s",
                start.elapsed().as_secs_f64()
            );
        }
    }

    pub fn new(model_name: String, paths: ModelPaths, _load_strategy: LoadStrategy) -> Self {
        Self {
            model_name,
            paths,
            loaded: false,
            native_runtime: None,
            on_progress: None,
        }
    }

    #[cfg(test)]
    fn with_runtime_session(
        model_name: String,
        paths: ModelPaths,
        runtime: Ltx2RuntimeSession,
    ) -> Self {
        Self {
            model_name,
            paths,
            loaded: false,
            native_runtime: Some(runtime),
            on_progress: None,
        }
    }

    fn emit(&self, stage: &str) {
        if let Some(callback) = &self.on_progress {
            callback(crate::ProgressEvent::StageStart {
                name: stage.to_string(),
            });
        }
    }

    fn info(&self, message: &str) {
        if let Some(callback) = &self.on_progress {
            callback(crate::ProgressEvent::Info {
                message: message.to_string(),
            });
        }
    }

    fn is_oom_error(err: &impl std::fmt::Display) -> bool {
        let msg = err.to_string().to_ascii_lowercase();
        msg.contains("out of memory")
            || msg.contains("out_of_memory")
            || msg.contains("cudaerrormemoryallocation")
    }

    fn gemma_root(&self) -> Result<PathBuf> {
        assets::gemma_root(&self.paths)
    }

    fn select_pipeline(&self, req: &GenerateRequest) -> Result<PipelineKind> {
        if let Some(mode) = req.pipeline {
            return Ok(match mode {
                Ltx2PipelineMode::OneStage => PipelineKind::OneStage,
                Ltx2PipelineMode::TwoStage => PipelineKind::TwoStage,
                Ltx2PipelineMode::TwoStageHq => PipelineKind::TwoStageHq,
                Ltx2PipelineMode::Distilled => PipelineKind::Distilled,
                Ltx2PipelineMode::IcLora => PipelineKind::IcLora,
                Ltx2PipelineMode::Keyframe => PipelineKind::Keyframe,
                Ltx2PipelineMode::A2Vid => PipelineKind::A2Vid,
                Ltx2PipelineMode::Retake => PipelineKind::Retake,
            });
        }

        if req.retake_range.is_some() {
            return Ok(PipelineKind::Retake);
        }
        if req.audio_file.is_some() {
            return Ok(PipelineKind::A2Vid);
        }
        if req.keyframes.as_ref().is_some_and(|items| items.len() > 1) {
            return Ok(PipelineKind::Keyframe);
        }
        if req.source_video.is_some() {
            return Ok(PipelineKind::IcLora);
        }
        if self.model_name.contains("distilled") {
            return Ok(PipelineKind::Distilled);
        }
        Ok(PipelineKind::TwoStage)
    }

    fn request_quantization(&self) -> Option<String> {
        assets::request_quantization(&self.model_name)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn camera_control_preset(name: &str) -> Option<lora::CameraControlPreset> {
        lora::camera_control_preset(name)
    }

    pub(crate) fn materialize_request(
        &self,
        req: &GenerateRequest,
        work_dir: &Path,
        output_path: &Path,
    ) -> Result<Ltx2GeneratePlan> {
        let pipeline = self.select_pipeline(req)?;
        let gemma_root = self.gemma_root()?;
        let prompt_tokens = GemmaAssets::discover(&gemma_root)?
            .encode_prompt_pair(&req.prompt, req.negative_prompt.as_deref())?;
        let conditioning = conditioning::stage_conditioning(req, work_dir)?;
        let loras = lora::resolve_loras(&self.model_name, req)?;
        let preset = preset::preset_for_model(&self.model_name)?;
        let execution_graph =
            execution::build_execution_graph(req, pipeline, &conditioning, &preset, loras.len());
        let spatial_upsampler_path = assets::resolve_spatial_upscaler_path(
            &self.model_name,
            &self.paths,
            req.spatial_upscale,
        )?
        .map(|path| path.to_string_lossy().to_string());
        let temporal_upsampler_path =
            assets::resolve_temporal_upscaler_path(&self.paths, req.temporal_upscale)?
                .map(|path| path.to_string_lossy().to_string());

        Ok(Ltx2GeneratePlan {
            pipeline,
            preset,
            checkpoint_is_distilled: self.model_name.contains("distilled"),
            execution_graph,
            checkpoint_path: self.paths.transformer.to_string_lossy().to_string(),
            distilled_checkpoint_path: pipeline
                .requires_distilled_checkpoint()
                .then(|| self.paths.transformer.to_string_lossy().to_string()),
            distilled_lora_path: self
                .paths
                .distilled_lora
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            spatial_upsampler_path,
            temporal_upsampler_path,
            gemma_root: gemma_root.to_string_lossy().to_string(),
            output_path: output_path.to_string_lossy().to_string(),
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone(),
            prompt_tokens,
            seed: req.seed.unwrap_or_else(rand_seed),
            width: req.width,
            height: req.height,
            num_frames: req.frames.unwrap_or(97),
            frame_rate: req.fps.unwrap_or(24),
            num_inference_steps: req.steps,
            guidance: req.guidance,
            quantization: self.request_quantization(),
            streaming_prefetch_count: Some(preset.streaming_prefetch_count),
            conditioning,
            loras,
            retake_range: req.retake_range.clone(),
            spatial_upscale: req.spatial_upscale,
            temporal_upscale: req.temporal_upscale,
        })
    }

    fn probe_video(&self, input_video: &Path) -> Result<ProbeMetadata> {
        media::probe_video(input_video)
    }

    fn native_device_for_backend(&self, backend: Ltx2Backend) -> Result<Device> {
        match backend {
            Ltx2Backend::Cuda => {
                self.info("CUDA detected, using native LTX-2 GPU path");
                let device = Device::new_cuda(0)?;
                configure_native_ltx2_cuda_device(&device)?;
                Ok(device)
            }
            Ltx2Backend::Cpu => {
                let forced_cpu = std::env::var("MOLD_DEVICE")
                    .map(|value| value.eq_ignore_ascii_case("cpu"))
                    .unwrap_or(false);
                if forced_cpu {
                    self.info("CPU forced via MOLD_DEVICE=cpu for native LTX-2");
                } else {
                    self.info("No CUDA detected; using native LTX-2 CPU fallback");
                }
                Ok(Device::Cpu)
            }
            Ltx2Backend::Metal => unreachable!("unsupported Metal backend should have errored"),
        }
    }

    fn load_runtime_session_on_device(
        &self,
        plan: &Ltx2GeneratePlan,
        device: Device,
    ) -> Result<Ltx2RuntimeSession> {
        let prompt_device = if Self::debug_force_cpu_prompt_encoder() && !device.is_cpu() {
            Device::Cpu
        } else {
            device.clone()
        };
        let dtype = gpu_dtype(&prompt_device);
        self.emit("Loading native LTX-2 prompt encoder");
        let prompt_encoder = NativePromptEncoder::load(
            Path::new(&plan.gemma_root),
            Path::new(&plan.checkpoint_path),
            &plan.preset,
            &prompt_device,
            dtype,
        )?;
        if prompt_device.is_cuda() {
            Ok(Ltx2RuntimeSession::new_deferred_cuda(prompt_encoder))
        } else {
            Ok(Ltx2RuntimeSession::new(device, prompt_encoder))
        }
    }

    fn create_runtime_session(&self, plan: &Ltx2GeneratePlan) -> Result<Ltx2RuntimeSession> {
        let backend = Ltx2Backend::detect();
        backend.ensure_supported()?;

        match self.load_runtime_session_on_device(plan, self.native_device_for_backend(backend)?) {
            Ok(runtime) => Ok(runtime),
            Err(err) if matches!(backend, Ltx2Backend::Cuda) && Self::is_oom_error(&err) => {
                self.info(
                    "Native LTX-2 prompt path ran out of CUDA memory; retrying with CPU fallback",
                );
                crate::device::reclaim_gpu_memory();
                self.load_runtime_session_on_device(plan, Device::Cpu)
            }
            Err(err) => Err(err),
        }
    }

    fn encode_native_video(
        &self,
        req: &GenerateRequest,
        plan: &Ltx2GeneratePlan,
        rendered: &NativeRenderedVideo,
        work_dir: &Path,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Option<ProbeMetadata>)> {
        if let Some(audio_track) = rendered.audio_track.as_ref() {
            let wav_path = work_dir.join("native-audio.wav");
            fs::write(
                &wav_path,
                media::encode_wav_f32_interleaved(
                    &audio_track.interleaved_samples,
                    audio_track.sample_rate,
                    audio_track.channels,
                )?,
            )?;
        }

        let output_encode_start = Instant::now();
        let output_bytes = match req.output_format {
            OutputFormat::Apng => {
                let metadata = video_enc::VideoMetadata {
                    prompt: req.prompt.clone(),
                    model: self.model_name.clone(),
                    seed: plan.seed,
                    steps: req.steps,
                    guidance: req.guidance,
                    width: plan.width,
                    height: plan.height,
                    frames: plan.num_frames,
                    fps: plan.frame_rate,
                };
                video_enc::encode_apng(&rendered.frames, plan.frame_rate, Some(&metadata))?
            }
            OutputFormat::Gif => video_enc::encode_gif(&rendered.frames, plan.frame_rate)?,
            #[cfg(feature = "webp")]
            OutputFormat::Webp => video_enc::encode_webp(&rendered.frames, plan.frame_rate)?,
            #[cfg(not(feature = "webp"))]
            OutputFormat::Webp => bail!("WebP output requires the 'webp' feature"),
            OutputFormat::Mp4 => {
                #[cfg(feature = "mp4")]
                {
                    let video_only = video_enc::encode_mp4(&rendered.frames, plan.frame_rate)?;
                    let mp4_path = work_dir.join("native-video.mp4");
                    fs::write(&mp4_path, &video_only)?;
                    if let Some(audio_track) = rendered.audio_track.as_ref() {
                        let muxed_path = work_dir.join("native-video-audio.mp4");
                        media::attach_aac_track_from_f32_interleaved(
                            &mp4_path,
                            &muxed_path,
                            &audio_track.interleaved_samples,
                            audio_track.sample_rate,
                            audio_track.channels,
                        )?;
                        fs::read(muxed_path)?
                    } else {
                        video_only
                    }
                }
                #[cfg(not(feature = "mp4"))]
                {
                    bail!("MP4 output requires the 'mp4' feature")
                }
            }
            other => bail!("{other:?} is not supported for LTX-2 video output"),
        };
        Self::log_timing("pipeline.encode_output", output_encode_start);

        let thumbnail_start = Instant::now();
        let thumbnail = video_enc::first_frame_png(&rendered.frames)?;
        Self::log_timing("pipeline.encode_thumbnail", thumbnail_start);
        let gif_preview_start = Instant::now();
        let gif_preview = if req.gif_preview {
            if req.output_format == OutputFormat::Gif {
                output_bytes.clone()
            } else {
                video_enc::encode_gif(&rendered.frames, plan.frame_rate)?
            }
        } else {
            Vec::new()
        };
        Self::log_timing("pipeline.encode_gif_preview", gif_preview_start);

        let probe_start = Instant::now();
        let probe = if req.output_format == OutputFormat::Mp4 {
            let path = work_dir.join("probe.mp4");
            fs::write(&path, &output_bytes)?;
            Some(self.probe_video(&path)?)
        } else {
            None
        };
        Self::log_timing("pipeline.probe_output", probe_start);

        Ok((output_bytes, thumbnail, gif_preview, probe))
    }
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
fn configure_native_ltx2_cuda_device(device: &Device) -> Result<()> {
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        let cuda = device.as_cuda_device()?;
        if cuda.is_event_tracking() {
            // Native LTX-2 runs on a single dedicated stream. Disabling CUDA event
            // tracking avoids teardown crashes in cudarc/candle when large native
            // video runs drop many tensors at the end of the request.
            unsafe {
                cuda.disable_event_tracking();
            }
        }
    }
    Ok(())
}

impl InferenceEngine for Ltx2Engine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if !self.loaded {
            self.load()?;
        }
        let start = Instant::now();
        self.emit("Preparing native LTX-2 request");

        let work_dir = tempfile::tempdir().context("failed to create LTX-2 temp directory")?;
        let native_output = work_dir.path().join("ltx2-native-output.mp4");
        let materialize_start = Instant::now();
        let plan = self.materialize_request(req, work_dir.path(), &native_output)?;
        Self::log_timing("pipeline.materialize_request", materialize_start);
        let planned_stage_count = plan.execution_graph.denoise_passes.len();
        self.emit(&format!(
            "Planned native LTX-2 graph: preset={}, denoise_stages={}, blocks={}, prompt_tokens={}/{}",
            plan.preset.name,
            planned_stage_count,
            plan.execution_graph.blocks.len(),
            plan.prompt_tokens.conditional.valid_len(),
            plan.prompt_tokens.unconditional.valid_len()
        ));
        if self.native_runtime.is_none() {
            let create_runtime_start = Instant::now();
            self.native_runtime = Some(self.create_runtime_session(&plan)?);
            Self::log_timing("pipeline.create_runtime", create_runtime_start);
        }
        let mut runtime = self
            .native_runtime
            .take()
            .context("native LTX-2 runtime session was not initialized")?;

        self.emit("Encoding prompt and preparing native LTX-2 runtime state");
        let prepare_start = Instant::now();
        let prepared = runtime.prepare(&plan)?;
        Self::log_timing("pipeline.prepare_runtime", prepare_start);
        self.emit("Executing native LTX-2 runtime");
        let render_start = Instant::now();
        let rendered = runtime.render_native_video(&plan, &prepared)?;
        Self::log_timing("pipeline.render_runtime", render_start);
        let encode_start = Instant::now();
        let (output_bytes, thumbnail_bytes, gif_preview, probe) =
            self.encode_native_video(req, &plan, &rendered, work_dir.path())?;
        Self::log_timing("pipeline.encode_native_video", encode_start);
        self.native_runtime = Some(runtime);

        let duration_ms =
            Some((plan.num_frames as u64 * 1000).div_ceil(plan.frame_rate.max(1) as u64));
        let width = probe
            .as_ref()
            .map(|probe| probe.width)
            .unwrap_or(plan.width);
        let height = probe
            .as_ref()
            .map(|probe| probe.height)
            .unwrap_or(plan.height);
        let frames = probe
            .as_ref()
            .and_then(|probe| probe.frames)
            .unwrap_or(plan.num_frames);
        let fps = probe
            .as_ref()
            .map(|probe| probe.fps)
            .unwrap_or(plan.frame_rate);
        let has_audio = if req.output_format == OutputFormat::Mp4 {
            probe
                .as_ref()
                .map(|probe| probe.has_audio)
                .unwrap_or(rendered.has_audio)
        } else {
            false
        };
        let audio_sample_rate = if req.output_format == OutputFormat::Mp4 {
            probe
                .as_ref()
                .and_then(|probe| probe.audio_sample_rate)
                .or(rendered.audio_sample_rate)
        } else {
            None
        };
        let audio_channels = if req.output_format == OutputFormat::Mp4 {
            probe
                .as_ref()
                .and_then(|probe| probe.audio_channels)
                .or(rendered.audio_channels)
        } else {
            None
        };

        Ok(GenerateResponse {
            images: vec![],
            video: Some(VideoData {
                data: output_bytes,
                format: req.output_format,
                width,
                height,
                frames,
                fps,
                thumbnail: thumbnail_bytes,
                gif_preview,
                has_audio,
                duration_ms: probe
                    .as_ref()
                    .and_then(|probe| probe.duration_ms)
                    .or(duration_ms),
                audio_sample_rate,
                audio_channels,
            }),
            generation_time_ms: start.elapsed().as_millis() as u64,
            model: self.model_name.clone(),
            seed_used: plan.seed,
        })
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn load(&mut self) -> Result<()> {
        self.emit("Preparing native LTX-2 runtime");
        if !self.paths.transformer.exists() {
            bail!(
                "missing LTX-2 checkpoint: {}",
                self.paths.transformer.display()
            );
        }
        let gemma_root = self.gemma_root()?;
        if !gemma_root.join("tokenizer.json").exists() {
            bail!(
                "missing Gemma tokenizer assets for LTX-2: {}",
                gemma_root.display()
            );
        }
        Ltx2Backend::detect().ensure_supported()?;
        self.loaded = true;
        Ok(())
    }

    fn unload(&mut self) {
        self.loaded = false;
        self.native_runtime = None;
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.on_progress = Some(callback);
    }

    fn clear_on_progress(&mut self) {
        self.on_progress = None;
    }

    fn model_paths(&self) -> Option<&ModelPaths> {
        Some(&self.paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use crate::ltx2::text::connectors::PaddingSide;
    use crate::ltx2::text::encoder::{GemmaConfig, GemmaHiddenStateEncoder};
    use crate::ltx2::text::prompt_encoder::{
        build_embeddings_processor, ConnectorSpec, NativePromptEncoder,
    };

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

    fn dummy_paths_in(root: &Path, gemma_root: &Path) -> ModelPaths {
        ModelPaths {
            transformer: root.join("ltx2.safetensors"),
            transformer_shards: vec![],
            vae: root.join("unused"),
            spatial_upscaler: Some(root.join("spatial.safetensors")),
            temporal_upscaler: Some(root.join("temporal.safetensors")),
            distilled_lora: Some(root.join("distilled-lora.safetensors")),
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![gemma_root.join("tokenizer.json")],
            text_tokenizer: None,
            decoder: None,
        }
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
            max_position_embeddings: 1024,
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
        tensors.insert(
            "model.norm.weight".to_string(),
            Tensor::zeros(cfg.hidden_size, DType::F32, &Device::Cpu).unwrap(),
        );
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
            for norm_name in ["attn1.q_norm", "attn1.k_norm"] {
                tensors.insert(
                    format!("{prefix}.transformer_1d_blocks.0.{norm_name}.weight"),
                    Tensor::ones(dim, DType::F32, &Device::Cpu).unwrap(),
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
                    apply_gated_attention: false,
                    positional_embedding_theta: 10_000.0,
                    positional_embedding_max_pos: &[32],
                    rope_type: crate::ltx2::model::LtxRopeType::Split,
                    double_precision_rope: true,
                    num_learnable_registers: Some(128),
                },
                Some(ConnectorSpec {
                    prefix: "model.diffusion_model.audio_embeddings_connector.",
                    num_attention_heads: 1,
                    attention_head_dim: 4,
                    num_layers: 1,
                    apply_gated_attention: false,
                    positional_embedding_theta: 10_000.0,
                    positional_embedding_max_pos: &[32],
                    rope_type: crate::ltx2::model::LtxRopeType::Split,
                    double_precision_rope: true,
                    num_learnable_registers: Some(128),
                }),
            )
            .unwrap(),
            PaddingSide::Left,
        );
        Ltx2RuntimeSession::new(Device::Cpu, prompt_encoder)
    }

    fn request(output_format: OutputFormat, enable_audio: Option<bool>) -> GenerateRequest {
        GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "ltx-2-19b-distilled:fp8".to_string(),
            width: 960,
            height: 576,
            steps: 8,
            guidance: 3.0,
            seed: Some(42),
            batch_size: 1,
            output_format,
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
            frames: Some(17),
            fps: Some(12),
            upscale_model: None,
            gif_preview: true,
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

    #[test]
    fn pipeline_defaults_to_distilled_for_distilled_models() {
        let engine = Ltx2Engine::new(
            "ltx-2.3-22b-distilled:fp8".to_string(),
            dummy_paths(),
            LoadStrategy::Sequential,
        );
        let req = GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "ltx-2.3-22b-distilled:fp8".to_string(),
            width: 1216,
            height: 704,
            steps: 8,
            guidance: 1.0,
            seed: Some(1),
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
        };
        assert_eq!(
            engine.select_pipeline(&req).unwrap(),
            PipelineKind::Distilled
        );
    }

    #[test]
    fn camera_control_preset_aliases_are_supported() {
        let preset = Ltx2Engine::camera_control_preset("dolly-in").unwrap();
        assert_eq!(
            preset.filename,
            "ltx-2-19b-lora-camera-control-dolly-in.safetensors"
        );
        assert!(Ltx2Engine::camera_control_preset("unknown").is_none());
    }

    #[test]
    fn fp8_models_use_fp8_cast_quantization() {
        let engine = Ltx2Engine::new(
            "ltx-2-19b-distilled:fp8".to_string(),
            dummy_paths(),
            LoadStrategy::Sequential,
        );
        assert_eq!(engine.request_quantization(), Some("fp8-cast".to_string()));
    }

    #[test]
    fn oom_error_detection_matches_cuda_allocator_strings() {
        assert!(Ltx2Engine::is_oom_error(&"CUDA out of memory"));
        assert!(Ltx2Engine::is_oom_error(&"cudaErrorMemoryAllocation"));
        assert!(!Ltx2Engine::is_oom_error(&"some other error"));
    }

    #[test]
    fn materialized_request_uses_streaming_defaults_for_fp8_smoke_path() {
        let gemma_dir = tempfile::tempdir().unwrap();
        write_test_gemma_assets(gemma_dir.path());
        let engine = Ltx2Engine::new(
            "ltx-2-19b-distilled:fp8".to_string(),
            dummy_paths_with_gemma_root(gemma_dir.path()),
            LoadStrategy::Sequential,
        );
        let req = GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "ltx-2-19b-distilled:fp8".to_string(),
            width: 960,
            height: 576,
            steps: 8,
            guidance: 3.0,
            seed: Some(42),
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
            frames: Some(17),
            fps: Some(12),
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
        };
        let temp_dir = tempfile::tempdir().unwrap();
        let bridge = engine
            .materialize_request(&req, temp_dir.path(), &temp_dir.path().join("out.mp4"))
            .unwrap();
        assert_eq!(bridge.quantization.as_deref(), Some("fp8-cast"));
        assert_eq!(bridge.streaming_prefetch_count, Some(2));
        assert_eq!(bridge.width, 960);
        assert_eq!(bridge.height, 576);
        assert_eq!(bridge.num_frames, 17);
        assert_eq!(bridge.frame_rate, 12);
        assert_eq!(bridge.prompt_tokens.conditional.len(), 256);
        assert_eq!(bridge.prompt_tokens.conditional.valid_len(), 1);
        assert_eq!(bridge.prompt_tokens.pad_token_id, 7);
    }

    #[test]
    fn load_uses_native_asset_checks_without_upstream_checkout() {
        let temp_dir = tempfile::tempdir().unwrap();
        let gemma_dir = temp_dir.path().join("gemma");
        fs::create_dir_all(&gemma_dir).unwrap();
        write_test_gemma_assets(&gemma_dir);
        let paths = dummy_paths_in(temp_dir.path(), &gemma_dir);
        fs::write(&paths.transformer, []).unwrap();

        let mut engine = Ltx2Engine::new(
            "ltx-2-19b-distilled:fp8".to_string(),
            paths,
            LoadStrategy::Sequential,
        );

        engine.load().unwrap();
        assert!(engine.is_loaded());
    }

    #[test]
    fn generate_runs_native_runtime_without_bridge_process() {
        let temp_dir = tempfile::tempdir().unwrap();
        let gemma_dir = temp_dir.path().join("gemma");
        fs::create_dir_all(&gemma_dir).unwrap();
        write_test_gemma_assets(&gemma_dir);
        let paths = dummy_paths_in(temp_dir.path(), &gemma_dir);
        fs::write(&paths.transformer, []).unwrap();

        let mut engine = Ltx2Engine::with_runtime_session(
            "ltx-2-19b-distilled:fp8".to_string(),
            paths,
            runtime_session(),
        );
        let response = engine
            .generate(&request(OutputFormat::Gif, Some(false)))
            .unwrap();
        let video = response.video.unwrap();

        assert_eq!(&video.data[..6], b"GIF89a");
        assert_eq!(&video.thumbnail[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&video.gif_preview[..6], b"GIF89a");
        assert_eq!(video.width, 960);
        assert_eq!(video.height, 576);
        assert_eq!(video.frames, 17);
        assert_eq!(video.fps, 12);
        assert!(!video.has_audio);
    }
}
