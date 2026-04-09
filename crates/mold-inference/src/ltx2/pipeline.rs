use anyhow::{anyhow, bail, Context, Result};
use mold_core::{
    GenerateRequest, GenerateResponse, Ltx2PipelineMode, ModelPaths, OutputFormat, VideoData,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use super::assets;
use super::backend::Ltx2Backend;
use super::conditioning;
use super::lora;
use super::media::{self, ProbeMetadata};
use super::plan::{Ltx2GeneratePlan, PipelineKind};
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::progress::ProgressCallback;

pub struct Ltx2Engine {
    model_name: String,
    paths: ModelPaths,
    loaded: bool,
    on_progress: Option<ProgressCallback>,
}

impl Ltx2Engine {
    pub fn new(model_name: String, paths: ModelPaths, _load_strategy: LoadStrategy) -> Self {
        Self {
            model_name,
            paths,
            loaded: false,
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

    fn search_roots() -> Vec<PathBuf> {
        let mut roots = Vec::new();
        if let Ok(cwd) = std::env::current_dir() {
            roots.extend(cwd.ancestors().map(Path::to_path_buf));
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                roots.extend(parent.ancestors().map(Path::to_path_buf));
            }
        }
        roots.sort();
        roots.dedup();
        roots
    }

    fn find_runtime_path(relatives: &[&str]) -> Option<PathBuf> {
        for root in Self::search_roots() {
            for relative in relatives {
                let candidate = root.join(relative);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
        None
    }

    fn upstream_root(&self) -> Result<PathBuf> {
        if let Some(path) = std::env::var_os("MOLD_LTX2_UPSTREAM_ROOT").map(PathBuf::from) {
            return Ok(path);
        }

        Self::find_runtime_path(&["tmp/LTX-2-upstream"]).ok_or_else(|| {
            anyhow!(
                "could not locate the LTX-2 upstream checkout; set MOLD_LTX2_UPSTREAM_ROOT or clone it into tmp/LTX-2-upstream"
            )
        })
    }

    fn bridge_script(&self) -> Result<PathBuf> {
        if let Some(path) = std::env::var_os("MOLD_LTX2_BRIDGE_PATH").map(PathBuf::from) {
            return Ok(path);
        }

        Self::find_runtime_path(&["scripts/ltx2_bridge.py", "share/mold/ltx2_bridge.py"])
            .ok_or_else(|| {
                anyhow!(
                    "could not locate ltx2_bridge.py; set MOLD_LTX2_BRIDGE_PATH or run from a checkout/package that includes the bridge script"
                )
            })
    }

    fn venv_python(upstream_root: &Path) -> PathBuf {
        upstream_root.join(".venv/bin/python")
    }

    fn python_env_ready(upstream_root: &Path) -> bool {
        let venv_python = Self::venv_python(upstream_root);
        if !venv_python.is_file() {
            return false;
        }

        Command::new(&venv_python)
            .arg("-c")
            .arg("import ltx_pipelines")
            .current_dir(upstream_root)
            .status()
            .is_ok_and(|status| status.success())
    }

    fn ensure_python_env(&self) -> Result<()> {
        let upstream_root = self.upstream_root()?;
        if !upstream_root.join("pyproject.toml").exists() {
            bail!(
                "LTX-2 upstream root '{}' does not look valid (missing pyproject.toml)",
                upstream_root.display()
            );
        }
        if Self::python_env_ready(&upstream_root) {
            return Ok(());
        }

        let status = Command::new("uv")
            .arg("sync")
            .arg("--frozen")
            .arg("--project")
            .arg(&upstream_root)
            .status()
            .context("failed to run 'uv sync' for LTX-2 upstream")?;
        if !status.success() {
            bail!("uv sync failed for {}", upstream_root.display());
        }
        if !Self::python_env_ready(&upstream_root) {
            bail!(
                "LTX-2 Python environment is still not importable after uv sync in {}",
                upstream_root.display()
            );
        }
        Ok(())
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

    fn materialize_request(
        &self,
        req: &GenerateRequest,
        work_dir: &Path,
        output_path: &Path,
    ) -> Result<Ltx2GeneratePlan> {
        let pipeline = self.select_pipeline(req)?;
        if req.temporal_upscale.is_some() {
            bail!("temporal upscaling is not implemented in the current upstream LTX-2 bridge");
        }
        let conditioning = conditioning::stage_conditioning(req, work_dir)?;
        let loras = lora::resolve_loras(&self.model_name, req)?;
        let spatial_upsampler_path = assets::resolve_spatial_upscaler_path(
            &self.model_name,
            &self.paths,
            req.spatial_upscale,
        )?
        .map(|path| path.to_string_lossy().to_string());

        Ok(Ltx2GeneratePlan {
            pipeline,
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
            gemma_root: self.gemma_root()?.to_string_lossy().to_string(),
            output_path: output_path.to_string_lossy().to_string(),
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone(),
            seed: req.seed.unwrap_or_else(rand_seed),
            width: req.width,
            height: req.height,
            num_frames: req.frames.unwrap_or(97),
            frame_rate: req.fps.unwrap_or(24),
            num_inference_steps: req.steps,
            quantization: self.request_quantization(),
            streaming_prefetch_count: Some(2),
            conditioning,
            loras,
            retake_range: req.retake_range.clone(),
        })
    }

    fn run_bridge(&self, request_path: &Path) -> Result<()> {
        let upstream_root = self.upstream_root()?;
        let status = Command::new("uv")
            .arg("run")
            .arg("--project")
            .arg(&upstream_root)
            .arg("python")
            .arg(self.bridge_script()?)
            .arg("--request")
            .arg(request_path)
            .arg("--upstream-root")
            .arg(&upstream_root)
            .env("PYTORCH_ALLOC_CONF", "expandable_segments:True")
            .status()
            .context("failed to invoke LTX-2 bridge")?;
        if !status.success() {
            bail!("LTX-2 bridge process failed");
        }
        Ok(())
    }

    fn transcode_output(
        &self,
        input_mp4: &Path,
        output_format: OutputFormat,
        out_path: &Path,
    ) -> Result<()> {
        media::transcode_output(input_mp4, output_format, out_path)
    }

    fn strip_audio_track(&self, input_mp4: &Path, out_path: &Path) -> Result<()> {
        media::strip_audio_track(input_mp4, out_path)
    }

    fn extract_thumbnail(&self, input_video: &Path, output_png: &Path) -> Result<()> {
        media::extract_thumbnail(input_video, output_png)
    }

    fn extract_gif_preview(&self, input_video: &Path, output_gif: &Path) -> Result<()> {
        media::extract_gif_preview(input_video, output_gif)
    }

    fn probe_video(&self, input_video: &Path) -> Result<ProbeMetadata> {
        media::probe_video(input_video)
    }
}

impl InferenceEngine for Ltx2Engine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if !self.loaded {
            self.load()?;
        }
        let start = Instant::now();
        let wants_audio = req
            .enable_audio
            .unwrap_or(req.output_format == OutputFormat::Mp4);
        self.emit("Preparing LTX-2 request");

        let work_dir = tempfile::tempdir().context("failed to create LTX-2 temp directory")?;
        let upstream_output = work_dir.path().join("ltx2-output.mp4");
        let plan = self.materialize_request(req, work_dir.path(), &upstream_output)?;
        let bridge_request = plan.to_bridge_request();
        let bridge_request_path = work_dir.path().join("request.json");
        fs::write(
            &bridge_request_path,
            serde_json::to_vec_pretty(&bridge_request)?,
        )?;

        self.emit("Running upstream LTX-2 pipeline");
        self.run_bridge(&bridge_request_path)?;

        let working_mp4 = if wants_audio {
            upstream_output.clone()
        } else {
            let silent_output = work_dir.path().join("ltx2-output-silent.mp4");
            self.strip_audio_track(&upstream_output, &silent_output)?;
            silent_output
        };

        let final_output = work_dir
            .path()
            .join(format!("final.{}", req.output_format.extension()));
        self.emit("Transcoding output");
        self.transcode_output(&working_mp4, req.output_format, &final_output)?;

        let thumbnail_path = work_dir.path().join("thumbnail.png");
        self.extract_thumbnail(&working_mp4, &thumbnail_path)?;
        let gif_preview = if req.gif_preview {
            let gif_path = work_dir.path().join("preview.gif");
            self.extract_gif_preview(&working_mp4, &gif_path)?;
            fs::read(gif_path)?
        } else {
            Vec::new()
        };
        let final_bytes = fs::read(&final_output)?;
        let thumbnail_bytes = fs::read(thumbnail_path)?;
        let probe = self.probe_video(&working_mp4)?;
        let seed = plan.seed;

        Ok(GenerateResponse {
            images: vec![],
            video: Some(VideoData {
                data: final_bytes,
                format: req.output_format,
                width: probe.width.max(req.width),
                height: probe.height.max(req.height),
                frames: probe.frames.unwrap_or_else(|| req.frames.unwrap_or(97)),
                fps: probe.fps.max(req.fps.unwrap_or(24)),
                thumbnail: thumbnail_bytes,
                gif_preview,
                has_audio: probe.has_audio && req.output_format == OutputFormat::Mp4,
                duration_ms: probe.duration_ms,
                audio_sample_rate: if req.output_format == OutputFormat::Mp4 {
                    probe.audio_sample_rate
                } else {
                    None
                },
                audio_channels: if req.output_format == OutputFormat::Mp4 {
                    probe.audio_channels
                } else {
                    None
                },
            }),
            generation_time_ms: start.elapsed().as_millis() as u64,
            model: self.model_name.clone(),
            seed_used: seed,
        })
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn load(&mut self) -> Result<()> {
        self.emit("Preparing Python LTX-2 environment");
        if !self.paths.transformer.exists() {
            bail!(
                "missing LTX-2 checkpoint: {}",
                self.paths.transformer.display()
            );
        }
        Ltx2Backend::detect().ensure_supported()?;
        self.ensure_python_env()?;
        self.loaded = true;
        Ok(())
    }

    fn unload(&mut self) {
        self.loaded = false;
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
    use std::path::PathBuf;

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
            text_encoder_files: vec![PathBuf::from("/tmp/gemma/tokenizer.model")],
            text_tokenizer: None,
            decoder: None,
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
    fn materialized_request_uses_streaming_defaults_for_fp8_smoke_path() {
        let engine = Ltx2Engine::new(
            "ltx-2-19b-distilled:fp8".to_string(),
            dummy_paths(),
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
    }
}
