use anyhow::{anyhow, bail, Context, Result};
use mold_core::{
    GenerateRequest, GenerateResponse, LoraWeight, Ltx2PipelineMode, ModelPaths, OutputFormat,
    VideoData,
};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::progress::ProgressCallback;

#[derive(Debug, Clone, Copy)]
struct CameraControlPreset {
    repo: &'static str,
    filename: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineKind {
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
    fn module_name(self) -> &'static str {
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

    fn requires_distilled_checkpoint(self) -> bool {
        matches!(self, Self::Distilled | Self::IcLora | Self::Retake)
    }
}

#[derive(Serialize)]
struct BridgeRequest {
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

#[derive(Default)]
struct ProbeMetadata {
    width: u32,
    height: u32,
    fps: u32,
    frames: Option<u32>,
    duration_ms: Option<u64>,
    has_audio: bool,
    audio_sample_rate: Option<u32>,
    audio_channels: Option<u32>,
}

pub struct Ltx2Engine {
    model_name: String,
    paths: ModelPaths,
    _load_strategy: LoadStrategy,
    loaded: bool,
    on_progress: Option<ProgressCallback>,
}

impl Ltx2Engine {
    pub fn new(model_name: String, paths: ModelPaths, load_strategy: LoadStrategy) -> Self {
        Self {
            model_name,
            paths,
            _load_strategy: load_strategy,
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

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn upstream_root(&self) -> PathBuf {
        Self::repo_root().join("tmp/LTX-2-upstream")
    }

    fn bridge_script(&self) -> PathBuf {
        Self::repo_root().join("scripts/ltx2_bridge.py")
    }

    fn ensure_python_env(&self) -> Result<()> {
        let upstream_root = self.upstream_root();
        let venv_python = upstream_root.join(".venv/bin/python");
        if venv_python.exists() {
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
        Ok(())
    }

    fn gemma_root(&self) -> Result<PathBuf> {
        self.paths
            .text_encoder_files
            .first()
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .ok_or_else(|| anyhow!("LTX-2 requires Gemma text encoder files to be available"))
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
        self.model_name
            .contains(":fp8")
            .then(|| "fp8-scaled-mm".to_string())
    }

    fn camera_control_preset(name: &str) -> Option<CameraControlPreset> {
        let normalized = name.trim().to_ascii_lowercase().replace('_', "-");
        match normalized.as_str() {
            "dolly-in" => Some(CameraControlPreset {
                repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
                filename: "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
            }),
            "dolly-left" => Some(CameraControlPreset {
                repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
                filename: "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
            }),
            "dolly-out" => Some(CameraControlPreset {
                repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
                filename: "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
            }),
            "dolly-right" => Some(CameraControlPreset {
                repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right",
                filename: "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
            }),
            "jib-down" => Some(CameraControlPreset {
                repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down",
                filename: "ltx-2-19b-lora-camera-control-jib-down.safetensors",
            }),
            "jib-up" => Some(CameraControlPreset {
                repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up",
                filename: "ltx-2-19b-lora-camera-control-jib-up.safetensors",
            }),
            "static" => Some(CameraControlPreset {
                repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static",
                filename: "ltx-2-19b-lora-camera-control-static.safetensors",
            }),
            _ => None,
        }
    }

    fn resolve_camera_control_preset(&self, name: &str) -> Result<PathBuf> {
        if self.model_name.contains("ltx-2.3") {
            bail!(
                "camera-control presets are currently published for LTX-2 19B only; pass an explicit .safetensors path for LTX-2.3"
            );
        }

        let preset = Self::camera_control_preset(name).ok_or_else(|| {
            anyhow!(
                "unknown camera-control preset '{name}' (expected one of: dolly-in, dolly-left, dolly-out, dolly-right, jib-down, jib-up, static)"
            )
        })?;

        mold_core::download::download_single_file_sync(
            preset.repo,
            preset.filename,
            Some("shared/ltx2-camera-control"),
        )
        .map_err(|err| anyhow!("failed to download camera-control preset '{name}': {err}"))
    }

    fn stage_input_file(dir: &Path, stem: &str, data: &[u8], default_ext: &str) -> Result<PathBuf> {
        let ext = if data.starts_with(&[0x89, b'P', b'N', b'G']) {
            "png"
        } else if data.starts_with(&[0xFF, 0xD8]) {
            "jpg"
        } else {
            default_ext
        };
        let path = dir.join(format!("{stem}.{ext}"));
        fs::write(&path, data)?;
        Ok(path)
    }

    fn materialize_request(
        &self,
        req: &GenerateRequest,
        work_dir: &Path,
        output_path: &Path,
    ) -> Result<BridgeRequest> {
        let pipeline = self.select_pipeline(req)?;
        if req.temporal_upscale.is_some() {
            bail!("temporal upscaling is not implemented in the current upstream LTX-2 bridge");
        }
        if req.spatial_upscale.is_some()
            && self
                .paths
                .spatial_upscaler
                .as_ref()
                .is_some_and(|path| path.to_string_lossy().contains("x2"))
            && matches!(
                req.spatial_upscale,
                Some(mold_core::Ltx2SpatialUpscale::X1_5)
            )
        {
            bail!("x1.5 spatial upscaling is not wired to a downloaded asset yet");
        }

        let mut images = Vec::new();
        if let Some(source_image) = &req.source_image {
            let path = Self::stage_input_file(work_dir, "source-image", source_image, "png")?;
            images.push(BridgeImage {
                path: path.to_string_lossy().to_string(),
                frame: 0,
                strength: req.strength as f32,
            });
        }
        if let Some(keyframes) = &req.keyframes {
            for (index, keyframe) in keyframes.iter().enumerate() {
                let path = Self::stage_input_file(
                    work_dir,
                    &format!("keyframe-{index:02}"),
                    &keyframe.image,
                    "png",
                )?;
                images.push(BridgeImage {
                    path: path.to_string_lossy().to_string(),
                    frame: keyframe.frame,
                    strength: 1.0,
                });
            }
        }

        let audio_path = req
            .audio_file
            .as_ref()
            .map(|bytes| Self::stage_input_file(work_dir, "conditioning-audio", bytes, "wav"))
            .transpose()?
            .map(|path| path.to_string_lossy().to_string());

        let video_path = req
            .source_video
            .as_ref()
            .map(|bytes| Self::stage_input_file(work_dir, "source-video", bytes, "mp4"))
            .transpose()?
            .map(|path| path.to_string_lossy().to_string());

        let mut loras = req.loras.clone().unwrap_or_default();
        for lora in &mut loras {
            if let Some(name) = lora.path.strip_prefix("camera-control:") {
                let resolved = self.resolve_camera_control_preset(name)?;
                lora.path = resolved.to_string_lossy().to_string();
            }
        }

        Ok(BridgeRequest {
            module: pipeline.module_name().to_string(),
            checkpoint_path: self.paths.transformer.to_string_lossy().to_string(),
            distilled_checkpoint_path: pipeline
                .requires_distilled_checkpoint()
                .then(|| self.paths.transformer.to_string_lossy().to_string()),
            distilled_lora_path: self
                .paths
                .distilled_lora
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            spatial_upsampler_path: self
                .paths
                .spatial_upscaler
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
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
            images,
            loras,
            audio_path,
            video_path,
            retake_start_seconds: req.retake_range.as_ref().map(|range| range.start_seconds),
            retake_end_seconds: req.retake_range.as_ref().map(|range| range.end_seconds),
        })
    }

    fn run_bridge(&self, request_path: &Path) -> Result<()> {
        let upstream_root = self.upstream_root();
        let status = Command::new("uv")
            .arg("run")
            .arg("--project")
            .arg(&upstream_root)
            .arg("python")
            .arg(self.bridge_script())
            .arg("--request")
            .arg(request_path)
            .status()
            .context("failed to invoke LTX-2 bridge")?;
        if !status.success() {
            bail!("LTX-2 bridge process failed");
        }
        Ok(())
    }

    fn run_ffmpeg<I, S>(&self, args: I, context_message: &str) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<std::ffi::OsStr>,
    {
        let status = Command::new("ffmpeg")
            .args(args)
            .status()
            .with_context(|| format!("failed to run ffmpeg for {context_message}"))?;
        if !status.success() {
            bail!("ffmpeg failed while {context_message}");
        }
        Ok(())
    }

    fn transcode_output(
        &self,
        input_mp4: &Path,
        output_format: OutputFormat,
        out_path: &Path,
    ) -> Result<()> {
        match output_format {
            OutputFormat::Mp4 => {
                fs::copy(input_mp4, out_path)?;
            }
            OutputFormat::Gif => {
                self.run_ffmpeg(
                    [
                        "-y",
                        "-i",
                        input_mp4.to_string_lossy().as_ref(),
                        out_path.to_string_lossy().as_ref(),
                    ],
                    "encoding GIF",
                )?;
            }
            OutputFormat::Apng => {
                self.run_ffmpeg(
                    [
                        "-y",
                        "-i",
                        input_mp4.to_string_lossy().as_ref(),
                        "-plays",
                        "0",
                        out_path.to_string_lossy().as_ref(),
                    ],
                    "encoding APNG",
                )?;
            }
            OutputFormat::Webp => {
                self.run_ffmpeg(
                    [
                        "-y",
                        "-i",
                        input_mp4.to_string_lossy().as_ref(),
                        "-loop",
                        "0",
                        out_path.to_string_lossy().as_ref(),
                    ],
                    "encoding WebP",
                )?;
            }
            other => bail!("{other:?} is not supported for LTX-2 video output"),
        }
        Ok(())
    }

    fn strip_audio_track(&self, input_mp4: &Path, out_path: &Path) -> Result<()> {
        self.run_ffmpeg(
            [
                "-y",
                "-i",
                input_mp4.to_string_lossy().as_ref(),
                "-an",
                "-c:v",
                "copy",
                out_path.to_string_lossy().as_ref(),
            ],
            "stripping audio track",
        )
    }

    fn extract_thumbnail(&self, input_video: &Path, output_png: &Path) -> Result<()> {
        self.run_ffmpeg(
            [
                "-y",
                "-i",
                input_video.to_string_lossy().as_ref(),
                "-frames:v",
                "1",
                output_png.to_string_lossy().as_ref(),
            ],
            "extracting thumbnail",
        )
    }

    fn extract_gif_preview(&self, input_video: &Path, output_gif: &Path) -> Result<()> {
        self.run_ffmpeg(
            [
                "-y",
                "-i",
                input_video.to_string_lossy().as_ref(),
                output_gif.to_string_lossy().as_ref(),
            ],
            "encoding GIF preview",
        )
    }

    fn probe_video(&self, input_video: &Path) -> Result<ProbeMetadata> {
        let output = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                input_video.to_string_lossy().as_ref(),
            ])
            .output()
            .context("failed to run ffprobe")?;
        if !output.status.success() {
            bail!("ffprobe failed for {}", input_video.display());
        }
        let value: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        let mut metadata = ProbeMetadata::default();
        let streams = value["streams"].as_array().cloned().unwrap_or_default();
        for stream in streams {
            match stream["codec_type"].as_str() {
                Some("video") => {
                    metadata.width = stream["width"].as_u64().unwrap_or_default() as u32;
                    metadata.height = stream["height"].as_u64().unwrap_or_default() as u32;
                    metadata.frames = stream["nb_frames"]
                        .as_str()
                        .and_then(|value| value.parse().ok())
                        .or_else(|| {
                            stream["nb_read_frames"]
                                .as_str()
                                .and_then(|value| value.parse().ok())
                        });
                    metadata.fps = stream["r_frame_rate"]
                        .as_str()
                        .and_then(parse_ffprobe_fps)
                        .unwrap_or(24);
                }
                Some("audio") => {
                    metadata.has_audio = true;
                    metadata.audio_sample_rate = stream["sample_rate"]
                        .as_str()
                        .and_then(|value| value.parse().ok());
                    metadata.audio_channels = stream["channels"].as_u64().map(|value| value as u32);
                }
                _ => {}
            }
        }
        metadata.duration_ms = value["format"]["duration"]
            .as_str()
            .and_then(|value| value.parse::<f64>().ok())
            .map(|seconds| (seconds * 1000.0).round() as u64);
        Ok(metadata)
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
        let bridge_request = self.materialize_request(req, work_dir.path(), &upstream_output)?;
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
        let seed = bridge_request.seed;

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

fn parse_ffprobe_fps(value: &str) -> Option<u32> {
    let (num, den) = value.split_once('/')?;
    let num: f64 = num.parse().ok()?;
    let den: f64 = den.parse().ok()?;
    if den == 0.0 {
        return None;
    }
    Some((num / den).round() as u32)
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
    fn parse_ffprobe_fps_rounds_fraction() {
        assert_eq!(parse_ffprobe_fps("24/1"), Some(24));
        assert_eq!(parse_ffprobe_fps("30000/1001"), Some(30));
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
}
