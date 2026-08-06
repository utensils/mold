#![allow(clippy::type_complexity)]

use anyhow::{bail, Context, Result};
use candle_core::Device;
use mold_core::{
    AudioData, GenerateRequest, GenerateResponse, Ltx2PipelineMode, ModelPaths, OutputFormat,
    VideoData,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::assets;
use super::backend::Ltx2Backend;
use super::conditioning::{self, StagedLatent};
use super::execution;
use super::lora;
use super::media::{self, ProbeMetadata};
use super::plan::{Ltx2GeneratePlan, PipelineKind};
use super::preset;
use super::runtime::{Ltx2RuntimeSession, NativeRenderedVideo};
use super::text::gemma::GemmaAssets;
use super::text::prompt_encoder::NativePromptEncoder;

/// Locate the pre-computed text embeddings that ship beside an IC-LoRA
/// control's weights.
///
/// Upstream's HDR pipeline takes `--text-embeddings` as a required argument
/// and never encodes a prompt (`hdr_ic_lora.py:250-256`, `:383`), so when the
/// companion is present it is the authority. The server reserves the control
/// adapter as the first LoRA slot, and the companion is downloaded into that
/// same directory, so the weights path locates it.
///
/// Returns `None` — and the ordinary Gemma encode runs — when the control is
/// not one that ships embeddings, or when the file is not on disk.
fn resolve_scene_embeddings_path(
    req: &GenerateRequest,
    loras: &[mold_core::LoraWeight],
) -> Option<String> {
    let control = req.ic_lora_control.as_deref()?;
    let filename = mold_core::ltx2_control::LTX2_CONTROL_ADAPTERS
        .iter()
        .find(|adapter| adapter.id == control)?
        .scene_embeddings_filename()?;
    let weights = Path::new(&loras.first()?.path);
    let candidate = weights.parent()?.join(filename);
    candidate
        .is_file()
        .then(|| candidate.to_string_lossy().into_owned())
}
use crate::chain::{ChainStageRenderer, ChainTail, StageOutcome, StageProgressEvent};
use crate::engine::{gpu_dtype, rand_seed, InferenceEngine, LoadStrategy};
use crate::ltx_video::video_enc;
use crate::progress::{InferenceCancellationToken, ProgressCallback};

/// Soft-conditioning strength for the cross-stage identity anchor on chain
/// continuations. The denoise mask at the anchor token becomes
/// `1 - strength = 0.6`, so the denoiser blends ~60% generated / ~40%
/// reference on every step — a gentle pull toward the source image rather
/// than a hard pin (hard-pinning a single pixel frame past the motion tail
/// would make continuations feel like cuts back to the starting shot).
const CHAIN_SOFT_ANCHOR_STRENGTH: f32 = 0.4;

/// Waveform thumbnail raster for audio-only gallery tiles. 16:9 so an audio
/// print sits in the same grid cell as a video print without reflowing it.
const AUDIO_THUMBNAIL_WIDTH: u32 = 640;
const AUDIO_THUMBNAIL_HEIGHT: u32 = 360;

pub struct Ltx2Engine {
    model_name: String,
    paths: ModelPaths,
    loaded: bool,
    native_runtime: Option<Ltx2RuntimeSession>,
    on_progress: Option<ProgressCallback>,
    cancellation: Option<InferenceCancellationToken>,
    pending_placement: Option<mold_core::types::DevicePlacement>,
    load_strategy: LoadStrategy,
    /// GPU ordinal this engine is pinned to. Every CUDA device operation must
    /// use this ordinal; hardcoding `0` can target a sibling worker's context.
    gpu_ordinal: usize,
    /// Optional preset hint used when the model name doesn't carry a
    /// recognisable family substring (`ltx-2.3`, `ltx-2`). Populated by
    /// `from_single_file` from the safetensors `__metadata__.model_version`
    /// so catalog (`cv:*` / `hf:*`) IDs select the right preset without
    /// requiring renames.
    preset_hint: Option<String>,
    /// Separate LTX-2.3 Gemma hidden-state projection used by diffusion-only
    /// and quantized checkpoints. Combined checkpoints leave this unset.
    text_projection_path: Option<PathBuf>,
    gemma_variant: Option<String>,
}

/// Reject an audio-wanting request against a checkpoint set that cannot decode
/// audio.
///
/// `gap` is a closure and the `wants_audio_output` test comes first, so the
/// probe never runs for the common case. Computing it eagerly made every LTX-2
/// request pay a safetensors header parse plus a formatted diagnostic string
/// that only an audio request would ever read — and the detailed message now
/// exists only when someone is about to see it, which is exactly when it
/// should be detailed.
fn validate_audio_output_request(
    req: &GenerateRequest,
    gap: impl FnOnce() -> Option<String>,
) -> Result<()> {
    if !execution::wants_audio_output(req) {
        return Ok(());
    }
    let Some(gap) = gap() else {
        return Ok(());
    };
    anyhow::bail!(
        "LTX-2 audio output is unavailable for model '{}': the resolved checkpoint set is \
         missing {gap}. Set enable_audio=false and retry; this request was rejected before \
         generation starts.",
        req.model
    );
}

impl Ltx2Engine {
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

    pub fn new(
        model_name: String,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
    ) -> Self {
        Self::new_with_gemma_variant(
            model_name,
            paths,
            load_strategy,
            gpu_ordinal,
            crate::runtime_env::value("MOLD_LTX2_GEMMA_VARIANT"),
        )
    }

    pub fn new_with_gemma_variant(
        model_name: String,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        gemma_variant: Option<String>,
    ) -> Self {
        let text_projection_path = paths
            .text_encoder_files
            .iter()
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.to_ascii_lowercase().contains("text_projection"))
            })
            .cloned();
        Self {
            model_name,
            paths,
            loaded: false,
            native_runtime: None,
            on_progress: None,
            cancellation: None,
            pending_placement: None,
            load_strategy,
            gpu_ordinal,
            preset_hint: None,
            text_projection_path,
            gemma_variant,
        }
    }

    /// Construct an LTX-2 engine from a Civitai single-file safetensors
    /// checkpoint.
    ///
    /// LTX-2 combined checkpoints (the standard Lightricks format) bundle
    /// both the video transformer (`transformer_blocks.*`) and the VAE
    /// (`vae.*`) in a single file. The runtime always loads both from
    /// `paths.transformer`, so on the checkpoint side this is structurally
    /// identical to `new()`.
    ///
    /// Validates the transformer layout and resolves either the bundled
    /// `vae.*` weights or a separate VAE companion from `paths.vae`.
    ///
    /// `paths` is the full resolved companion graph (text_encoder_files,
    /// upscalers, distilled_lora, …). `transformer` and `vae` are
    /// overridden to point at the single checkpoint; everything else is
    /// preserved so the Gemma TE companion (Civitai catalog entries don't
    /// bundle the encoder) and any other resolved companions reach the
    /// runtime intact. Discarding `paths` here is what bit cv:* LTX-2
    /// loads in the catalog rollout — the runtime then bailed with `LTX-2 requires
    /// Gemma text encoder files to be available`.
    pub fn from_single_file(
        model_name: String,
        checkpoint: PathBuf,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
    ) -> anyhow::Result<Self> {
        Self::from_single_file_with_gemma_variant(
            model_name,
            checkpoint,
            paths,
            load_strategy,
            gpu_ordinal,
            crate::runtime_env::value("MOLD_LTX2_GEMMA_VARIANT"),
        )
    }

    pub fn from_single_file_with_gemma_variant(
        model_name: String,
        checkpoint: PathBuf,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        gemma_variant: Option<String>,
    ) -> anyhow::Result<Self> {
        if !checkpoint.exists() {
            anyhow::bail!(
                "single-file LTX-2 checkpoint not found: {}",
                checkpoint.display()
            );
        }

        let bundle = super::single_file::load(&checkpoint).map_err(|e| {
            anyhow::anyhow!(
                "failed to parse single-file LTX-2 checkpoint {}: {e}",
                checkpoint.display()
            )
        })?;

        let vae = if bundle.has_vae {
            PathBuf::default()
        } else {
            if paths.vae.as_os_str().is_empty() {
                anyhow::bail!(
                    "LTX-2 checkpoint {} contains no VAE weights (`vae.*` keys). \
                     Pull the matching LTX-2 VAE companion and retry.",
                    checkpoint.display()
                );
            }
            if !paths.vae.is_file() {
                anyhow::bail!(
                    "LTX-2 VAE companion not on disk: {}. Re-pull the catalog model to fetch it.",
                    paths.vae.display()
                );
            }
            paths.vae.clone()
        };

        let preset_hint = bundle.model_version.or_else(|| {
            let filename = checkpoint
                .file_name()
                .and_then(|name| name.to_str())?
                .to_ascii_lowercase();
            (filename.contains("ltx23") || filename.contains("ltx2.3")).then(|| "2.3.0".to_string())
        });

        let paths = ModelPaths {
            transformer: checkpoint,
            transformer_shards: Vec::new(),
            vae,
            ..paths
        };

        let mut engine = Self::new_with_gemma_variant(
            model_name,
            paths,
            load_strategy,
            gpu_ordinal,
            gemma_variant,
        );
        // Catalog (`cv:*`) IDs don't contain `ltx-2.3` / `ltx-2` substrings,
        // so `preset_for_model` would bail. The bundled `model_version`
        // from the safetensors `__metadata__` (e.g. `"2.3.0"`) is the
        // authoritative source — record it as a hint that
        // `materialize_request` consults via `preset_for_model_with_hint`.
        engine.preset_hint = preset_hint;
        if !bundle.has_text_projection
            && engine
                .preset_hint
                .as_deref()
                .is_some_and(|hint| hint.starts_with("2.3"))
            && engine.text_projection_path.is_none()
        {
            anyhow::bail!(
                "LTX-2.3 checkpoint contains no text embedding projection. \
                 Pull the ltx2.3-text-projection companion and retry."
            );
        }
        Ok(engine)
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
            cancellation: None,
            pending_placement: None,
            load_strategy: LoadStrategy::Sequential,
            gpu_ordinal: 0,
            preset_hint: None,
            text_projection_path: None,
            gemma_variant: None,
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

    fn checkpoint(&self) -> Result<()> {
        if let Some(token) = &self.cancellation {
            token.checkpoint()?;
        }
        Ok(())
    }

    fn is_oom_error(err: &impl std::fmt::Display) -> bool {
        let msg = err.to_string().to_ascii_lowercase();
        msg.contains("out of memory")
            || msg.contains("out_of_memory")
            || msg.contains("cudaerrormemoryallocation")
    }

    fn unload_runtime_state(&mut self) -> Option<usize> {
        self.loaded = false;
        let had_cuda_state = self
            .native_runtime
            .as_ref()
            .is_some_and(Ltx2RuntimeSession::has_cuda_state);
        self.native_runtime = None;
        had_cuda_state.then_some(self.gpu_ordinal)
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
                Ltx2PipelineMode::LipDub => PipelineKind::LipDub,
                Ltx2PipelineMode::T2a => PipelineKind::T2a,
            });
        }

        if req.retake_range.is_some() {
            return Ok(PipelineKind::Retake);
        }
        if req.audio_file.is_some() || req.audio_file_path.is_some() {
            return Ok(PipelineKind::A2Vid);
        }
        if req.keyframes.as_ref().is_some_and(|items| items.len() > 1) {
            return Ok(PipelineKind::Keyframe);
        }
        if req.source_video.is_some() || req.source_video_path.is_some() {
            return Ok(PipelineKind::IcLora);
        }
        if self.model_name.contains("distilled") {
            // Distilled checkpoints also require a spatial upsampler (single
            // upscale stage instead of two denoise passes); without one,
            // fall back to a plain one-stage denoise that runs the
            // transformer end-to-end on the requested resolution.
            return Ok(if self.paths.spatial_upscaler.is_some() {
                PipelineKind::Distilled
            } else {
                PipelineKind::OneStage
            });
        }
        // TwoStage runs an upscale-and-refine pass after stage 1 and bails
        // at runtime if `spatial_upscaler` isn't on disk. Single-file
        // catalog (`cv:*`) checkpoints don't ship the upsampler asset, so
        // fall back to OneStage when it's missing — the user gets a clean
        // single-pass video instead of a 422 several stages in.
        Ok(if self.paths.spatial_upscaler.is_some() {
            PipelineKind::TwoStage
        } else {
            PipelineKind::OneStage
        })
    }

    fn request_quantization(&self) -> Option<String> {
        assets::request_quantization(&self.model_name)
    }

    #[allow(dead_code)]
    fn camera_control_preset(
        name: &str,
    ) -> Option<&'static mold_core::ltx2_camera::Ltx2CameraControlPreset> {
        lora::camera_control_preset(name)
    }

    pub(crate) fn materialize_request(
        &self,
        req: &GenerateRequest,
        work_dir: &Path,
        output_path: &Path,
    ) -> Result<Ltx2GeneratePlan> {
        validate_audio_output_request(req, || super::audio_output_gap(&self.paths))?;
        let pipeline = self.select_pipeline(req)?;
        let gemma_root = self.gemma_root()?;
        let prompt_tokens = GemmaAssets::discover(&gemma_root)?
            .encode_prompt_pair(&req.prompt, req.negative_prompt.as_deref())?;
        let conditioning = conditioning::stage_conditioning(req, work_dir)?;
        let loras = lora::resolve_loras(&self.paths, req)?;
        let preset =
            preset::preset_for_model_with_hint(&self.model_name, self.preset_hint.as_deref())?;
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
        // Lip dub re-voices an existing clip, so its length and rate belong to
        // the reference video, not the request. The server pre-fills the same
        // numbers so validation and VRAM admission see the truth; deriving
        // them here from the same helper keeps a forced-local run — which
        // never passes through the server — on the identical timeline.
        let (num_frames, frame_rate) = match pipeline {
            PipelineKind::LipDub => {
                let reference = conditioning.video_path.as_deref().context(
                    "the LTX-2 lip-dub pipeline requires a reference video (source_video)",
                )?;
                let probe = media::probe_video(Path::new(reference)).with_context(|| {
                    format!("failed to read the lip-dub reference video '{reference}'")
                })?;
                let frames = probe.frames.with_context(|| {
                    format!("lip-dub reference video '{reference}' reports no frame count")
                })?;
                let timing = mold_core::validation::resolve_lip_dub_timing(
                    mold_core::validation::LipDubReference {
                        frames,
                        fps: probe.fps,
                        has_audio: probe.has_audio,
                    },
                    req.frames,
                    req.fps,
                )
                .map_err(anyhow::Error::msg)?;
                // The server says this too, but a forced-local run never sees
                // that path — and a silently retimed dub looks fine and is out
                // of sync, so it is worth saying twice.
                for warning in &timing.warnings {
                    self.info(warning);
                }
                (timing.frames, timing.fps)
            }
            _ => (req.frames.unwrap_or(97), req.fps.unwrap_or(24)),
        };

        Ok(Ltx2GeneratePlan {
            hdr_exr_dir: req.hdr_exr_dir.clone(),
            hdr_exr_full_float: req.hdr_exr_full_float,
            hdr_exr_window: None,
            reference_frame_offset: 0,
            scene_embeddings_path: resolve_scene_embeddings_path(req, &loras),
            pipeline,
            preset,
            checkpoint_is_distilled: self.model_name.contains("distilled"),
            execution_graph,
            checkpoint_path: self.paths.transformer.to_string_lossy().to_string(),
            vae_checkpoint_path: if self.paths.vae.as_os_str().is_empty()
                || self.paths.vae == self.paths.transformer
            {
                self.paths.transformer.to_string_lossy().to_string()
            } else {
                self.paths.vae.to_string_lossy().to_string()
            },
            vae_in_checkpoint: self.paths.vae.as_os_str().is_empty()
                || self.paths.vae == self.paths.transformer,
            text_projection_path: self
                .text_projection_path
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
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
            num_frames,
            frame_rate,
            num_inference_steps: req.steps,
            guidance: req.guidance,
            quantization: self.request_quantization(),
            streaming_prefetch_count: Some(preset.streaming_prefetch_count),
            conditioning,
            loras,
            retake_range: req.retake_range.clone(),
            spatial_upscale: req.spatial_upscale,
            temporal_upscale: req.temporal_upscale,
            guidance_overrides: req.guidance_overrides.clone(),
            // The scheduler's admitted peak, when a worker bound one for this
            // dispatch. `None` on the CLI / test paths keeps the legacy
            // free-VRAM-only sizing.
            vram_grant_bytes: crate::device::thread_vram_grant_bytes(),
        })
    }

    fn probe_video(&self, input_video: &Path) -> Result<ProbeMetadata> {
        media::probe_video(input_video)
    }

    fn native_device_for_backend(&self, backend: Ltx2Backend) -> Result<Device> {
        match backend {
            Ltx2Backend::Cuda => {
                self.info("CUDA detected, using native LTX-2 GPU path");
                let device = Device::new_cuda(self.gpu_ordinal)?;
                configure_native_ltx2_cuda_device(&device)?;
                Ok(device)
            }
            Ltx2Backend::Cpu => {
                let forced_cpu = crate::runtime_env::value("MOLD_DEVICE")
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

    fn load_runtime_session_with_devices(
        &self,
        plan: &Ltx2GeneratePlan,
        device: Device,
        prompt_device: Device,
    ) -> Result<Ltx2RuntimeSession> {
        let load_start = Instant::now();
        log_prompt_encoder_placement(&device, &prompt_device);
        let dtype = gpu_dtype(&prompt_device);
        self.emit("Loading native LTX-2 prompt encoder");
        let prompt_encoder = NativePromptEncoder::load(
            Path::new(&plan.gemma_root),
            Path::new(&plan.checkpoint_path),
            plan.text_projection_path.as_deref().map(Path::new),
            &plan.preset,
            &prompt_device,
            dtype,
            self.gemma_variant.as_deref(),
        )?;
        Self::log_timing("pipeline.create_runtime.load_prompt_encoder", load_start);
        // Cross-device case (transformer on CUDA, encoder on CPU/sibling GPU)
        // can't use the deferred-cuda path because the prompt encoder doesn't
        // need a CUDA stream sync at the transformer's ordinal. Fall back to
        // the synchronous path; encode-time `move_prompt_encoding_to_device`
        // handles the cross-device tensor copy.
        let same_device = device.same_device(&prompt_device);
        if prompt_device.is_cuda() && same_device {
            Ok(Ltx2RuntimeSession::new_deferred_cuda(
                prompt_encoder,
                self.gpu_ordinal,
            ))
        } else {
            Ok(Ltx2RuntimeSession::new(
                device,
                prompt_encoder,
                self.gpu_ordinal,
            ))
        }
    }

    fn runtime_device_refs(
        &self,
    ) -> (
        Option<mold_core::types::DeviceRef>,
        Option<mold_core::types::DeviceRef>,
    ) {
        ltx2_runtime_device_refs(self.pending_placement.as_ref())
    }

    fn create_runtime_session(&self, plan: &Ltx2GeneratePlan) -> Result<Ltx2RuntimeSession> {
        let backend = Ltx2Backend::detect();
        backend.ensure_supported()?;

        // The scheduler leases the transformer/VAE device independently from
        // Gemma. Never let a CPU text-encoder placement move image-to-video
        // conditioning or denoising off that leased accelerator.
        let (runtime_ref, prompt_ref) = self.runtime_device_refs();
        let device =
            crate::device::resolve_device(runtime_ref, || self.native_device_for_backend(backend))?;
        if device.is_cuda() {
            configure_native_ltx2_cuda_device(&device)?;
        }
        let prompt_device = crate::device::resolve_device(prompt_ref.clone(), || {
            Ok(resolve_prompt_encoder_device(&device, self.gpu_ordinal))
        })?;
        // Only auto CUDA placement should retry on OOM — if the user explicitly
        // pinned the encoder to a GPU, surface the OOM rather than silently
        // rewriting their request.
        let override_is_auto = matches!(prompt_ref, None | Some(mold_core::types::DeviceRef::Auto));
        let prompt_device_is_cpu = prompt_device.is_cpu();
        match self.load_runtime_session_with_devices(plan, device.clone(), prompt_device) {
            Ok(runtime) => Ok(runtime),
            Err(err)
                if matches!(backend, Ltx2Backend::Cuda)
                    && override_is_auto
                    && !prompt_device_is_cpu
                    && Self::is_oom_error(&err) =>
            {
                self.info(
                    "Native LTX-2 prompt encoder ran out of CUDA memory; retrying Gemma on CPU \
                     while keeping the transformer and VAE on CUDA",
                );
                let _ = crate::device::post_drop_free_vram_bytes(self.gpu_ordinal);
                let (transformer_device, prompt_placement) =
                    prompt_encoder_oom_retry_placement(&device);
                self.load_runtime_session_with_devices(
                    plan,
                    transformer_device,
                    prompt_placement.into_device(),
                )
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
        let output_bytes = match req.resolved_output_format() {
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
            if req.resolved_output_format() == OutputFormat::Gif {
                output_bytes.clone()
            } else {
                video_enc::encode_gif(&rendered.frames, plan.frame_rate)?
            }
        } else {
            Vec::new()
        };
        Self::log_timing("pipeline.encode_gif_preview", gif_preview_start);

        let probe_start = Instant::now();
        let probe = if req.resolved_output_format() == OutputFormat::Mp4 {
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

impl Ltx2Engine {
    /// Resolve the video an extend request continues into decoded RGB frames.
    ///
    /// Inline bytes are staged to `work_dir` because the decoder is
    /// file-backed; a server-local path is used as-is (the server has already
    /// resolved it against its allow roots).
    fn load_extend_source(
        req: &GenerateRequest,
        work_dir: &Path,
    ) -> Result<(Vec<image::RgbImage>, media::ProbeMetadata)> {
        let path = match (&req.extend_video, &req.extend_video_path) {
            (Some(bytes), _) => {
                conditioning::stage_input_file(work_dir, "extend-video", bytes, "mp4")?
            }
            (None, Some(path)) => PathBuf::from(path),
            (None, None) => bail!("extend requested without extend_video or extend_video_path"),
        };
        let (probe, frames) = media::decode_video_frames_from_path(&path).with_context(|| {
            format!("failed to decode the video to extend ({})", path.display())
        })?;
        if frames.is_empty() {
            bail!(
                "the video to extend ({}) decoded to zero frames",
                path.display()
            );
        }
        Ok((frames, probe))
    }

    /// Continue an existing video in one request.
    ///
    /// This is the chain motion-tail handoff with the carryover coming from a
    /// decoded file instead of a previous stage: the last
    /// `extend_overlap_frames` pixel frames are re-encoded through the video
    /// VAE as conditioning, the model renders `frames` frames whose leading
    /// overlap reproduces that tail, and the delivered output is the original
    /// followed by everything past the overlap.
    fn extend_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.checkpoint()?;
        if !self.loaded {
            self.load()?;
        }
        self.checkpoint()?;
        let start = Instant::now();
        self.emit("Preparing native LTX-2 continuation");

        let work_dir = tempfile::tempdir().context("failed to create LTX-2 temp directory")?;
        let native_output = work_dir.path().join("ltx2-native-output.mp4");
        let (source_frames, probe) = Self::load_extend_source(req, work_dir.path())?;

        let overlap = req.effective_extend_overlap_frames();
        if (source_frames.len() as u32) < overlap {
            bail!(
                "the video to extend has {} frames but the requested overlap is {overlap}; \
                 lower --extend-overlap or supply a longer clip",
                source_frames.len(),
            );
        }
        // Materialize the plan up front so the source is checked against the
        // shape the render will ACTUALLY use. `req.fps` is frequently unset —
        // the plan then supplies the model default — so validating `req`
        // directly would let a 30 fps clip be continued at 24 fps and
        // re-encoded at 24, silently retiming the footage we were handed.
        let mut plan = self.materialize_request(req, work_dir.path(), &native_output)?;

        // The stitched output is one video, so the continuation has to render
        // on the source's own lattice. Rejecting is better than silently
        // rescaling: a mid-video resolution change is always a surprise.
        if probe.width != plan.width || probe.height != plan.height {
            bail!(
                "the video to extend is {}x{} but this request renders {}x{}; \
                 continuations must render at the source's resolution",
                probe.width,
                probe.height,
                plan.width,
                plan.height,
            );
        }
        if probe.fps != plan.frame_rate {
            bail!(
                "the video to extend runs at {} fps but this request renders {} fps; \
                 continuations must render at the source's frame rate{}",
                probe.fps,
                plan.frame_rate,
                if req.fps.is_none() {
                    format!(" (pass --fps {} to match the source)", probe.fps)
                } else {
                    String::new()
                },
            );
        }

        let tail_start = source_frames.len() - overlap as usize;
        let carry = ChainTail {
            frames: overlap,
            tail_rgb_frames: source_frames[tail_start..].to_vec(),
        };

        let outcome = self.render_chain_stage(req, Some(&carry), overlap, None)?;
        self.checkpoint()?;

        let source_len = source_frames.len();
        let frames = stitch_extend_frames(source_frames, &outcome.frames, overlap)?;
        let appended = frames.len() - source_len;

        // `encode_native_video` reads the frame count off the plan, so describe
        // the *stitched* result rather than the clip the transformer rendered.
        plan.num_frames = frames.len() as u32;
        let rendered = NativeRenderedVideo {
            frames,
            hdr_frames_written: None,
            audio_track: outcome.audio,
            has_audio: false,
            audio_sample_rate: None,
            audio_channels: None,
        };
        let (output_bytes, thumbnail_bytes, gif_preview, out_probe) =
            self.encode_native_video(req, &plan, &rendered, work_dir.path())?;

        self.emit(&format!(
            "Extended {source_len} source frames with {appended} new frames ({} total)",
            rendered.frames.len(),
        ));

        let fps = out_probe
            .as_ref()
            .map(|probe| probe.fps)
            .unwrap_or(plan.frame_rate);
        Ok(GenerateResponse {
            audio: None,
            images: vec![],
            video: Some(VideoData {
                data: output_bytes,
                format: req.resolved_output_format(),
                width: plan.width,
                height: plan.height,
                frames: out_probe
                    .as_ref()
                    .and_then(|probe| probe.frames)
                    .unwrap_or(plan.num_frames),
                fps,
                pipeline: Some(plan.pipeline.wire_mode()),
                thumbnail: thumbnail_bytes,
                gif_preview,
                has_audio: false,
                duration_ms: out_probe
                    .as_ref()
                    .and_then(|probe| probe.duration_ms)
                    .or(Some(
                        (plan.num_frames as u64 * 1000).div_ceil(fps.max(1) as u64),
                    )),
                audio_sample_rate: None,
                audio_channels: None,
            }),
            generation_time_ms: start.elapsed().as_millis() as u64,
            model: self.model_name.clone(),
            seed_used: plan.seed,
            gpu: None,
        })
    }

    fn generate_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if req.is_extend() {
            return self.extend_inner(req);
        }
        if self.select_pipeline(req)?.is_audio_only() {
            return self.generate_audio_inner(req);
        }
        self.checkpoint()?;
        if !self.loaded {
            self.load()?;
        }
        self.checkpoint()?;
        let start = Instant::now();
        self.emit("Preparing native LTX-2 request");

        let work_dir = tempfile::tempdir().context("failed to create LTX-2 temp directory")?;
        let native_output = work_dir.path().join("ltx2-native-output.mp4");
        let materialize_start = Instant::now();
        let plan = self.materialize_request(req, work_dir.path(), &native_output)?;
        self.checkpoint()?;
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
        let create_runtime_start = Instant::now();
        // Reuse a persisted runtime only if it can serve this plan. An LTX-2
        // session consumes its prompt encoder on first `prepare()` (see
        // runtime.rs `prepare()` — the take+drop frees VRAM for the
        // transformer); a stale session left behind by a prior chain run
        // survives intact for same-prompt continuations via the session-
        // level encoding cache, but we must rebuild from scratch when the
        // prompt changes so `prepare()` doesn't error on a consumed encoder.
        let mut runtime = match self.native_runtime.take() {
            Some(runtime) if runtime.can_reuse_for(&plan) => runtime,
            _ => self.create_runtime_session(&plan)?,
        };
        Self::log_timing("pipeline.create_runtime", create_runtime_start);

        self.emit("Encoding prompt and preparing native LTX-2 runtime state");
        let prepare_start = Instant::now();
        let prepared = runtime.prepare_with_progress(&plan, self.on_progress.as_ref())?;
        self.checkpoint()?;
        Self::log_timing("pipeline.prepare_runtime", prepare_start);
        self.emit("Executing native LTX-2 runtime");
        let render_start = Instant::now();
        let rendered = runtime.render_native_video(
            &plan,
            &prepared,
            self.on_progress.as_ref(),
            self.cancellation.as_ref(),
        )?;
        self.checkpoint()?;
        Self::log_timing("pipeline.render_runtime", render_start);

        // The EXR sequence is a sidecar: the gallery artifact stays the
        // tonemapped video, because a frame sequence is many files and
        // gigabytes that the one-file-per-generation model cannot hold.
        // The frames were written during decode rather than buffered and
        // written here: each one is width*height*3 f32, so holding the clip
        // would cost 12 GB at LTX-2's 20-second 1080p ceiling.
        if let (Some(dir), Some(written)) =
            (plan.hdr_exr_dir.as_deref(), rendered.hdr_frames_written)
        {
            tracing::info!(
                target: "mold::ltx2",
                "wrote {written} EXR frame(s) to {dir}"
            );
        }

        let encode_start = Instant::now();
        let (output_bytes, thumbnail_bytes, gif_preview, probe) =
            self.encode_native_video(req, &plan, &rendered, work_dir.path())?;
        Self::log_timing("pipeline.encode_native_video", encode_start);
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
        let has_audio = if req.resolved_output_format() == OutputFormat::Mp4 {
            probe
                .as_ref()
                .map(|probe| probe.has_audio)
                .unwrap_or(rendered.has_audio)
        } else {
            false
        };
        let audio_sample_rate = if req.resolved_output_format() == OutputFormat::Mp4 {
            probe
                .as_ref()
                .and_then(|probe| probe.audio_sample_rate)
                .or(rendered.audio_sample_rate)
        } else {
            None
        };
        let audio_channels = if req.resolved_output_format() == OutputFormat::Mp4 {
            probe
                .as_ref()
                .and_then(|probe| probe.audio_channels)
                .or(rendered.audio_channels)
        } else {
            None
        };

        Ok(GenerateResponse {
            audio: None,
            images: vec![],
            video: Some(VideoData {
                data: output_bytes,
                format: req.resolved_output_format(),
                width,
                height,
                frames,
                fps,
                pipeline: Some(plan.pipeline.wire_mode()),
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
            gpu: None,
        })
    }

    /// Text-to-audio: render an audio-only artifact.
    ///
    /// Parallel to [`Self::generate_inner`] rather than a branch inside it —
    /// almost none of the video path applies (no frames, no video VAE, no
    /// container mux, no probe), and the parts that do are the four lines
    /// below.
    fn generate_audio_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.checkpoint()?;
        if !self.loaded {
            self.load()?;
        }
        self.checkpoint()?;
        let start = Instant::now();
        self.emit("Preparing native LTX-2 text-to-audio request");

        // Ahead of `materialize_request`, whose shared audio guard advises
        // "set enable_audio=false" — advice a text-to-audio request cannot
        // take, because audio is the only thing it produces.
        if let Some(gap) = super::audio_output_gap(&self.paths) {
            bail!(
                "LTX-2 text-to-audio is unavailable for model '{}': the resolved checkpoint set \
                 is missing {gap}. Choose a checkpoint that ships them; this request was \
                 rejected before generation starts.",
                req.model
            );
        }

        let work_dir = tempfile::tempdir().context("failed to create LTX-2 temp directory")?;
        let native_output = work_dir.path().join("ltx2-native-output.wav");
        let plan = self.materialize_request(req, work_dir.path(), &native_output)?;
        self.checkpoint()?;

        let mut runtime = match self.native_runtime.take() {
            Some(runtime) if runtime.can_reuse_for(&plan) => runtime,
            _ => self.create_runtime_session(&plan)?,
        };
        self.emit("Encoding prompt and preparing native LTX-2 runtime state");
        let prepared = match runtime.prepare_with_progress(&plan, self.on_progress.as_ref()) {
            Ok(prepared) => prepared,
            Err(err) => {
                self.native_runtime = Some(runtime);
                return Err(err);
            }
        };
        self.emit("Executing native LTX-2 text-to-audio runtime");
        let render_result = runtime.render_native_audio(
            &plan,
            &prepared,
            self.on_progress.as_ref(),
            self.cancellation.as_ref(),
        );
        self.native_runtime = Some(runtime);
        let track = render_result?;
        self.checkpoint()?;

        let channels = u32::from(track.channels.max(1));
        let frames = track.interleaved_samples.len() as u64 / u64::from(channels);
        let duration_ms = (frames * 1000).div_ceil(u64::from(track.sample_rate.max(1)));
        let data = media::encode_wav_i16_interleaved(
            &track.interleaved_samples,
            track.sample_rate,
            track.channels,
        )?;
        let thumbnail = media::render_waveform_thumbnail_png(
            &track.interleaved_samples,
            track.channels,
            AUDIO_THUMBNAIL_WIDTH,
            AUDIO_THUMBNAIL_HEIGHT,
        )?;
        Self::log_timing("pipeline.encode_native_audio", start);

        Ok(GenerateResponse {
            images: vec![],
            video: None,
            audio: Some(AudioData {
                data,
                format: OutputFormat::Wav,
                sample_rate: track.sample_rate,
                channels,
                duration_ms,
                thumbnail,
                thumbnail_width: AUDIO_THUMBNAIL_WIDTH,
                thumbnail_height: AUDIO_THUMBNAIL_HEIGHT,
            }),
            generation_time_ms: start.elapsed().as_millis() as u64,
            model: self.model_name.clone(),
            seed_used: plan.seed,
            gpu: None,
        })
    }

    /// Render a single chain stage, optionally conditioning on a carryover
    /// tail from the prior stage.
    ///
    /// `motion_tail_pixel_frames` is the number of pixel frames to narrow
    /// off the emitted latents for the *next* stage's carryover. `0`
    /// returns an error (nonsensical — use the regular single-clip path
    /// if no tail is wanted).
    ///
    /// Scope: single-stage and distilled LTX-2 pipelines. Multi-pass and
    /// specialized conditioning pipelines return an error up-front so the
    /// chain orchestrator fails fast.
    pub(crate) fn render_chain_stage(
        &mut self,
        req: &GenerateRequest,
        carry: Option<&ChainTail>,
        motion_tail_pixel_frames: u32,
        hdr_sidecar: Option<&crate::chain::StageSidecar>,
    ) -> Result<StageOutcome> {
        if let Some(token) = self.cancellation.as_ref() {
            token.checkpoint()?;
        }
        if motion_tail_pixel_frames == 0 {
            bail!("render_chain_stage: motion_tail_pixel_frames must be > 0");
        }
        if !self.loaded {
            self.load()?;
        }
        let start = Instant::now();
        self.emit("Preparing native LTX-2 chain stage");

        let pipeline = self.select_pipeline(req)?;
        // IC-LoRA joins the chain contract only under an orchestrator HDR
        // sidecar, which supplies the per-stage reference window the generic
        // chain carry cannot: the whole clip is a regrade of a shared SDR
        // reference, and each stage conditions on its own temporal slice.
        let ic_lora_hdr_stage = pipeline == PipelineKind::IcLora && hdr_sidecar.is_some();
        if !pipeline_supports_render_chain(pipeline) && !ic_lora_hdr_stage {
            bail!(
                "sequence clips render through the one-stage, distilled, two-stage, and \
                 two-stage-hq LTX-2 pipelines; {:?} conditions on inputs a clip carry cannot \
                 supply",
                pipeline,
            );
        }

        let work_dir = tempfile::tempdir().context("failed to create LTX-2 temp directory")?;
        let native_output = work_dir.path().join("ltx2-native-output.mp4");
        let mut plan = self.materialize_request(req, work_dir.path(), &native_output)?;
        apply_stage_sidecar_to_plan(&mut plan, hdr_sidecar);
        if let Some(token) = self.cancellation.as_ref() {
            token.checkpoint()?;
        }

        // Inject carryover RGB frames as a StagedLatent at frame 0. The
        // runtime VAE-encodes them fresh on the receiving side so every
        // resulting latent slot has correct causal/continuation semantics
        // in this clip's own time axis (see conditioning.rs StagedLatent
        // docstring + runtime.rs maybe_load_stage_video_conditioning).
        //
        // When the chain request carries a starting image (i2v flow), the
        // orchestrator passes it through on every stage. Stage 0 uses it
        // as the frame-0 i2v replacement — great. On continuations the
        // motion-tail pin owns frame 0, so we re-route any frame-0 staged
        // image to a non-zero frame with reduced "soft anchor" strength:
        // the image becomes a durable identity reference appended to the
        // token sequence (via the `VideoTokenAppendCondition` path in
        // `maybe_load_stage_video_conditioning`), giving the free-region
        // denoise a persistent cross-attention anchor for subject / scene
        // appearance without freezing any tokens. Without this anchor,
        // identity drift compounds stage-over-stage because each clip's
        // only long-range reference is its own drifted last-frame carry.
        if let Some(tail) = carry {
            if req.source_image.is_some() {
                tracing::warn!(
                    "smooth continuation received source_image; it will be repurposed as a soft \
                     identity anchor. Use transition: cut|fade to seed the stage with a fresh i2v."
                );
            }
            if tail.tail_rgb_frames.is_empty() {
                bail!(
                    "render_chain_stage: carry.tail_rgb_frames is empty; caller must provide at least one frame"
                );
            }

            // Re-route any frame-0 staged image into the soft-anchor
            // append slot. The anchor frame is the first pixel past the
            // motion-tail pin, so the reference token's RoPE sits exactly
            // where new content starts — cross-attention propagates
            // identity into the free region most directly from there.
            // `CHAIN_SOFT_ANCHOR_STRENGTH = 0.4` gives the denoise mask a
            // value of `1 - 0.4 = 0.6` at the anchor token, so the
            // denoiser blends ~60% generated / ~40% reference every step.
            let anchor_frame = motion_tail_pixel_frames;
            for image in plan.conditioning.images.iter_mut() {
                if image.frame == 0 {
                    image.frame = anchor_frame;
                    image.strength = CHAIN_SOFT_ANCHOR_STRENGTH;
                }
            }

            plan.conditioning.latents.push(StagedLatent {
                tail_rgb_frames: tail.tail_rgb_frames.clone(),
                frame: 0,
                strength: 1.0,
            });
        }

        // Reuse an existing runtime session if we have one AND it can
        // serve this plan. Between stages of a same-prompt chain the
        // session-level encoding cache handles the consumed-encoder
        // invariant; if the prompt shifts (or a stale session leaked in
        // from a prior run) we drop the runtime and rebuild so
        // `prepare()` doesn't error on a missing encoder.
        let mut runtime = match self.native_runtime.take() {
            Some(runtime) if runtime.can_reuse_for(&plan) => runtime,
            _ => self.create_runtime_session(&plan)?,
        };

        self.emit("Executing native LTX-2 chain stage runtime");
        if let Some(token) = self.cancellation.as_ref() {
            token.checkpoint()?;
        }
        let prepared = match runtime.prepare_with_progress(&plan, self.on_progress.as_ref()) {
            Ok(prepared) => prepared,
            Err(err) => {
                self.native_runtime = Some(runtime);
                return Err(err);
            }
        };
        let render_result = runtime.render_native_video(
            &plan,
            &prepared,
            self.on_progress.as_ref(),
            self.cancellation.as_ref(),
        );
        self.native_runtime = Some(runtime);
        let rendered = render_result?;
        if let Some(token) = self.cancellation.as_ref() {
            token.checkpoint()?;
        }

        let frames = rendered.frames;
        let audio = rendered.audio_track;
        let hdr_frames_written = rendered.hdr_frames_written;
        let tail_pixel_frames = motion_tail_pixel_frames as usize;
        if frames.len() < tail_pixel_frames {
            bail!(
                "LTX-2 render returned {} pixel frames but the chain caller requested a {}-frame tail; \
                 this is a pipeline wiring bug",
                frames.len(),
                motion_tail_pixel_frames,
            );
        }
        let tail_start = frames.len() - tail_pixel_frames;
        let tail_rgb_frames = frames[tail_start..].to_vec();

        let generation_time_ms = start.elapsed().as_millis() as u64;
        Self::log_timing("pipeline.render_chain_stage", start);

        Ok(StageOutcome {
            frames,
            tail: ChainTail {
                frames: motion_tail_pixel_frames,
                tail_rgb_frames,
            },
            audio,
            hdr_frames_written,
            generation_time_ms,
        })
    }
}

/// Join an extend request's source clip to its continuation.
///
/// The continuation's leading `overlap` frames are a re-render of the source
/// tail that conditioned it, so they are dropped rather than delivered twice.
/// Everything after that is genuinely new footage appended to the source.
pub(crate) fn stitch_extend_frames(
    mut source: Vec<image::RgbImage>,
    continuation: &[image::RgbImage],
    overlap: u32,
) -> Result<Vec<image::RgbImage>> {
    let overlap = overlap as usize;
    if continuation.len() <= overlap {
        bail!(
            "LTX-2 continuation returned {} frames but the {overlap}-frame overlap consumes all \
             of them, leaving nothing new to append; this is a pipeline wiring bug",
            continuation.len(),
        );
    }
    source.extend_from_slice(&continuation[overlap..]);
    Ok(source)
}

/// Stamp (or clear) a chain stage's HDR EXR target on its materialized plan.
///
/// The sidecar argument is the ONLY route to a plan-level EXR target for
/// stage renders. A request-borne `hdr_exr_dir` without a window is cleared
/// rather than honoured: a per-stage decode numbering its EXRs from zero
/// would collide with every other stage and disagree with the stitched
/// timeline — the latent extend-path misalignment issue #688 predicted.
/// The single-render path never comes through here, so its request-derived
/// target is untouched.
fn apply_stage_sidecar_to_plan(
    plan: &mut Ltx2GeneratePlan,
    hdr_sidecar: Option<&crate::chain::StageSidecar>,
) {
    match hdr_sidecar {
        Some(sidecar) => {
            plan.hdr_exr_dir = Some(sidecar.exr_dir.to_string_lossy().to_string());
            plan.hdr_exr_full_float = sidecar.full_float;
            plan.hdr_exr_window = Some(sidecar.window);
            plan.reference_frame_offset = sidecar.reference_frame_offset;
        }
        None => {
            plan.hdr_exr_dir = None;
            plan.hdr_exr_full_float = false;
            plan.hdr_exr_window = None;
        }
    }
}

/// Pipelines whose runtime honours the chain contract: a hard frame-0 token
/// replacement from the carry tail, re-encoded at each stage's own pixel grid.
///
/// Two-stage qualifies because `render_real_two_stage_av` already re-loads
/// conditioning at the stage-2 pixel shape — the same thing upstream's
/// `ti2vid_two_stages` does with its image conditionings — and because the
/// carry is decoded RGB, not a latent, so it survives stage 1's implicit x2
/// downsample without a shape mismatch.
fn pipeline_supports_render_chain(pipeline: PipelineKind) -> bool {
    matches!(
        pipeline,
        PipelineKind::OneStage
            | PipelineKind::Distilled
            | PipelineKind::TwoStage
            | PipelineKind::TwoStageHq
    )
}

impl ChainStageRenderer for Ltx2Engine {
    fn render_stage(
        &mut self,
        stage_req: &GenerateRequest,
        carry: Option<&ChainTail>,
        motion_tail_pixel_frames: u32,
        hdr_sidecar: Option<&crate::chain::StageSidecar>,
        _stage_progress: Option<&mut dyn FnMut(StageProgressEvent)>,
    ) -> Result<StageOutcome> {
        // `_stage_progress` is intentionally unused in v1: per-stage
        // denoise events flow through `self.on_progress` already. The server
        // route installs an on_progress callback that forwards
        // those events onto the chain SSE stream with `stage_idx` tagged
        // in. If the orchestrator later needs denoise-step events routed
        // through its own channel, we can plumb `stage_progress` into a
        // temporary ProgressCallback wrapper here.
        self.render_chain_stage(stage_req, carry, motion_tail_pixel_frames, hdr_sidecar)
    }
}

impl InferenceEngine for Ltx2Engine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.pending_placement = req.placement.clone();
        let result = self.generate_inner(req);
        self.pending_placement = None;
        result
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
        if let Some(ordinal) = self.unload_runtime_state() {
            let _ = crate::device::post_drop_free_vram_bytes(ordinal);
        }
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.on_progress = Some(callback);
    }

    fn clear_on_progress(&mut self) {
        self.on_progress = None;
    }

    fn set_cancellation_token(&mut self, token: InferenceCancellationToken) {
        self.cancellation = Some(token);
    }

    fn clear_cancellation_token(&mut self) {
        self.cancellation = None;
    }

    fn batch_execution_capability(&self) -> crate::BatchExecutionCapability {
        crate::batch_execution_capability_for_family("ltx2")
            .expect("production LTX-2 batch capability must be registered")
    }

    fn model_paths(&self) -> Option<&ModelPaths> {
        Some(&self.paths)
    }

    fn configured_load_strategy(&self) -> Option<LoadStrategy> {
        Some(self.load_strategy)
    }

    fn configured_block_offload(&self) -> Option<bool> {
        // LTX-2 uses its native adaptive streaming runtime; the generic
        // request-controlled block-offload flag is not part of its factory
        // contract.
        Some(false)
    }

    fn as_chain_renderer(&mut self) -> Option<&mut dyn crate::chain::ChainStageRenderer> {
        Some(self)
    }
}

fn ltx2_runtime_device_refs(
    placement: Option<&mold_core::types::DevicePlacement>,
) -> (
    Option<mold_core::types::DeviceRef>,
    Option<mold_core::types::DeviceRef>,
) {
    let runtime = placement
        .and_then(|placement| placement.advanced.as_ref())
        .map(|advanced| match &advanced.transformer {
            // LTX-2's transformer and VAE share one native runtime device.
            // Honor an explicit VAE pin when the transformer remains Auto;
            // a concrete transformer pin is the primary runtime authority.
            mold_core::types::DeviceRef::Auto => advanced.vae.clone(),
            transformer => transformer.clone(),
        });
    let prompt = placement.map(|placement| placement.text_encoders.clone());
    (runtime, prompt)
}

/// Resolve the device for the LTX-2 Gemma 3 12B prompt encoder given the
/// transformer's chosen device.
///
/// - Transformer on CPU/Metal: keep the encoder on the same device. CPU
///   means the user opted out of GPU end-to-end and Metal LTX-2 isn't
///   supported anyway (caller will have errored before this).
/// - Transformer on CUDA: defer to the auto-resolver in
///   [`crate::device::resolve_ltx2_gemma_placement`], which honors the
///   `MOLD_LTX2_GEMMA_DEVICE` override and walks active GPU → siblings →
///   CPU on a free-VRAM probe.
pub(crate) fn resolve_prompt_encoder_device(
    transformer_device: &Device,
    gpu_ordinal: usize,
) -> Device {
    if !transformer_device.is_cuda() {
        return transformer_device.clone();
    }
    crate::device::resolve_ltx2_gemma_placement(gpu_ordinal).into_device()
}

/// An OOM while constructing Gemma changes only the prompt-encoder placement.
/// Keep this policy hardware-independent so CPU-only CI can prove that the
/// transformer's already-selected CUDA device is preserved through the retry.
fn prompt_encoder_oom_retry_placement<T: Clone>(
    transformer_device: &T,
) -> (T, crate::device::LtxGemmaPlacement) {
    (
        transformer_device.clone(),
        crate::device::LtxGemmaPlacement::Cpu,
    )
}

fn log_prompt_encoder_placement(transformer_device: &Device, prompt_device: &Device) {
    if transformer_device.same_device(prompt_device) {
        return;
    }
    let label = if prompt_device.is_cpu() {
        "CPU".to_string()
    } else if prompt_device.is_cuda() {
        "GPU (sibling ordinal)".to_string()
    } else {
        "non-CUDA device".to_string()
    };
    tracing::info!(
        prompt_encoder_device = %label,
        "LTX-2 Gemma encoder placed off the transformer device — \
         encode-time tensor copy will move conditioning back to the transformer GPU"
    );
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

    fn solid_frame(value: u8) -> image::RgbImage {
        image::RgbImage::from_pixel(4, 4, image::Rgb([value, value, value]))
    }

    fn frame_values(frames: &[image::RgbImage]) -> Vec<u8> {
        frames
            .iter()
            .map(|frame| frame.get_pixel(0, 0)[0])
            .collect()
    }

    /// The continuation's leading overlap re-renders the source tail that
    /// conditioned it. Delivering those frames would visibly stutter the seam,
    /// so exactly `overlap` frames are dropped — no more, no fewer.
    #[test]
    fn stitch_extend_drops_exactly_the_overlap() {
        let source: Vec<_> = (1..=5).map(solid_frame).collect();
        // The continuation reproduces frames 4 and 5, then adds 6, 7, 8.
        let continuation: Vec<_> = [4u8, 5, 6, 7, 8].into_iter().map(solid_frame).collect();

        let stitched = stitch_extend_frames(source, &continuation, 2).unwrap();
        assert_eq!(frame_values(&stitched), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn stitch_extend_keeps_the_source_when_only_one_frame_is_new() {
        let source: Vec<_> = (1..=3).map(solid_frame).collect();
        let continuation: Vec<_> = [3u8, 4].into_iter().map(solid_frame).collect();

        let stitched = stitch_extend_frames(source, &continuation, 1).unwrap();
        assert_eq!(frame_values(&stitched), vec![1, 2, 3, 4]);
    }

    /// A continuation no longer than its overlap would append nothing, which
    /// means the render was wired up wrong — fail loudly instead of silently
    /// returning the untouched source as though the extend had succeeded.
    #[test]
    fn stitch_extend_rejects_a_continuation_that_adds_nothing() {
        let source: Vec<_> = (1..=3).map(solid_frame).collect();
        let continuation: Vec<_> = [3u8, 4].into_iter().map(solid_frame).collect();

        let err = stitch_extend_frames(source.clone(), &continuation, 2).unwrap_err();
        assert!(format!("{err}").contains("nothing new"), "got: {err}");

        let err = stitch_extend_frames(source, &continuation, 5).unwrap_err();
        assert!(format!("{err}").contains("nothing new"), "got: {err}");
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

    fn runtime_prompt_encoder() -> NativePromptEncoder {
        let cfg = tiny_gemma_config();
        let gemma = GemmaHiddenStateEncoder::new(&cfg, zero_gemma_var_builder(&cfg)).unwrap();
        NativePromptEncoder::new(
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
        )
    }

    fn runtime_session() -> Ltx2RuntimeSession {
        let prompt_encoder = runtime_prompt_encoder();
        Ltx2RuntimeSession::new(Device::Cpu, prompt_encoder, 0)
    }

    fn request(output_format: OutputFormat, enable_audio: Option<bool>) -> GenerateRequest {
        GenerateRequest {
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "ltx-2-19b-distilled:fp8".to_string(),
            width: 960,
            height: 576,
            steps: 8,
            guidance: 3.0,
            seed: Some(42),
            batch_size: 1,
            output_format: Some(output_format),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
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
            frames: Some(17),
            fps: Some(12),
            upscale_model: None,
            gif_preview: true,
            enable_audio,
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
        }
    }

    #[test]
    fn pipeline_falls_back_to_one_stage_when_spatial_upscaler_missing() {
        // Catalog (`cv:*`) LTX-2 single-file checkpoints don't ship the
        // spatial upsampler asset (it's a separate Lightricks file the
        // companion list doesn't pull). The runtime's TwoStage / Distilled
        // paths require it and would bail mid-generation; the engine
        // should pick OneStage instead so the user gets a single-pass
        // video instead of a 500 several stages in.
        let gemma = tempfile::tempdir().unwrap();
        let mut paths = dummy_paths_with_gemma_root(gemma.path());
        paths.spatial_upscaler = None;

        let engine_22b = Ltx2Engine::new(
            "cv:2752735".to_string(),
            paths.clone(),
            LoadStrategy::Sequential,
            0,
        );
        let req = bare_t2v_req("cv:2752735");
        assert_eq!(
            engine_22b.select_pipeline(&req).unwrap(),
            PipelineKind::OneStage,
            "no spatial upsampler → OneStage (catalog cv:* default)"
        );

        let engine_distilled = Ltx2Engine::new(
            "ltx-2-19b-distilled:fp8".to_string(),
            paths,
            LoadStrategy::Sequential,
            0,
        );
        let req_distilled = bare_t2v_req("ltx-2-19b-distilled:fp8");
        assert_eq!(
            engine_distilled.select_pipeline(&req_distilled).unwrap(),
            PipelineKind::OneStage,
            "distilled name + missing spatial upsampler → OneStage fallback"
        );
    }

    #[test]
    fn audio_request_is_rejected_before_runtime_for_video_only_checkpoint_assets() {
        let mut req = bare_t2v_req("cv:3143864");
        req.source_image = Some(vec![0x89, b'P', b'N', b'G']);
        req.enable_audio = Some(true);

        let err =
            validate_audio_output_request(&req, || Some("the audio VAE and the vocoder".into()))
                .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("cv:3143864"), "got: {message}");
        assert!(message.contains("enable_audio=false"), "got: {message}");
        assert!(
            message.contains("before generation starts"),
            "got: {message}"
        );
        // The reason is carried through verbatim rather than flattened back
        // into "both the audio VAE and vocoder tensors".
        assert!(
            message.contains("the audio VAE and the vocoder"),
            "got: {message}"
        );

        // A complete checkpoint set passes.
        validate_audio_output_request(&req, || None).expect("no gap means no rejection");
    }

    /// The probe parses a safetensors header and formats a diagnostic string.
    /// Only an audio request can ever read it, so a request that does not want
    /// audio must not pay for it — this sits on the path of every LTX-2 render.
    #[test]
    fn audio_capability_probe_is_skipped_when_audio_is_not_wanted() {
        use std::cell::Cell;

        let probed = Cell::new(0usize);
        let mut req = bare_t2v_req("ltx-2-19b-dev:fp8");
        req.enable_audio = Some(false);
        req.output_format = Some(OutputFormat::Mp4);

        validate_audio_output_request(&req, || {
            probed.set(probed.get() + 1);
            Some("the vocoder".into())
        })
        .expect("audio was explicitly disabled");
        assert_eq!(
            probed.get(),
            0,
            "the probe must not run for a silent render"
        );

        // ...and it does run once the request actually wants audio.
        req.enable_audio = Some(true);
        let err = validate_audio_output_request(&req, || {
            probed.set(probed.get() + 1);
            Some("the vocoder".into())
        })
        .unwrap_err();
        assert_eq!(probed.get(), 1);
        assert!(err.to_string().contains("the vocoder"));
    }

    /// The message has to say *which* asset is missing, and separate "absent"
    /// from "present under an unrecognised layout" — the latter reads as a bad
    /// download otherwise, which is how the 19B vocoder-layout bug survived.
    #[test]
    fn audio_output_gap_names_the_specific_missing_asset() {
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;

        fn fixture(tag: &str, keys: &[&str]) -> std::path::PathBuf {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "mold-ltx2-gap-{tag}-{}-{}.safetensors",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let zero = 0.0f32.to_le_bytes().to_vec();
            let bufs: Vec<Vec<u8>> = keys.iter().map(|_| zero.clone()).collect();
            let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
            for (key, buf) in keys.iter().zip(bufs.iter()) {
                tensors.insert(
                    (*key).to_string(),
                    TensorView::new(SafeDtype::F32, vec![1], buf).unwrap(),
                );
            }
            serialize_to_file(&tensors, &None, &path).unwrap();
            path
        }

        fn paths_for(transformer: &std::path::Path) -> ModelPaths {
            ModelPaths {
                transformer: transformer.to_path_buf(),
                transformer_shards: vec![],
                vae: PathBuf::new(),
                spatial_upscaler: None,
                temporal_upscaler: None,
                distilled_lora: None,
                t5_encoder: None,
                clip_encoder: None,
                t5_tokenizer: None,
                clip_tokenizer: None,
                clip_encoder_2: None,
                clip_tokenizer_2: None,
                text_encoder_files: vec![],
                text_tokenizer: None,
                decoder: None,
            }
        }

        // Flat 19B layout: complete, and must report no gap at all.
        let flat = fixture(
            "flat",
            &[
                "audio_vae.per_channel_statistics.mean-of-means",
                "vocoder.conv_pre.weight",
            ],
        );
        assert_eq!(super::super::audio_output_gap(&paths_for(&flat)), None);

        // Audio VAE present, no vocoder tensors whatsoever.
        let no_vocoder = fixture(
            "no-vocoder",
            &["audio_vae.per_channel_statistics.mean-of-means"],
        );
        let gap = super::super::audio_output_gap(&paths_for(&no_vocoder)).unwrap();
        assert_eq!(gap, "the vocoder", "got: {gap}");

        // Vocoder tensors present but under a spelling this build cannot read.
        let odd_vocoder = fixture(
            "odd-vocoder",
            &[
                "audio_vae.per_channel_statistics.mean-of-means",
                "vocoder.some_future_layout.weight",
            ],
        );
        let gap = super::super::audio_output_gap(&paths_for(&odd_vocoder)).unwrap();
        assert!(
            gap.contains("not in a layout this build recognises"),
            "got: {gap}"
        );
        assert!(gap.contains("vocoder"), "got: {gap}");

        for path in [flat, no_vocoder, odd_vocoder] {
            let _ = std::fs::remove_file(path);
        }
    }

    fn bare_t2v_req(model: &str) -> GenerateRequest {
        GenerateRequest {
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            prompt: "test".to_string(),
            negative_prompt: None,
            model: model.to_string(),
            width: 768,
            height: 512,
            steps: 4,
            guidance: 3.5,
            seed: Some(42),
            batch_size: 1,
            output_format: Some(OutputFormat::Mp4),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
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
            frames: Some(25),
            fps: Some(24),
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
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
        }
    }

    #[test]
    fn pipeline_defaults_to_distilled_for_distilled_models() {
        let engine = Ltx2Engine::new(
            "ltx-2.3-22b-distilled:fp8".to_string(),
            dummy_paths(),
            LoadStrategy::Sequential,
            0,
        );
        let req = GenerateRequest {
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "ltx-2.3-22b-distilled:fp8".to_string(),
            width: 1216,
            height: 704,
            steps: 8,
            guidance: 1.0,
            seed: Some(1),
            batch_size: 1,
            output_format: Some(OutputFormat::Mp4),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
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
        };
        assert_eq!(
            engine.select_pipeline(&req).unwrap(),
            PipelineKind::Distilled
        );
    }

    #[test]
    fn constructor_consumes_the_selected_load_mode_explicitly() {
        let engine = Ltx2Engine::new(
            "ltx-2:fp8".to_string(),
            dummy_paths(),
            LoadStrategy::Sequential,
            3,
        );
        assert_eq!(
            engine.configured_load_strategy(),
            Some(LoadStrategy::Sequential)
        );
        assert_eq!(engine.configured_block_offload(), Some(false));
    }

    #[test]
    fn from_single_file_preserves_companion_paths() {
        // Regression: the original catalog route wired `cv:*` LTX-2 catalog entries into
        // `Ltx2Engine::from_single_file` but the constructor used to build
        // a fresh `ModelPaths` with `text_encoder_files: Vec::new()`,
        // discarding the Gemma TE companion the catalog bridge had
        // resolved. The runtime then bailed at `gemma_root` with
        // `LTX-2 requires Gemma text encoder files to be available`.
        // Pin the fix: companion fields (text_encoder_files,
        // spatial_upscaler, temporal_upscaler, distilled_lora) survive
        // the rebuild; only `transformer` and `vae` are overridden.
        let temp = tempfile::tempdir().unwrap();
        let checkpoint = temp.path().join("ltx2_combined.safetensors");
        // Build a minimal valid safetensors header with one transformer
        // key + one vae key so `single_file::load` returns has_vae=true.
        write_minimal_combined_ltx2_checkpoint(&checkpoint);

        let mut input_paths = dummy_paths_with_gemma_root(&temp.path().join("gemma"));
        input_paths.transformer = PathBuf::from("/wrong/path-should-be-overridden");
        input_paths.vae = PathBuf::from("/wrong/vae-should-be-cleared");
        let gemma_files_in = input_paths.text_encoder_files.clone();
        let spatial_in = input_paths.spatial_upscaler.clone();
        let temporal_in = input_paths.temporal_upscaler.clone();
        let distilled_in = input_paths.distilled_lora.clone();

        let engine = Ltx2Engine::from_single_file(
            "cv:2752735".to_string(),
            checkpoint.clone(),
            input_paths,
            LoadStrategy::Sequential,
            0,
        )
        .expect("from_single_file should succeed on a valid combined checkpoint");

        assert_eq!(
            engine.paths.transformer, checkpoint,
            "transformer must point at the single-file checkpoint"
        );
        assert_eq!(
            engine.paths.vae,
            PathBuf::default(),
            "vae must be cleared — runtime reads it from the same checkpoint via vb.pp(\"vae\")"
        );
        assert_eq!(
            engine.paths.text_encoder_files, gemma_files_in,
            "text_encoder_files (Gemma TE) must survive the rebuild — \
             dropping it is the cv:* loading regression"
        );
        assert_eq!(engine.paths.spatial_upscaler, spatial_in);
        assert_eq!(engine.paths.temporal_upscaler, temporal_in);
        assert_eq!(engine.paths.distilled_lora, distilled_in);
    }

    #[test]
    fn from_transformer_only_single_file_preserves_external_vae_for_chains() {
        let temp = tempfile::tempdir().unwrap();
        // Civitai converter checkpoints frequently omit model_version metadata,
        // so retain the version marker from the original filename as a fallback.
        let checkpoint = temp.path().join("ltx23_transformer.safetensors");
        let vae = temp.path().join("ltx2_vae.safetensors");
        let text_projection = temp.path().join("ltx-2.3_text_projection_bf16.safetensors");
        let gemma = temp.path().join("gemma");
        fs::create_dir_all(&gemma).unwrap();
        write_test_gemma_assets(&gemma);
        write_minimal_ltx2_checkpoint(&checkpoint, false);
        write_minimal_ltx2_checkpoint(&vae, true);
        fs::write(&text_projection, b"projection companion fixture").unwrap();

        let mut input_paths = dummy_paths_with_gemma_root(&gemma);
        input_paths.vae = vae.clone();
        input_paths.text_encoder_files.push(text_projection.clone());
        let engine = Ltx2Engine::from_single_file(
            "cv:3143864".to_string(),
            checkpoint.clone(),
            input_paths,
            LoadStrategy::Sequential,
            0,
        )
        .expect("transformer-only LTX-2 should use its resolved VAE companion");

        assert_eq!(engine.paths.transformer, checkpoint);
        assert_eq!(engine.paths.vae, vae);
        assert_eq!(engine.preset_hint.as_deref(), Some("2.3.0"));

        // Chain stages call materialize_request independently; pin the path
        // into the per-stage plan so every stage uses the same external VAE.
        let mut req = bare_t2v_req("cv:3143864");
        req.width = 64;
        req.height = 64;
        req.frames = Some(9);
        req.enable_audio = Some(false);
        let plan = engine
            .materialize_request(&req, temp.path(), &temp.path().join("out.mp4"))
            .unwrap();
        assert_eq!(plan.vae_checkpoint_path, vae.to_string_lossy());
        assert!(!plan.vae_in_checkpoint);
        assert_eq!(
            plan.text_projection_path.as_deref(),
            Some(text_projection.to_string_lossy().as_ref())
        );
    }

    fn write_minimal_combined_ltx2_checkpoint(path: &std::path::Path) {
        write_minimal_ltx2_checkpoint(path, true);
    }

    fn write_minimal_ltx2_checkpoint(path: &std::path::Path, include_vae: bool) {
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;
        let zero = 0.0f32.to_le_bytes().to_vec();
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        tensors.insert(
            "transformer_blocks.0.attn1.to_q.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1], &zero).unwrap(),
        );
        if include_vae {
            tensors.insert(
                "vae.encoder.conv_in.weight".to_string(),
                TensorView::new(SafeDtype::F32, vec![1], &zero).unwrap(),
            );
        }
        serialize_to_file(&tensors, &None, path).unwrap();
    }

    #[test]
    fn camera_control_preset_aliases_are_supported() {
        let preset = Ltx2Engine::camera_control_preset("dolly-in").unwrap();
        assert_eq!(
            preset.hf_filename,
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
            0,
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
            0,
        );
        let req = GenerateRequest {
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "ltx-2-19b-distilled:fp8".to_string(),
            width: 960,
            height: 576,
            steps: 8,
            guidance: 3.0,
            seed: Some(42),
            batch_size: 1,
            output_format: Some(OutputFormat::Mp4),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
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
            frames: Some(17),
            fps: Some(12),
            upscale_model: None,
            gif_preview: false,
            enable_audio: Some(false),
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
        write_minimal_ltx2_checkpoint(&paths.vae, true);

        let mut engine = Ltx2Engine::new(
            "ltx-2-19b-distilled:fp8".to_string(),
            paths,
            LoadStrategy::Sequential,
            0,
        );

        engine.load().unwrap();
        assert!(engine.is_loaded());
    }

    #[test]
    fn ltx2_unload_drops_runtime_and_reports_cuda_state() {
        let mut engine = Ltx2Engine::with_runtime_session(
            "ltx-2-19b-distilled:fp8".to_string(),
            dummy_paths(),
            Ltx2RuntimeSession::new_deferred_cuda(runtime_prompt_encoder(), 3),
        );
        engine.loaded = true;
        engine.gpu_ordinal = 3;

        assert_eq!(engine.unload_runtime_state(), Some(3));
        assert!(!engine.loaded);
        assert!(engine.native_runtime.is_none());
    }

    #[test]
    fn ltx2_unload_cpu_runtime_needs_no_cuda_synchronization() {
        let mut engine = Ltx2Engine::with_runtime_session(
            "ltx-2-19b-distilled:fp8".to_string(),
            dummy_paths(),
            runtime_session(),
        );
        engine.loaded = true;

        assert_eq!(engine.unload_runtime_state(), None);
        assert!(!engine.loaded);
        assert!(engine.native_runtime.is_none());
    }

    #[test]
    fn generate_runs_native_runtime_without_bridge_process() {
        let temp_dir = tempfile::tempdir().unwrap();
        let gemma_dir = temp_dir.path().join("gemma");
        fs::create_dir_all(&gemma_dir).unwrap();
        write_test_gemma_assets(&gemma_dir);
        let paths = dummy_paths_in(temp_dir.path(), &gemma_dir);
        fs::write(&paths.transformer, []).unwrap();
        write_minimal_ltx2_checkpoint(&paths.vae, true);

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
        assert!(engine.native_runtime.is_none());
    }

    #[test]
    fn render_chain_stage_rejects_a_pipeline_a_clip_carry_cannot_feed() {
        // Keyframe routes every conditioning item through the guiding-latent
        // path, so a frame-0 carry becomes a soft attractor instead of the
        // hard prefix pin the chain contract needs. Reject it up front,
        // before any runtime work happens.
        let mut engine = Ltx2Engine::with_runtime_session(
            "ltx-2-19b:fp8".to_string(),
            dummy_paths(),
            runtime_session(),
        );
        engine.loaded = true;
        let mut req = request(OutputFormat::Mp4, Some(false));
        req.pipeline = Some(mold_core::Ltx2PipelineMode::Keyframe);
        let err = engine
            .render_chain_stage(&req, None, 4, None)
            .expect_err("must fail on a pipeline a clip carry cannot feed");
        let msg = format!("{err}");
        assert!(
            msg.contains("Keyframe") || msg.contains("keyframe"),
            "error must name the offending pipeline, got: {msg}",
        );
    }

    /// A two-stage model must get *past* the gate. It will still fail later on
    /// the placeholder fixture checkpoint — the point is that the failure is
    /// no longer the gate.
    #[test]
    fn render_chain_stage_admits_a_two_stage_pipeline() {
        let mut engine = Ltx2Engine::with_runtime_session(
            "ltx-2-19b:fp8".to_string(),
            dummy_paths(),
            runtime_session(),
        );
        engine.loaded = true;
        let req = request(OutputFormat::Mp4, Some(false));
        let carry = ChainTail {
            frames: 4,
            tail_rgb_frames: vec![image::RgbImage::new(64, 64); 4],
        };
        if let Err(err) = engine.render_chain_stage(&req, Some(&carry), 4, None) {
            let msg = format!("{err}");
            assert!(
                !msg.contains("sequence clips render through"),
                "two-stage must not be rejected by the chain gate, got: {msg}",
            );
        }
    }

    /// A request-borne `hdr_exr_dir` on a chain/extend stage is cleared
    /// unless an explicit sidecar window arrives with it — the only route to
    /// a stage-level EXR target is the orchestrator's per-stage window, so a
    /// continuation can never stream clip-local `frame_00000.exr..` over
    /// another stage's frames (the latent extend-path bug #688 predicted).
    #[test]
    fn stage_sidecar_is_the_only_route_to_a_stage_exr_target() {
        let gemma_dir = tempfile::tempdir().unwrap();
        write_test_gemma_assets(gemma_dir.path());
        let plan = |dir: Option<&str>| -> Ltx2GeneratePlan {
            let engine = Ltx2Engine::with_runtime_session(
                "ltx-2-19b:fp8".to_string(),
                dummy_paths_with_gemma_root(gemma_dir.path()),
                runtime_session(),
            );
            let mut req = request(OutputFormat::Mp4, Some(false));
            req.hdr_exr_dir = dir.map(str::to_string);
            req.hdr_exr_full_float = dir.is_some();
            let work_dir = tempfile::tempdir().unwrap();
            let output = work_dir.path().join("out.mp4");
            engine
                .materialize_request(&req, work_dir.path(), &output)
                .unwrap()
        };

        // Without a window, the request-derived target is cleared.
        let mut cleared = plan(Some("/tmp/stage_exr"));
        assert_eq!(cleared.hdr_exr_dir.as_deref(), Some("/tmp/stage_exr"));
        super::apply_stage_sidecar_to_plan(&mut cleared, None);
        assert_eq!(cleared.hdr_exr_dir, None);
        assert!(!cleared.hdr_exr_full_float);
        assert_eq!(cleared.hdr_exr_window, None);

        // With a window, the sidecar is authoritative — dir, precision,
        // window, and the stage's reference offset all come from it.
        let mut stamped = plan(None);
        let sidecar = crate::chain::StageSidecar {
            exr_dir: std::path::PathBuf::from("/tmp/chain_exr"),
            full_float: true,
            window: crate::chain::ExrStageWindow {
                skip_leading: 17,
                start_index: 97,
                write_count: 24,
            },
            reference_frame_offset: 80,
        };
        super::apply_stage_sidecar_to_plan(&mut stamped, Some(&sidecar));
        assert_eq!(stamped.hdr_exr_dir.as_deref(), Some("/tmp/chain_exr"));
        assert!(stamped.hdr_exr_full_float);
        assert_eq!(
            stamped.hdr_exr_window,
            Some(crate::chain::ExrStageWindow {
                skip_leading: 17,
                start_index: 97,
                write_count: 24,
            }),
        );
        assert_eq!(stamped.reference_frame_offset, 80);
    }

    /// IC-LoRA joins the chain contract only under an orchestrator HDR
    /// sidecar. Without one it stays rejected — the reference-window
    /// authority is what makes a chained regrade well-defined.
    #[test]
    fn render_chain_stage_admits_ic_lora_only_with_a_sidecar() {
        let mut engine = Ltx2Engine::with_runtime_session(
            "ltx-2-19b:fp8".to_string(),
            dummy_paths(),
            runtime_session(),
        );
        engine.loaded = true;
        let mut req = request(OutputFormat::Mp4, Some(false));
        req.pipeline = Some(mold_core::Ltx2PipelineMode::IcLora);
        let err = engine
            .render_chain_stage(&req, None, 4, None)
            .expect_err("IcLora without a sidecar must stay rejected");
        assert!(
            format!("{err}").contains("sequence clips render through"),
            "got: {err}",
        );

        let sidecar = crate::chain::StageSidecar {
            exr_dir: std::path::PathBuf::from("/tmp/chain_exr"),
            full_float: false,
            window: crate::chain::ExrStageWindow {
                skip_leading: 0,
                start_index: 0,
                write_count: 97,
            },
            reference_frame_offset: 0,
        };
        // With a sidecar the gate admits IcLora; the render still fails
        // later on the placeholder fixture checkpoint — the point is the
        // failure is no longer the gate.
        if let Err(err) = engine.render_chain_stage(&req, None, 4, Some(&sidecar)) {
            let msg = format!("{err}");
            assert!(
                !msg.contains("sequence clips render through"),
                "IcLora with a sidecar must pass the chain gate, got: {msg}",
            );
        }
    }

    /// Two-stage joins the chain-capable set: `render_real_two_stage_av`
    /// already re-encodes conditioning at the stage-2 pixel shape, and the
    /// carry is decoded RGB rather than a latent, so it survives the x2 grid
    /// change by construction.
    ///
    /// The other four stay out for conditioning reasons, not effort:
    /// `Keyframe` routes every condition — including a frame-0 carry — through
    /// the guiding-latent path, which is a soft attractor where the chain
    /// contract needs a hard prefix pin; `A2Vid` and `IcLora` require
    /// conditioning that is mutually exclusive with the carry; and `Retake`
    /// regenerates a window of an existing clip, which has no "next clip"
    /// (`IcLora` is admitted by `render_chain_stage` only when an HDR
    /// sidecar supplies the per-stage reference window — see
    /// `render_chain_stage_admits_ic_lora_only_with_a_sidecar`).
    #[test]
    fn render_chain_supports_single_stage_distilled_and_two_stage_pipelines() {
        for supported in [
            PipelineKind::OneStage,
            PipelineKind::Distilled,
            PipelineKind::TwoStage,
            PipelineKind::TwoStageHq,
        ] {
            assert!(
                super::pipeline_supports_render_chain(supported),
                "{supported:?} must be chain-capable"
            );
        }
        for unsupported in [
            PipelineKind::Keyframe,
            PipelineKind::A2Vid,
            PipelineKind::IcLora,
            PipelineKind::Retake,
        ] {
            assert!(
                !super::pipeline_supports_render_chain(unsupported),
                "{unsupported:?} must stay out of the chain path"
            );
        }
    }

    #[test]
    fn render_chain_stage_rejects_zero_motion_tail() {
        // Zero-frame motion tail is nonsensical — it would narrow nothing off
        // for the next stage. Fast-fail before any allocation.
        let mut engine = Ltx2Engine::with_runtime_session(
            "ltx-2-19b-distilled:fp8".to_string(),
            dummy_paths(),
            runtime_session(),
        );
        engine.loaded = true;
        let req = request(OutputFormat::Mp4, Some(false));
        let err = engine
            .render_chain_stage(&req, None, 0, None)
            .expect_err("must fail on zero motion tail");
        let msg = format!("{err}");
        assert!(
            msg.contains("motion_tail_pixel_frames"),
            "error must name the motion_tail constraint, got: {msg}",
        );
    }

    /// CPU transformer → encoder pinned to the same device. The auto resolver
    /// must short-circuit before probing GPUs (which on a CUDA-less host
    /// would still pick CPU, but on a CUDA host must not place a 23 GB
    /// encoder on a card the transformer chose to skip).
    #[test]
    fn resolve_prompt_encoder_device_keeps_cpu_when_transformer_is_cpu() {
        let prior_main = std::env::var_os("MOLD_LTX2_GEMMA_DEVICE");
        let prior_legacy = std::env::var_os("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
        unsafe {
            std::env::remove_var("MOLD_LTX2_GEMMA_DEVICE");
            std::env::remove_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
        }

        let resolved = resolve_prompt_encoder_device(&Device::Cpu, 0);
        assert!(resolved.is_cpu());

        unsafe {
            if let Some(v) = prior_main {
                std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", v);
            }
            if let Some(v) = prior_legacy {
                std::env::set_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER", v);
            }
        }
    }

    #[test]
    fn prompt_encoder_oom_retry_preserves_transformer_device() {
        // Regression: the old OOM branch retried
        // `load_runtime_session_on_device(plan, Device::Cpu)`, silently moving
        // the ConvRot transformer and video VAE to CPU along with Gemma.
        // Use a sentinel instead of constructing a CUDA device so this policy
        // test remains hardware-independent in CPU-only CI.
        let (transformer, prompt) = prompt_encoder_oom_retry_placement(&7usize);
        assert_eq!(transformer, 7);
        assert_eq!(prompt, crate::device::LtxGemmaPlacement::Cpu);
    }

    #[test]
    fn cpu_text_placement_keeps_ltx2_runtime_on_leased_device() {
        let placement = mold_core::types::DevicePlacement {
            text_encoders: mold_core::types::DeviceRef::Cpu,
            advanced: Some(mold_core::types::AdvancedPlacement {
                transformer: mold_core::types::DeviceRef::device("cuda:0"),
                vae: mold_core::types::DeviceRef::device("cuda:0"),
                ..Default::default()
            }),
        };

        let (runtime, prompt) = ltx2_runtime_device_refs(Some(&placement));

        assert_eq!(
            runtime,
            Some(mold_core::types::DeviceRef::device("cuda:0")),
            "scheduler-materialized CPU text placement must not move image-to-video VAE work off the leased GPU",
        );
        assert_eq!(prompt, Some(mold_core::types::DeviceRef::Cpu));
    }

    #[test]
    fn legacy_cpu_text_override_does_not_become_runtime_placement() {
        let placement = mold_core::types::DevicePlacement {
            text_encoders: mold_core::types::DeviceRef::Cpu,
            advanced: None,
        };

        let (runtime, prompt) = ltx2_runtime_device_refs(Some(&placement));

        assert_eq!(
            runtime, None,
            "without an explicit transformer placement the runtime must retain its CUDA-first backend selection",
        );
        assert_eq!(prompt, Some(mold_core::types::DeviceRef::Cpu));
    }

    #[test]
    fn explicit_vae_placement_selects_runtime_when_transformer_is_auto() {
        let placement = mold_core::types::DevicePlacement {
            text_encoders: mold_core::types::DeviceRef::Cpu,
            advanced: Some(mold_core::types::AdvancedPlacement {
                transformer: mold_core::types::DeviceRef::Auto,
                vae: mold_core::types::DeviceRef::device("cuda:1"),
                ..Default::default()
            }),
        };

        let (runtime, prompt) = ltx2_runtime_device_refs(Some(&placement));

        assert_eq!(
            runtime,
            Some(mold_core::types::DeviceRef::device("cuda:1")),
            "the single-device native runtime must honor a concrete VAE pin when the transformer remains Auto",
        );
        assert_eq!(prompt, Some(mold_core::types::DeviceRef::Cpu));
    }

    /// `MOLD_LTX2_GEMMA_DEVICE=cpu` pins the encoder to CPU even when the
    /// transformer device is CUDA-shaped. We exercise this through the
    /// device-level resolver because the runtime path needs the same
    /// decision the load path will make and constructing a real CUDA
    /// device in CI isn't possible.
    #[test]
    fn resolver_picks_cpu_when_env_pins_cpu() {
        let prior_main = std::env::var_os("MOLD_LTX2_GEMMA_DEVICE");
        let prior_legacy = std::env::var_os("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
        unsafe {
            std::env::remove_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
            std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", "cpu");
        }
        assert_eq!(
            crate::device::resolve_ltx2_gemma_placement(0),
            crate::device::LtxGemmaPlacement::Cpu,
        );
        unsafe {
            std::env::remove_var("MOLD_LTX2_GEMMA_DEVICE");
            if let Some(v) = prior_main {
                std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", v);
            }
            if let Some(v) = prior_legacy {
                std::env::set_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER", v);
            }
        }
    }
}
