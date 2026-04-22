#![allow(clippy::too_many_arguments)]

use anyhow::{Context, Result};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::ltx_video::sampling::{
    FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig, TimeShiftType,
};
use image::{imageops, GenericImage, Rgb, RgbImage};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::env;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use super::conditioning::retake_temporal_mask;
use super::execution::SamplerMode;
use super::guidance::{
    BatchedPerturbationConfig, MultiModalGuider, MultiModalGuiderParams, Perturbation,
    PerturbationConfig, PerturbationType,
};
use super::lora;
use super::media;
use super::model::{
    audio_temporal_positions, cross_modal_temporal_positions, derive_stage1_render_shape,
    get_pixel_coords, scale_video_time_to_seconds, spatially_upsample_frames,
    temporally_upsample_frames_x2, video_token_positions,
    video_transformer::{Ltx2AvTransformer3DModel, Ltx2VideoTransformer3DModelConfig},
    video_vae::{AutoencoderKLLtx2Video, AutoencoderKLLtx2VideoConfig},
    AudioLatentShape, AudioPatchifier, DecodedAudio, Ltx2AudioDecoder, Ltx2AudioEncoder,
    Ltx2VocoderWithBwe, SpatioTemporalScaleFactors, VideoLatentPatchifier, VideoLatentShape,
    VideoPixelShape,
};
use super::plan::{Ltx2GeneratePlan, PipelineKind};
use super::sampler::{euler_step, res2s_step};
use super::text::connectors::EmbeddingsProcessorOutput;
use super::text::prompt_encoder::{NativePromptEncoder, NativePromptEncoding};
use crate::device::{fmt_gb, free_vram_bytes};
use crate::engine::{gpu_dtype, seeded_randn};
use crate::img_utils::{decode_source_image, NormalizeRange};
use crate::ltx_video::latent_upsampler::LatentUpsampler;
use crate::progress::{ProgressCallback, ProgressEvent, ProgressReporter};
use crate::weight_loader::load_fp8_safetensors;
use mold_core::{LoraWeight, Ltx2SpatialUpscale, TimeRange};

pub const LTX2_VIDEO_LATENT_CHANNELS: usize = 128;
pub const LTX2_AUDIO_LATENT_CHANNELS: usize = 8;
pub const LTX2_AUDIO_MEL_BINS: usize = 16;
pub const LTX2_AUDIO_SAMPLE_RATE: usize = 16_000;
pub const LTX2_AUDIO_HOP_LENGTH: usize = 160;
pub const LTX2_AUDIO_LATENT_DOWNSAMPLE_FACTOR: usize = 4;

#[derive(Debug)]
pub struct NativePreparedRun {
    pub prompt: NativePromptEncoding,
    pub debug_alt_prompt: Option<EmbeddingsProcessorOutput>,
    pub video_pixel_shape: VideoPixelShape,
    pub video_latent_shape: VideoLatentShape,
    pub audio_latent_shape: Option<AudioLatentShape>,
    pub video_positions: Tensor,
    pub audio_positions: Option<Tensor>,
    #[allow(dead_code)]
    pub cross_modal_temporal_positions: Option<(Tensor, Tensor)>,
    pub retake_mask: Option<Vec<f32>>,
}

#[derive(Debug)]
pub struct NativeRenderedVideo {
    pub frames: Vec<RgbImage>,
    pub audio_track: Option<NativeAudioTrack>,
    pub has_audio: bool,
    pub audio_sample_rate: Option<u32>,
    pub audio_channels: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct NativeAudioTrack {
    pub interleaved_samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug)]
struct NativeConditioningAudio {
    latents: Tensor,
    original_track: Option<NativeAudioTrack>,
}

#[derive(Debug)]
struct NativeConditioningVideo {
    latents: Tensor,
}

#[derive(Debug, Clone)]
struct VideoTokenReplacement {
    start_token: usize,
    tokens: Tensor,
    strength: f64,
}

#[derive(Debug, Clone)]
struct VideoTokenAppendCondition {
    tokens: Tensor,
    positions: Tensor,
    strength: f64,
}

#[derive(Debug, Clone, Default)]
struct StageVideoConditioning {
    replacements: Vec<VideoTokenReplacement>,
    appended: Vec<VideoTokenAppendCondition>,
}

impl StageVideoConditioning {
    fn is_empty(&self) -> bool {
        self.replacements.is_empty() && self.appended.is_empty()
    }
}

#[derive(Debug, Clone, Copy)]
struct RenderPromptInputOptions {
    include_unconditional: bool,
    include_alt: bool,
}

#[derive(Debug)]
struct RenderPromptInputs {
    cond_context: Tensor,
    uncond_context: Option<Tensor>,
    audio_shape: Option<AudioLatentShape>,
    audio_context: Option<Tensor>,
    uncond_audio_context: Option<Tensor>,
    alt_context: Option<Tensor>,
    alt_audio_context: Option<Tensor>,
    video_positions: Tensor,
    audio_positions: Option<Tensor>,
}

fn prepare_render_prompt_inputs(
    prepared: &NativePreparedRun,
    device: &candle_core::Device,
    options: RenderPromptInputOptions,
) -> Result<RenderPromptInputs> {
    let cond_context = prepared
        .prompt
        .conditional
        .video_encoding
        .to_device(device)?;
    let uncond_context = if options.include_unconditional {
        Some(
            prepared
                .prompt
                .unconditional
                .video_encoding
                .to_device(device)?,
        )
    } else {
        None
    };
    let audio_context = prepared
        .prompt
        .conditional
        .audio_encoding
        .as_ref()
        .map(|tensor| tensor.to_device(device))
        .transpose()?;
    let uncond_audio_context = if options.include_unconditional {
        prepared
            .prompt
            .unconditional
            .audio_encoding
            .as_ref()
            .map(|tensor| tensor.to_device(device))
            .transpose()?
    } else {
        None
    };
    let alt_context = if options.include_alt {
        prepared
            .debug_alt_prompt
            .as_ref()
            .map(|prompt| prompt.video_encoding.to_device(device))
            .transpose()?
    } else {
        None
    };
    let alt_audio_context = if options.include_alt {
        prepared
            .debug_alt_prompt
            .as_ref()
            .and_then(|prompt| prompt.audio_encoding.as_ref())
            .map(|tensor| tensor.to_device(device))
            .transpose()?
    } else {
        None
    };
    let video_positions = prepared.video_positions.to_device(device)?;
    let audio_positions = prepared
        .audio_positions
        .as_ref()
        .map(|tensor| tensor.to_device(device))
        .transpose()?;

    Ok(RenderPromptInputs {
        cond_context,
        uncond_context,
        audio_shape: prepared.audio_latent_shape,
        audio_context,
        uncond_audio_context,
        alt_context,
        alt_audio_context,
        video_positions,
        audio_positions,
    })
}

struct Ltx2VaeLatentStats {
    mean: Tensor,
    std: Tensor,
}

impl Ltx2VaeLatentStats {
    fn load(plan: &Ltx2GeneratePlan, device: &candle_core::Device, dtype: DType) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&Path::new(&plan.checkpoint_path)),
                dtype,
                device,
            )?
        };
        let config = ltx2_video_vae_config(plan);
        let stats_vb = vb.pp("vae").pp("per_channel_statistics");
        let mean = if stats_vb.contains_tensor("mean-of-means") {
            stats_vb.get(config.latent_channels, "mean-of-means")?
        } else {
            tracing::debug!(
                checkpoint = %plan.checkpoint_path,
                "native LTX-2 VAE checkpoint missing mean-of-means statistics, falling back to config defaults"
            );
            Tensor::new(config.latents_mean.as_slice(), device)?.to_dtype(dtype)?
        };
        let std = if stats_vb.contains_tensor("std-of-means") {
            stats_vb.get(config.latent_channels, "std-of-means")?
        } else {
            tracing::debug!(
                checkpoint = %plan.checkpoint_path,
                "native LTX-2 VAE checkpoint missing std-of-means statistics, falling back to config defaults"
            );
            Tensor::new(config.latents_std.as_slice(), device)?.to_dtype(dtype)?
        };
        Ok(Self { mean, std })
    }

    fn normalize(&self, latents: &Tensor) -> Result<Tensor> {
        let channels = latents.dim(1)?;
        let mean = self
            .mean
            .reshape((1, channels, 1, 1, 1))?
            .to_device(latents.device())?
            .to_dtype(latents.dtype())?;
        let std = self
            .std
            .reshape((1, channels, 1, 1, 1))?
            .to_device(latents.device())?
            .to_dtype(latents.dtype())?;
        latents
            .broadcast_sub(&mean)?
            .broadcast_div(&std)
            .map_err(Into::into)
    }

    fn denormalize(&self, latents: &Tensor) -> Result<Tensor> {
        let channels = latents.dim(1)?;
        let mean = self
            .mean
            .reshape((1, channels, 1, 1, 1))?
            .to_device(latents.device())?
            .to_dtype(latents.dtype())?;
        let std = self
            .std
            .reshape((1, channels, 1, 1, 1))?
            .to_device(latents.device())?
            .to_dtype(latents.dtype())?;
        latents
            .broadcast_mul(&std)?
            .broadcast_add(&mean)
            .map_err(Into::into)
    }
}

pub struct Ltx2RuntimeSession {
    device: Option<candle_core::Device>,
    prompt_encoder: Option<NativePromptEncoder>,
    /// Cached output of the last successful `encode_prompt_pair_with_unconditional`
    /// call. The prompt encoder is intentionally consumed during the first
    /// `prepare()` so its VRAM can be freed for the transformer (see the
    /// `take()` + drop pattern below); that leaves subsequent `prepare()`
    /// calls on the same session with no encoder. For the render-chain
    /// path every stage shares the same prompt tokens, so we cache the
    /// encoding after the first encode and reuse it on follow-up stages —
    /// no re-encode, no encoder re-load, no VRAM re-hit.
    cached_prompt_encoding: Option<CachedPromptEncoding>,
    /// Optional slot wired into `render_real_distilled_av` so
    /// `Ltx2Engine::render_chain_stage` can snapshot the pre-VAE-decode
    /// final latents and forward them to the next chain stage as a
    /// [`super::chain::ChainTail`]. `None` outside chain flow.
    pub(crate) tail_capture: Option<std::sync::Arc<std::sync::Mutex<Option<Tensor>>>>,
    /// GPU ordinal inherited from `Ltx2Engine`. Used for the deferred CUDA
    /// device creation in `prepare()` and for post-OOM context reset.
    gpu_ordinal: usize,
}

/// Remembers the last `encode_prompt_pair_with_unconditional` call so
/// successive `prepare()` calls with the same prompt can skip the encoder
/// entirely — used by the render-chain path where stages share a prompt.
struct CachedPromptEncoding {
    token_pair: super::text::gemma::EncodedPromptPair,
    encode_unconditional: bool,
    encoding: NativePromptEncoding,
    prompt_device_is_cuda: bool,
    prepared_device: candle_core::Device,
}

impl Ltx2RuntimeSession {
    pub fn new(
        device: candle_core::Device,
        prompt_encoder: NativePromptEncoder,
        gpu_ordinal: usize,
    ) -> Self {
        Self {
            device: Some(device),
            prompt_encoder: Some(prompt_encoder),
            cached_prompt_encoding: None,
            tail_capture: None,
            gpu_ordinal,
        }
    }

    pub fn new_deferred_cuda(prompt_encoder: NativePromptEncoder, gpu_ordinal: usize) -> Self {
        Self {
            device: None,
            prompt_encoder: Some(prompt_encoder),
            cached_prompt_encoding: None,
            tail_capture: None,
            gpu_ordinal,
        }
    }

    /// Arm the pre-VAE-decode latent capture slot. The distilled render
    /// path writes its `final_video_latents` into the returned slot when
    /// this is set, letting a caller drain the raw latents after a render
    /// completes. Kept after the v1.1 decoded-pixel-carryover switch in
    /// case future work (e.g. quality-diagnostic tooling) wants access
    /// to the pre-decode tensor; the production chain path no longer
    /// arms it.
    #[allow(dead_code)]
    pub(crate) fn arm_tail_capture(&mut self) -> std::sync::Arc<std::sync::Mutex<Option<Tensor>>> {
        let slot = std::sync::Arc::new(std::sync::Mutex::new(None));
        self.tail_capture = Some(std::sync::Arc::clone(&slot));
        slot
    }

    /// Disarm the latent capture slot. See [`arm_tail_capture`].
    #[allow(dead_code)]
    pub(crate) fn clear_tail_capture(&mut self) {
        self.tail_capture = None;
    }

    /// Whether this session can serve `plan` without a rebuild. Returns
    /// `true` if the encoder is still available OR the cached encoding
    /// matches the plan's prompt tokens. Callers use this to decide
    /// whether to reuse a persisted runtime (fast path — keeps transformer
    /// and VAE warm) or drop it and build a fresh one (the only way to
    /// recover when the encoder has been consumed on a prior `prepare()`
    /// and a different prompt arrives).
    pub fn can_reuse_for(&self, plan: &Ltx2GeneratePlan) -> bool {
        if self.prompt_encoder.is_some() {
            return true;
        }
        let Ok(encode_unconditional) = prompt_requires_unconditional_context(plan) else {
            return false;
        };
        // Alt-prompt debug mode requires the live encoder; cache alone
        // isn't sufficient.
        if ltx_debug_alt_prompt().is_some() {
            return false;
        }
        self.cached_prompt_encoding.as_ref().is_some_and(|cached| {
            cached.encode_unconditional == encode_unconditional
                && cached.token_pair == plan.prompt_tokens
        })
    }

    pub fn prepare(&mut self, plan: &Ltx2GeneratePlan) -> Result<NativePreparedRun> {
        let prepare_total_start = Instant::now();
        let mut stage1_shape = derive_stage1_render_shape(
            plan.width,
            plan.height,
            plan.num_frames,
            plan.frame_rate,
            plan.spatial_upscale,
            plan.temporal_upscale,
        );
        if pipeline_uses_two_stage_spatial_refinement(plan.pipeline)
            && plan.spatial_upscale.is_none()
            && stage1_shape.width > 16
            && stage1_shape.height > 16
        {
            let implicit_x2_shape = derive_stage1_render_shape(
                plan.width,
                plan.height,
                plan.num_frames,
                plan.frame_rate,
                Some(Ltx2SpatialUpscale::X2),
                plan.temporal_upscale,
            );
            stage1_shape.width = implicit_x2_shape.width;
            stage1_shape.height = implicit_x2_shape.height;
        }
        let encode_unconditional_prompt = prompt_requires_unconditional_context(plan)?;
        let alt_prompt_env = ltx_debug_alt_prompt();
        // Chain path fast-path: if a previous `prepare()` already encoded
        // the exact same prompt+unconditional combo, reuse those embeddings
        // instead of demanding the encoder back. Disabled when the
        // `MOLD_LTX_DEBUG_ALT_PROMPT` debug hook is active because that branch
        // still needs the live encoder.
        let cache_hit = alt_prompt_env.is_none()
            && self.cached_prompt_encoding.as_ref().is_some_and(|cached| {
                cached.encode_unconditional == encode_unconditional_prompt
                    && cached.token_pair == plan.prompt_tokens
            });
        let (prompt_device_is_cuda, prepared_device, prompt, debug_alt_prompt) = if cache_hit {
            let cached = self
                .cached_prompt_encoding
                .as_ref()
                .expect("cache_hit implies cached_prompt_encoding is Some");
            log_timing("prepare.prompt_pair", Instant::now());
            (
                cached.prompt_device_is_cuda,
                cached.prepared_device.clone(),
                cached.encoding.clone(),
                None,
            )
        } else {
            let mut prompt_encoder = self
                .prompt_encoder
                .take()
                .context("native LTX-2 prompt encoder is unavailable")?;
            let prompt_device_is_cuda = prompt_encoder.device().is_cuda();
            let prepared_device = if prompt_device_is_cuda || prompt_encoder.device().is_metal() {
                candle_core::Device::Cpu
            } else {
                prompt_encoder.device().clone()
            };
            let prompt_encode_start = Instant::now();
            let prompt = move_prompt_encoding_to_device(
                prompt_encoder.encode_prompt_pair_with_unconditional(
                    &plan.prompt_tokens,
                    encode_unconditional_prompt,
                )?,
                &prepared_device,
            )?;
            log_timing("prepare.prompt_pair", prompt_encode_start);
            let alt_prompt_start = Instant::now();
            let debug_alt_prompt = match alt_prompt_env.clone() {
                Some(alt_prompt) => {
                    let assets =
                        super::text::gemma::GemmaAssets::discover(Path::new(&plan.gemma_root))
                            .with_context(|| {
                                format!(
                            "failed to discover Gemma assets for alternate prompt debug at '{}'",
                            plan.gemma_root
                        )
                            })?;
                    let alt_tokens =
                        assets.encode_prompt_pair(&alt_prompt, plan.negative_prompt.as_deref())?;
                    let alt_prompt = prompt_encoder
                        .encode_prompt_pair(&alt_tokens)
                        .context("failed to encode alternate debug prompt")?;
                    Some(move_embeddings_output_to_device(
                        alt_prompt.conditional,
                        &prepared_device,
                    )?)
                }
                None => None,
            };
            log_timing("prepare.alt_prompt", alt_prompt_start);
            let prompt_debug_start = Instant::now();
            if ltx_debug_enabled() {
                log_prompt_debug_stats(plan, &prompt)?;
                if let Some(alt_prompt) = debug_alt_prompt.as_ref() {
                    log_alt_prompt_debug_stats(plan, &prompt.conditional, alt_prompt)?;
                }
            }
            log_timing("prepare.prompt_debug", prompt_debug_start);
            // Cache the encoding for the next chain stage. Dropping the
            // encoder here (end of the else branch) still happens — we're
            // only holding on to the `NativePromptEncoding` output, not the
            // encoder itself, so the VRAM-free property of the original
            // take() pattern is preserved.
            self.cached_prompt_encoding = Some(CachedPromptEncoding {
                token_pair: plan.prompt_tokens.clone(),
                encode_unconditional: encode_unconditional_prompt,
                encoding: prompt.clone(),
                prompt_device_is_cuda,
                prepared_device: prepared_device.clone(),
            });
            (
                prompt_device_is_cuda,
                prepared_device,
                prompt,
                debug_alt_prompt,
            )
        };
        let device_handoff_start = Instant::now();
        if prompt_device_is_cuda {
            if self.device.is_none() {
                crate::device::reclaim_gpu_memory(self.gpu_ordinal);
                self.device = Some(new_native_cuda_device(self.gpu_ordinal)?);
            } else if let Some(device) = self.device.as_ref() {
                if device.is_cuda() {
                    device.synchronize()?;
                }
            }
        }
        log_timing("prepare.device_handoff", device_handoff_start);
        let positions_start = Instant::now();
        let pixel_shape = VideoPixelShape {
            batch: 1,
            frames: stage1_shape.frames as usize,
            height: stage1_shape.height as usize,
            width: stage1_shape.width as usize,
            fps: stage1_shape.fps as f32,
        };
        let scale_factors = SpatioTemporalScaleFactors::default();
        let video_latent_shape = VideoLatentShape::from_pixel_shape(
            pixel_shape,
            LTX2_VIDEO_LATENT_CHANNELS,
            scale_factors,
        );
        let video_patchifier = VideoLatentPatchifier::new(1);
        let video_positions = scale_video_time_to_seconds(
            &get_pixel_coords(
                &video_token_positions(video_patchifier, video_latent_shape, &prepared_device)?,
                scale_factors,
                true,
            )?,
            pixel_shape.fps,
        )?;

        // The public LTX-2 checkpoints are audio-video transformers even for
        // silent exports. Keep the internal audio branch active whenever the
        // prompt encoder emitted audio conditioning so the denoiser stays on the
        // same multimodal path as upstream; export semantics remain silent
        // unless the request explicitly wants audio output.
        let prompt_has_audio_conditioning = prompt.conditional.audio_encoding.is_some()
            || prompt.unconditional.audio_encoding.is_some();
        let wants_audio_latents = if ltx_debug_disable_audio_branch_enabled() {
            false
        } else {
            plan.execution_graph.wants_audio_output
                || plan.execution_graph.uses_audio_conditioning
                || prompt_has_audio_conditioning
        };
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
                let audio_positions =
                    audio_temporal_positions(audio_patchifier, audio_shape, &prepared_device)?;
                let cross_modal =
                    cross_modal_temporal_positions(&video_positions, &audio_positions)?;
                (Some(audio_shape), Some(audio_positions), Some(cross_modal))
            } else {
                (None, None, None)
            };
        log_timing("prepare.positions", positions_start);

        let retake_mask_start = Instant::now();
        let retake_mask = plan
            .retake_range
            .as_ref()
            .map(|range| retake_temporal_mask(range, stage1_shape.fps, stage1_shape.frames))
            .transpose()?;
        log_timing("prepare.retake_mask", retake_mask_start);
        log_timing("prepare.total", prepare_total_start);

        Ok(NativePreparedRun {
            prompt,
            debug_alt_prompt,
            video_pixel_shape: pixel_shape,
            video_latent_shape,
            audio_latent_shape,
            video_positions,
            audio_positions,
            cross_modal_temporal_positions,
            retake_mask,
        })
    }

    pub fn render_native_video(
        &self,
        plan: &Ltx2GeneratePlan,
        prepared: &NativePreparedRun,
        progress: Option<&ProgressCallback>,
    ) -> Result<NativeRenderedVideo> {
        let device = self
            .device
            .as_ref()
            .context("native LTX-2 compute device was not initialized")?;
        if let Some(rendered) = self.try_render_real_video(plan, prepared, device, progress)? {
            if ltx_debug_enabled() || env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
                eprintln!(
                    "[ltx2-debug] render_native_video using real path pipeline={:?}",
                    plan.pipeline
                );
            }
            return Ok(rendered);
        }
        if ltx_debug_enabled() || env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
            eprintln!(
                "[ltx2-debug] render_native_video falling back to placeholder path pipeline={:?}",
                plan.pipeline
            );
        }

        let summary = RenderSummary::from_prepared(prepared)?;
        let seed = plan.seed ^ 0x4c54_5832_4e41_5449;
        let mut rng = StdRng::seed_from_u64(seed);
        let phase = rng.gen_range(0.0..std::f32::consts::TAU);
        let base_width = prepared.video_pixel_shape.width as u32;
        let base_height = prepared.video_pixel_shape.height as u32;
        let base_frames = prepared.video_pixel_shape.frames as u32;
        let overlays = load_conditioning_overlays(plan, base_width, base_height, base_frames)?;

        let mut frames = Vec::with_capacity(base_frames as usize);
        for frame_idx in 0..base_frames {
            let mut frame = RgbImage::new(base_width, base_height);
            let t = if base_frames <= 1 {
                0.0
            } else {
                frame_idx as f32 / (base_frames - 1) as f32
            };
            let retake_strength = prepared
                .retake_mask
                .as_ref()
                .and_then(|mask| mask.get(frame_idx as usize))
                .copied()
                .unwrap_or(0.0);
            fill_background(
                &mut frame,
                t,
                phase,
                &summary,
                retake_strength,
                plan.execution_graph.uses_audio_conditioning,
                plan.execution_graph.uses_reference_video_conditioning,
            );
            apply_conditioning_overlays(&mut frame, frame_idx, base_frames, &overlays);
            frames.push(frame);
        }
        if plan.temporal_upscale.is_some() {
            frames = temporally_upsample_frames_x2(&frames, Some(plan.num_frames));
        }
        if plan.spatial_upscale.is_some() || plan.width != base_width || plan.height != base_height
        {
            frames = spatially_upsample_frames(&frames, plan.width, plan.height);
        }

        Ok(NativeRenderedVideo {
            frames,
            audio_track: None,
            has_audio: plan.execution_graph.wants_audio_output,
            audio_sample_rate: plan.execution_graph.wants_audio_output.then_some(48_000),
            audio_channels: plan.execution_graph.wants_audio_output.then_some(2),
        })
    }

    fn try_render_real_video(
        &self,
        plan: &Ltx2GeneratePlan,
        prepared: &NativePreparedRun,
        device: &candle_core::Device,
        progress: Option<&ProgressCallback>,
    ) -> Result<Option<NativeRenderedVideo>> {
        if !supports_real_video_path(plan) {
            if ltx_debug_enabled() || env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
                eprintln!(
                    "[ltx2-debug] real path rejected by supports_real_video_path pipeline={:?}",
                    plan.pipeline
                );
            }
            return Ok(None);
        }
        if !Path::new(&plan.checkpoint_path).is_file() {
            if ltx_debug_enabled() || env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
                eprintln!(
                    "[ltx2-debug] real path rejected because checkpoint is missing: {}",
                    plan.checkpoint_path
                );
            }
            return Ok(None);
        }
        let render = match plan.pipeline {
            PipelineKind::Distilled => render_real_distilled_av(
                plan,
                prepared,
                device,
                progress,
                self.tail_capture.as_ref(),
            ),
            PipelineKind::OneStage => render_real_one_stage_av(plan, prepared, device, progress),
            PipelineKind::TwoStage
            | PipelineKind::TwoStageHq
            | PipelineKind::IcLora
            | PipelineKind::Keyframe
            | PipelineKind::A2Vid => render_real_two_stage_av(plan, prepared, device, progress),
            PipelineKind::Retake => render_real_retake_av(plan, prepared, device, progress),
        };
        match render {
            Ok(rendered) => Ok(Some(rendered)),
            Err(err) if is_placeholder_checkpoint_error(&err) => {
                if ltx_debug_enabled() || env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
                    eprintln!(
                        "[ltx2-debug] real path fell back due to placeholder checkpoint error: {err:#}"
                    );
                }
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }
}

fn move_prompt_encoding_to_device(
    prompt: NativePromptEncoding,
    device: &candle_core::Device,
) -> Result<NativePromptEncoding> {
    Ok(NativePromptEncoding {
        conditional: move_embeddings_output_to_device(prompt.conditional, device)?,
        unconditional: move_embeddings_output_to_device(prompt.unconditional, device)?,
    })
}

fn move_embeddings_output_to_device(
    output: EmbeddingsProcessorOutput,
    device: &candle_core::Device,
) -> Result<EmbeddingsProcessorOutput> {
    Ok(EmbeddingsProcessorOutput {
        video_encoding: output.video_encoding.to_device(device)?,
        audio_encoding: output
            .audio_encoding
            .map(|tensor| tensor.to_device(device))
            .transpose()?,
        attention_mask: output.attention_mask.to_device(device)?,
    })
}

#[derive(Debug, Clone)]
struct ConditioningOverlay {
    frame: u32,
    strength: f32,
    image: RgbImage,
}

#[derive(Debug, Clone, Copy)]
struct RenderSummary {
    video_mean: f32,
    video_energy: f32,
    audio_mean: f32,
    audio_energy: f32,
    negative_bias: f32,
}

impl RenderSummary {
    fn from_prepared(prepared: &NativePreparedRun) -> Result<Self> {
        let video_mean = tensor_mean(&prepared.prompt.conditional.video_encoding)?;
        let negative_bias = tensor_mean(&prepared.prompt.unconditional.video_encoding)?;
        let video_energy = tensor_energy(&prepared.video_positions)?;
        let audio_mean = prepared
            .prompt
            .conditional
            .audio_encoding
            .as_ref()
            .map(tensor_mean)
            .transpose()?
            .unwrap_or(0.0);
        let audio_energy = prepared
            .audio_positions
            .as_ref()
            .map(tensor_energy)
            .transpose()?
            .unwrap_or(0.0);
        Ok(Self {
            video_mean,
            video_energy,
            audio_mean,
            audio_energy,
            negative_bias,
        })
    }
}

fn tensor_mean(tensor: &Tensor) -> Result<f32> {
    Ok(tensor
        .flatten_all()?
        .to_dtype(DType::F32)?
        .mean_all()?
        .to_scalar::<f32>()?)
}

fn tensor_energy(tensor: &Tensor) -> Result<f32> {
    Ok(tensor
        .flatten_all()?
        .to_dtype(DType::F32)?
        .abs()?
        .mean_all()?
        .to_scalar::<f32>()?)
}

fn emit_denoise_progress(
    progress: Option<&ProgressCallback>,
    step: usize,
    total: usize,
    elapsed: std::time::Duration,
) {
    if let Some(progress) = progress {
        progress(ProgressEvent::DenoiseStep {
            step,
            total,
            elapsed,
        });
    }
}

fn load_conditioning_overlays(
    plan: &Ltx2GeneratePlan,
    width: u32,
    height: u32,
    stage_frames: u32,
) -> Result<Vec<ConditioningOverlay>> {
    plan.conditioning
        .images
        .iter()
        .map(|image| {
            let overlay = image::open(&image.path)
                .with_context(|| {
                    format!("failed to load staged conditioning image '{}'", image.path)
                })?
                .to_rgb8();
            Ok(ConditioningOverlay {
                frame: remap_conditioning_frame(image.frame, plan.num_frames, stage_frames),
                strength: image.strength,
                image: imageops::resize(&overlay, width, height, imageops::FilterType::Triangle),
            })
        })
        .collect()
}

fn remap_conditioning_frame(source_frame: u32, source_total: u32, target_total: u32) -> u32 {
    if source_total <= 1 || target_total <= 1 {
        return 0;
    }
    let mapped = ((source_frame as u64 * (target_total - 1) as u64)
        + ((source_total - 1) / 2) as u64)
        / (source_total - 1) as u64;
    mapped.min((target_total - 1) as u64) as u32
}

fn fill_background(
    frame: &mut RgbImage,
    t: f32,
    phase: f32,
    summary: &RenderSummary,
    retake_strength: f32,
    uses_audio_conditioning: bool,
    uses_reference_video: bool,
) {
    let width = frame.width().max(1) as f32;
    let height = frame.height().max(1) as f32;
    let motion = 1.5 + summary.video_energy.abs() * 3.0;
    let audio_motion = 1.0 + summary.audio_energy.abs() * 2.0;
    let bias = summary.negative_bias.tanh() * 0.15;
    let highlight = 0.15 + retake_strength * 0.35;

    for (x, y, pixel) in frame.enumerate_pixels_mut() {
        let fx = x as f32 / width;
        let fy = y as f32 / height;
        let primary = ((fx * 6.0 + t * motion + phase).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let secondary =
            ((fy * 4.0 - t * (motion * 0.7) + phase * 0.5).cos() * 0.5 + 0.5).clamp(0.0, 1.0);
        let ripple =
            (((fx + fy) * (3.0 + summary.audio_mean.abs()) + t * audio_motion + phase * 1.7).sin()
                * 0.5
                + 0.5)
                .clamp(0.0, 1.0);

        let mut r = primary * (200.0 + summary.video_mean.abs() * 80.0) + secondary * 32.0;
        let mut g = secondary * (180.0 + summary.audio_mean.abs() * 90.0) + ripple * 40.0;
        let mut b = ripple * 220.0 + primary * 18.0 + bias * 255.0;

        if uses_audio_conditioning && fy > 0.78 {
            let bars = ((fx * 18.0 + t * 9.0 + phase).sin() * 0.5 + 0.5) * 110.0;
            g += bars;
            b += bars * 0.35;
        }
        if uses_reference_video && fx < 0.08 {
            r += 36.0;
            b += 22.0;
        }
        if retake_strength > 0.0 && (!(0.03..=0.97).contains(&fx) || !(0.03..=0.97).contains(&fy)) {
            r += highlight * 255.0;
            g += highlight * 96.0;
        }

        *pixel = Rgb([
            r.clamp(0.0, 255.0) as u8,
            g.clamp(0.0, 255.0) as u8,
            b.clamp(0.0, 255.0) as u8,
        ]);
    }
}

fn apply_conditioning_overlays(
    frame: &mut RgbImage,
    frame_idx: u32,
    total_frames: u32,
    overlays: &[ConditioningOverlay],
) {
    for overlay in overlays {
        let alpha = overlay_alpha(overlay, frame_idx, total_frames);
        if alpha <= 0.0 {
            continue;
        }
        for (dst, src) in frame.pixels_mut().zip(overlay.image.pixels()) {
            let alpha = alpha.clamp(0.0, 1.0);
            let inv = 1.0 - alpha;
            *dst = Rgb([
                (dst[0] as f32 * inv + src[0] as f32 * alpha).round() as u8,
                (dst[1] as f32 * inv + src[1] as f32 * alpha).round() as u8,
                (dst[2] as f32 * inv + src[2] as f32 * alpha).round() as u8,
            ]);
        }
    }
}

fn overlay_alpha(overlay: &ConditioningOverlay, frame_idx: u32, total_frames: u32) -> f32 {
    let distance = overlay.frame.abs_diff(frame_idx) as f32;
    let spread = (total_frames.max(8) as f32 / 6.0).max(1.0);
    let falloff = (1.0 - distance / spread).clamp(0.0, 1.0);
    (overlay.strength.max(0.1) * falloff).clamp(0.0, 0.85)
}

#[cfg(feature = "cuda")]
fn new_native_cuda_device(ordinal: usize) -> Result<candle_core::Device> {
    let device = candle_core::Device::new_cuda(ordinal)?;
    let cuda = device.as_cuda_device()?;
    if cuda.is_event_tracking() {
        unsafe {
            cuda.disable_event_tracking();
        }
    }
    Ok(device)
}

#[cfg(not(feature = "cuda"))]
fn new_native_cuda_device(_ordinal: usize) -> Result<candle_core::Device> {
    anyhow::bail!("CUDA backend is unavailable in this build")
}

const DISTILLED_STAGE1_SIGMAS_NO_TERMINAL: &[f32] = &[
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875,
];

const DISTILLED_STAGE2_SIGMAS_NO_TERMINAL: &[f32] = &[0.909375, 0.725, 0.421875];

fn pipeline_uses_two_stage_spatial_refinement(pipeline: PipelineKind) -> bool {
    matches!(
        pipeline,
        PipelineKind::Distilled
            | PipelineKind::TwoStage
            | PipelineKind::TwoStageHq
            | PipelineKind::IcLora
            | PipelineKind::Keyframe
            | PipelineKind::A2Vid
    )
}

fn effective_native_guidance_scale(plan: &Ltx2GeneratePlan) -> f64 {
    match plan.pipeline {
        PipelineKind::Distilled | PipelineKind::Retake => 1.0,
        _ => plan.guidance,
    }
}

fn stage_guidance_scale(plan: &Ltx2GeneratePlan, stage_index: usize) -> Result<f64> {
    Ok(match (plan.pipeline, stage_index) {
        (PipelineKind::Distilled | PipelineKind::IcLora | PipelineKind::Retake, _) => 1.0,
        (PipelineKind::TwoStage, 1)
        | (PipelineKind::TwoStageHq, 1)
        | (PipelineKind::A2Vid, 1)
        | (PipelineKind::Keyframe, 1) => 1.0,
        _ => {
            let _ = denoise_pass_plan(plan, stage_index)?;
            effective_native_guidance_scale(plan)
        }
    })
}

fn stage_sampler_mode(plan: &Ltx2GeneratePlan, stage_index: usize) -> Result<SamplerMode> {
    Ok(match (plan.pipeline, stage_index) {
        (PipelineKind::TwoStageHq, 0 | 1) => SamplerMode::Res2S,
        _ => denoise_pass_plan(plan, stage_index)?.sampler,
    })
}

fn multimodal_guider_requires_unconditional_context(params: &MultiModalGuiderParams) -> bool {
    (params.cfg_scale - 1.0).abs() > f64::EPSILON
}

fn stage_multimodal_guider_params(
    plan: &Ltx2GeneratePlan,
    stage_index: usize,
) -> Option<(MultiModalGuiderParams, MultiModalGuiderParams)> {
    match (plan.pipeline, stage_index) {
        (PipelineKind::A2Vid, 0) => {
            let stg_block = if plan.preset.name == "ltx-2.3-22b" {
                28
            } else {
                29
            };
            Some((
                MultiModalGuiderParams {
                    cfg_scale: 3.0,
                    stg_scale: 1.0,
                    stg_blocks: vec![stg_block],
                    rescale_scale: 0.7,
                    modality_scale: 3.0,
                    skip_step: 0,
                },
                MultiModalGuiderParams::default(),
            ))
        }
        (PipelineKind::TwoStage | PipelineKind::Keyframe, 0) => {
            let stg_block = if plan.preset.name == "ltx-2.3-22b" {
                28
            } else {
                29
            };
            Some((
                MultiModalGuiderParams {
                    cfg_scale: 3.0,
                    stg_scale: 1.0,
                    stg_blocks: vec![stg_block],
                    rescale_scale: 0.7,
                    modality_scale: 3.0,
                    skip_step: 0,
                },
                MultiModalGuiderParams {
                    cfg_scale: 7.0,
                    stg_scale: 1.0,
                    stg_blocks: vec![stg_block],
                    rescale_scale: 0.7,
                    modality_scale: 3.0,
                    skip_step: 0,
                },
            ))
        }
        (PipelineKind::TwoStageHq, 0) => Some((
            MultiModalGuiderParams {
                cfg_scale: 3.0,
                stg_scale: 0.0,
                stg_blocks: Vec::new(),
                rescale_scale: 0.45,
                modality_scale: 3.0,
                skip_step: 0,
            },
            MultiModalGuiderParams {
                cfg_scale: 7.0,
                stg_scale: 0.0,
                stg_blocks: Vec::new(),
                rescale_scale: 1.0,
                modality_scale: 3.0,
                skip_step: 0,
            },
        )),
        _ => None,
    }
}

fn prompt_requires_unconditional_context(plan: &Ltx2GeneratePlan) -> Result<bool> {
    if ltx_debug_enabled() || ltx_debug_compare_uncond_enabled() {
        return Ok(true);
    }
    prompt_requires_unconditional_context_for_plan(plan)
}

fn prompt_requires_unconditional_context_for_plan(plan: &Ltx2GeneratePlan) -> Result<bool> {
    for stage_index in 0..plan.execution_graph.denoise_passes.len() {
        if stage_requires_unconditional_context(plan, stage_index)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn stage_requires_unconditional_context(
    plan: &Ltx2GeneratePlan,
    stage_index: usize,
) -> Result<bool> {
    if stage_guidance_scale(plan, stage_index)? > 1.0 {
        return Ok(true);
    }
    Ok(
        stage_multimodal_guider_params(plan, stage_index).is_some_and(
            |(video_params, audio_params)| {
                multimodal_guider_requires_unconditional_context(&video_params)
                    || multimodal_guider_requires_unconditional_context(&audio_params)
            },
        ),
    )
}

fn stage_distilled_lora_scale(plan: &Ltx2GeneratePlan, stage_index: usize) -> Result<Option<f64>> {
    let pass = denoise_pass_plan(plan, stage_index)?;
    Ok(match (plan.pipeline, stage_index) {
        (PipelineKind::TwoStageHq, 0) => Some(0.25),
        (PipelineKind::TwoStageHq, 1) => Some(0.5),
        _ if pass.apply_distilled_lora && !plan.checkpoint_is_distilled => Some(1.0),
        _ => None,
    })
}

fn supports_real_video_path(plan: &Ltx2GeneratePlan) -> bool {
    let native_plain_or_image_conditioning = plan.conditioning.audio_path.is_none()
        && plan.conditioning.video_path.is_none()
        && !plan.execution_graph.uses_audio_conditioning
        && !plan.execution_graph.uses_reference_video_conditioning
        && !plan.execution_graph.uses_retake_masking
        && plan.loras.is_empty();
    let native_audio_conditioning = plan.conditioning.audio_path.is_some()
        && plan.conditioning.video_path.is_none()
        && plan.execution_graph.uses_audio_conditioning
        && !plan.execution_graph.uses_reference_video_conditioning
        && !plan.execution_graph.uses_retake_masking
        && plan.loras.is_empty()
        && plan.spatial_upscale.is_none();
    let native_retake = plan.conditioning.video_path.is_some()
        && plan.execution_graph.uses_retake_masking
        && plan.loras.is_empty()
        && plan.spatial_upscale.is_none()
        && plan.temporal_upscale.is_none();
    let native_ic_lora = plan.conditioning.audio_path.is_none()
        && plan.conditioning.video_path.is_some()
        && plan.execution_graph.uses_reference_video_conditioning
        && !plan.execution_graph.uses_audio_conditioning
        && !plan.execution_graph.uses_retake_masking
        && !plan.loras.is_empty()
        && plan.spatial_upscale.is_none();
    match plan.pipeline {
        PipelineKind::Distilled => native_plain_or_image_conditioning,
        PipelineKind::OneStage => {
            native_plain_or_image_conditioning
                && plan.spatial_upscale.is_none()
                && plan.temporal_upscale.is_none()
        }
        PipelineKind::TwoStage | PipelineKind::TwoStageHq | PipelineKind::Keyframe => {
            native_plain_or_image_conditioning
        }
        PipelineKind::A2Vid => native_audio_conditioning,
        PipelineKind::IcLora => native_ic_lora,
        PipelineKind::Retake => native_retake,
    }
}

fn denoise_pass_plan(
    plan: &Ltx2GeneratePlan,
    stage_index: usize,
) -> Result<&crate::ltx2::execution::DenoisePassPlan> {
    plan.execution_graph
        .denoise_passes
        .get(stage_index)
        .with_context(|| {
            format!(
                "missing LTX-2 denoise pass plan for stage {}",
                stage_index + 1
            )
        })
}

fn stage_lora_stack(plan: &Ltx2GeneratePlan, stage_index: usize) -> Result<Vec<LoraWeight>> {
    if matches!(plan.pipeline, PipelineKind::IcLora) && stage_index > 0 {
        return Ok(Vec::new());
    }
    let mut loras = plan.loras.clone();
    if let Some(scale) = stage_distilled_lora_scale(plan, stage_index)? {
        let path = plan
            .distilled_lora_path
            .clone()
            .context("native LTX-2 two-stage runtime requires a distilled LoRA asset")?;
        loras.push(LoraWeight { path, scale });
    }
    Ok(loras)
}

fn stage_sigmas_no_terminal(
    plan: &Ltx2GeneratePlan,
    stage_index: usize,
    device: &candle_core::Device,
) -> Result<Vec<f32>> {
    let pass = denoise_pass_plan(plan, stage_index)?;
    if stage_index == 1
        && matches!(
            plan.pipeline,
            PipelineKind::TwoStage | PipelineKind::TwoStageHq
        )
    {
        return Ok(DISTILLED_STAGE2_SIGMAS_NO_TERMINAL.to_vec());
    }
    if pass.uses_distilled_checkpoint {
        return Ok(match stage_index {
            0 => DISTILLED_STAGE1_SIGMAS_NO_TERMINAL.to_vec(),
            1 => DISTILLED_STAGE2_SIGMAS_NO_TERMINAL.to_vec(),
            _ => anyhow::bail!("unsupported distilled denoise stage {}", stage_index + 1),
        });
    }

    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(ltx2_scheduler_config())?;
    scheduler.set_timesteps(
        Some(plan.num_inference_steps as usize),
        device,
        None,
        None,
        None,
    )?;
    let sigmas = scheduler.sigmas().to_device(&candle_core::Device::Cpu)?;
    let sigmas = sigmas.to_vec1::<f32>()?;
    Ok(sigmas[..sigmas.len().saturating_sub(1)].to_vec())
}

fn video_latent_shape_from_tensor(latents: &Tensor) -> Result<VideoLatentShape> {
    let (batch, channels, frames, height, width) = latents.dims5()?;
    Ok(VideoLatentShape {
        batch,
        channels,
        frames,
        height,
        width,
    })
}

fn pixel_shape_for_video_latents(latent_shape: VideoLatentShape, fps: u32) -> VideoPixelShape {
    let pixel_shape = latent_shape.upscale(SpatioTemporalScaleFactors::default());
    VideoPixelShape {
        batch: pixel_shape.batch,
        frames: pixel_shape.frames,
        height: pixel_shape.height,
        width: pixel_shape.width,
        fps: fps as f32,
    }
}

#[allow(dead_code)]
fn source_image_only_conditioning(plan: &Ltx2GeneratePlan) -> bool {
    matches!(plan.conditioning.images.as_slice(), [image] if image.frame == 0)
        && !plan.execution_graph.uses_keyframe_conditioning
}

#[allow(dead_code)]
fn keyframe_only_conditioning(plan: &Ltx2GeneratePlan) -> bool {
    !plan.conditioning.images.is_empty()
        && plan.conditioning.images.iter().all(|image| image.frame > 0)
        && plan.execution_graph.uses_keyframe_conditioning
}

fn offset_video_time_positions(pixel_coords: &Tensor, frame_offset: u32) -> Result<Tensor> {
    let temporal = pixel_coords
        .i((.., 0..1, .., ..))?
        .affine(1.0, frame_offset as f64)?;
    let height_width = pixel_coords.i((.., 1.., .., ..))?;
    Tensor::cat(&[temporal, height_width], 1).map_err(Into::into)
}

fn scale_video_spatial_positions(positions: &Tensor, factor: usize) -> Result<Tensor> {
    if factor == 1 {
        return Ok(positions.clone());
    }
    let temporal = positions.i((.., 0..1, .., ..))?;
    let height = positions
        .i((.., 1..2, .., ..))?
        .affine(factor as f64, 0.0)?;
    let width = positions
        .i((.., 2..3, .., ..))?
        .affine(factor as f64, 0.0)?;
    Tensor::cat(&[temporal, height, width], 1).map_err(Into::into)
}

fn append_condition_from_video_latents(
    latents: &Tensor,
    pixel_shape: VideoPixelShape,
    frame_offset: u32,
    spatial_position_scale: usize,
    strength: f64,
) -> Result<VideoTokenAppendCondition> {
    let patchifier = VideoLatentPatchifier::new(1);
    let tokens = patchifier.patchify(&latents.to_dtype(DType::F32)?)?;
    let latent_shape = video_latent_shape_from_tensor(latents)?;
    let latent_coords = patchifier.get_patch_grid_bounds(latent_shape, latents.device())?;
    let pixel_coords =
        get_pixel_coords(&latent_coords, SpatioTemporalScaleFactors::default(), true)?;
    let positions = scale_video_spatial_positions(
        &scale_video_time_to_seconds(
            &offset_video_time_positions(&pixel_coords, frame_offset)?,
            pixel_shape.fps,
        )?,
        spatial_position_scale,
    )?
    .to_dtype(DType::F32)?;
    Ok(VideoTokenAppendCondition {
        tokens,
        positions,
        strength,
    })
}

fn maybe_load_stage_video_conditioning(
    plan: &Ltx2GeneratePlan,
    pixel_shape: VideoPixelShape,
    device: &candle_core::Device,
    dtype: DType,
    include_reference_video: bool,
) -> Result<StageVideoConditioning> {
    if plan.conditioning.images.is_empty()
        && plan.conditioning.latents.is_empty()
        && !include_reference_video
    {
        return Ok(StageVideoConditioning::default());
    }

    // The VAE is needed for staged images, reference video ingest, and —
    // on chain continuations — re-encoding the emitting stage's trailing
    // RGB frames into a proper-slot-semantics conditioning latent. Every
    // StagedLatent now carries RGB frames, so any non-empty
    // plan.conditioning.latents implies a VAE load.
    let need_vae = !plan.conditioning.images.is_empty()
        || include_reference_video
        || !plan.conditioning.latents.is_empty();
    let mut vae = if need_vae {
        let mut loaded = load_ltx2_video_vae(plan, device, dtype)?;
        loaded.use_tiling = false;
        loaded.use_framewise_decoding = false;
        Some(loaded)
    } else {
        None
    };

    let patchifier = VideoLatentPatchifier::new(1);
    let mut conditioning = StageVideoConditioning::default();
    for image in &plan.conditioning.images {
        let vae = vae.as_mut().expect(
            "need_vae guarantees the VAE is loaded whenever plan.conditioning.images is non-empty",
        );
        let bytes = std::fs::read(&image.path).with_context(|| {
            format!(
                "failed to read staged LTX-2 conditioning image '{}'",
                image.path
            )
        })?;
        let decoded = decode_source_image(
            &bytes,
            pixel_shape.width as u32,
            pixel_shape.height as u32,
            NormalizeRange::MinusOneToOne,
            device,
            dtype,
        )?;
        let video = decoded.unsqueeze(2)?;
        let latents = vae.encode(&video).with_context(|| {
            format!(
                "failed to encode native LTX-2 conditioning image '{}'",
                image.path
            )
        })?;
        let tokens = patchifier.patchify(&latents.to_dtype(DType::F32)?)?;
        let use_guiding_latent = matches!(plan.pipeline, PipelineKind::Keyframe);
        if image.frame == 0 && !use_guiding_latent {
            conditioning.replacements.push(VideoTokenReplacement {
                start_token: 0,
                tokens,
                strength: image.strength as f64,
            });
        } else {
            conditioning
                .appended
                .push(append_condition_from_video_latents(
                    &latents,
                    pixel_shape,
                    image.frame,
                    1,
                    image.strength as f64,
                )?);
        }
    }
    // Chain carryover: every StagedLatent is a contiguous RGB window from
    // the end of the emitting stage. Re-encoding on the receiving side
    // (rather than slicing the emitting stage's final latent tensor) keeps
    // slot semantics aligned with the receiving clip's time axis — slot 0
    // is a proper causal 1-pixel encoding, slot 1+ are proper 8-pixel
    // continuation encodings, with no ambiguity about which latent slot
    // corresponds to which pixel-frame range.
    for staged in &plan.conditioning.latents {
        if staged.tail_rgb_frames.is_empty() {
            anyhow::bail!(
                "StagedLatent has an empty tail_rgb_frames; at least one frame is required"
            );
        }
        let vae = vae.as_mut().expect(
            "need_vae guarantees the VAE is loaded whenever plan.conditioning.latents is non-empty",
        );
        let video = video_tensor_from_frames(&staged.tail_rgb_frames, device, dtype)
            .context("encode chain tail RGB frames into pixel tensor for carryover")?;
        let latents = vae
            .encode(&video)
            .context("failed to encode chain tail RGB frames through the LTX-2 video VAE")?
            .to_dtype(DType::F32)?;
        let use_guiding_latent = matches!(plan.pipeline, PipelineKind::Keyframe);
        if staged.frame == 0 && !use_guiding_latent {
            let tokens = patchifier.patchify(&latents)?;
            conditioning.replacements.push(VideoTokenReplacement {
                start_token: 0,
                tokens,
                strength: staged.strength as f64,
            });
        } else {
            conditioning
                .appended
                .push(append_condition_from_video_latents(
                    &latents,
                    pixel_shape,
                    staged.frame,
                    1,
                    staged.strength as f64,
                )?);
        }
    }
    if include_reference_video {
        let vae = vae.as_mut().expect(
            "need_vae guarantees the VAE is loaded whenever include_reference_video is true",
        );
        let video_path = plan.conditioning.video_path.as_ref().with_context(|| {
            format!(
                "native {:?} stage requested reference video conditioning without a staged source_video",
                plan.pipeline
            )
        })?;
        let reference_downscale_factor = lora::reference_video_downscale_factor(&plan.loras)?;
        if !pixel_shape.width.is_multiple_of(reference_downscale_factor)
            || !pixel_shape
                .height
                .is_multiple_of(reference_downscale_factor)
        {
            anyhow::bail!(
                "native LTX-2 IC-LoRA output dimensions ({}x{}) must be divisible by reference_downscale_factor ({reference_downscale_factor})",
                pixel_shape.width,
                pixel_shape.height
            );
        }
        let ref_width = pixel_shape.width / reference_downscale_factor;
        let ref_height = pixel_shape.height / reference_downscale_factor;
        let (_metadata, mut frames) = media::decode_video_frames(Path::new(video_path))?;
        if frames.len() > pixel_shape.frames {
            frames.truncate(pixel_shape.frames);
        }
        let resized = frames
            .into_iter()
            .map(|frame| {
                if frame.width() == ref_width as u32 && frame.height() == ref_height as u32 {
                    frame
                } else {
                    imageops::resize(
                        &frame,
                        ref_width as u32,
                        ref_height as u32,
                        imageops::FilterType::Lanczos3,
                    )
                }
            })
            .collect::<Vec<_>>();
        let video = video_tensor_from_frames(&resized, device, dtype)?;
        let latents = vae.encode(&video).with_context(|| {
            format!(
                "failed to encode native LTX-2 IC-LoRA reference video '{}'",
                video_path
            )
        })?;
        conditioning
            .appended
            .push(append_condition_from_video_latents(
                &latents,
                pixel_shape,
                0,
                reference_downscale_factor,
                1.0,
            )?);
    }
    drop(vae);
    if device.is_cuda() {
        device.synchronize()?;
    }
    Ok(conditioning)
}

fn apply_video_token_replacements(
    video_latents: &Tensor,
    replacements: &[VideoTokenReplacement],
) -> Result<Tensor> {
    let mut patched = video_latents.clone();
    for replacement in replacements {
        let total_tokens = patched.dim(1)?;
        let replacement_tokens = replacement
            .tokens
            .to_device(patched.device())?
            .to_dtype(patched.dtype())?;
        let count = replacement_tokens.dim(1)?;
        if replacement.start_token + count > total_tokens {
            anyhow::bail!(
                "conditioning replacement exceeds video token count: start={} count={} total={total_tokens}",
                replacement.start_token,
                count
            );
        }
        let current = patched.narrow(1, replacement.start_token, count)?;
        let blended = if replacement.strength <= 0.0 {
            current
        } else if replacement.strength >= 1.0 {
            replacement_tokens
        } else {
            current
                .affine(1.0 - replacement.strength, 0.0)?
                .broadcast_add(&replacement_tokens.affine(replacement.strength, 0.0)?)?
        };
        let mut parts = Vec::with_capacity(3);
        if replacement.start_token != 0 {
            parts.push(patched.narrow(1, 0, replacement.start_token)?);
        }
        parts.push(blended);
        let end = replacement.start_token + count;
        if end < total_tokens {
            parts.push(patched.narrow(1, end, total_tokens - end)?);
        }
        let refs = parts.iter().collect::<Vec<_>>();
        patched = Tensor::cat(&refs, 1)?;
    }
    Ok(patched)
}

/// Build the "clean reference" tensor used by the denoise mask blend at every
/// step. For replacement-based conditioning (e.g. i2v source image) with
/// `strength < 1.0`, `video_latents` already holds `noise*(1-s) + source*s` at
/// the replacement positions. If we reuse that as the clean target, the
/// denoise-mask blend pulls those tokens toward a noisy ghost of the image at
/// every step — the first latent frame never converges to the pure source.
///
/// Re-applying the replacements with strength 1.0 overwrites those positions
/// with the pure source tokens, leaving appended keyframe tokens (already
/// full-strength in `apply_appended_video_conditioning`) and pure-noise
/// regions untouched.
fn clean_latents_for_conditioning(
    video_latents: &Tensor,
    conditioning: &StageVideoConditioning,
) -> Result<Tensor> {
    if conditioning.replacements.is_empty() {
        return Ok(video_latents.clone());
    }
    let hard_replacements: Vec<VideoTokenReplacement> = conditioning
        .replacements
        .iter()
        .map(|replacement| VideoTokenReplacement {
            start_token: replacement.start_token,
            tokens: replacement.tokens.clone(),
            strength: 1.0,
        })
        .collect();
    apply_video_token_replacements(video_latents, &hard_replacements)
}

fn apply_appended_video_conditioning(
    video_latents: &Tensor,
    video_positions: &Tensor,
    appended: &[VideoTokenAppendCondition],
) -> Result<(Tensor, Tensor)> {
    if appended.is_empty() {
        return Ok((video_latents.clone(), video_positions.clone()));
    }

    let mut token_parts = vec![video_latents.clone()];
    let mut position_parts = vec![video_positions.clone()];
    for condition in appended {
        let tokens = if condition.strength <= 0.0 {
            condition
                .tokens
                .zeros_like()?
                .to_device(video_latents.device())?
                .to_dtype(video_latents.dtype())?
        } else {
            condition
                .tokens
                .to_device(video_latents.device())?
                .to_dtype(video_latents.dtype())?
        };
        token_parts.push(tokens);
        position_parts.push(
            condition
                .positions
                .to_device(video_positions.device())?
                .to_dtype(video_positions.dtype())?,
        );
    }
    let token_refs = token_parts.iter().collect::<Vec<_>>();
    let position_refs = position_parts.iter().collect::<Vec<_>>();
    Ok((
        Tensor::cat(&token_refs, 1)?,
        Tensor::cat(&position_refs, 2)?,
    ))
}

fn apply_stage_video_conditioning(
    video_latents: &Tensor,
    video_positions: &Tensor,
    conditioning: &StageVideoConditioning,
) -> Result<(Tensor, Tensor)> {
    let replaced = apply_video_token_replacements(video_latents, &conditioning.replacements)?;
    apply_appended_video_conditioning(&replaced, video_positions, &conditioning.appended)
}

fn reapply_stage_video_conditioning(
    video_latents: &Tensor,
    base_token_count: usize,
    conditioning: &StageVideoConditioning,
) -> Result<Tensor> {
    let total_tokens = video_latents.dim(1)?;
    if total_tokens < base_token_count {
        anyhow::bail!(
            "video token count ({total_tokens}) is smaller than base token count ({base_token_count})"
        );
    }

    let base = video_latents.narrow(1, 0, base_token_count)?;
    let hard_replacements = conditioning
        .replacements
        .iter()
        .filter(|replacement| replacement.strength >= 1.0)
        .cloned()
        .collect::<Vec<_>>();
    let base = apply_video_token_replacements(&base, &hard_replacements)?;
    if conditioning.appended.is_empty() {
        return Ok(base);
    }

    let mut parts = vec![base];
    for condition in &conditioning.appended {
        // Appended conditioning tokens must remain present for the whole
        // denoise loop. Their strength is expressed via the denoise mask;
        // dropping "soft" appended tokens here desynchronizes the token
        // count from the cached clean latents and mask tensors.
        parts.push(
            condition
                .tokens
                .to_device(video_latents.device())?
                .to_dtype(video_latents.dtype())?,
        );
    }
    let refs = parts.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 1).map_err(Into::into)
}

fn strip_appended_video_conditioning(
    video_latents: &Tensor,
    base_token_count: usize,
) -> Result<Tensor> {
    let total_tokens = video_latents.dim(1)?;
    if total_tokens < base_token_count {
        anyhow::bail!(
            "video token count ({total_tokens}) is smaller than base token count ({base_token_count})"
        );
    }
    if total_tokens == base_token_count {
        return Ok(video_latents.clone());
    }
    video_latents
        .narrow(1, 0, base_token_count)
        .map_err(Into::into)
}

fn build_video_conditioning_denoise_mask(
    base_token_count: usize,
    conditioning: &StageVideoConditioning,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut values = vec![1.0f32; base_token_count];
    for replacement in &conditioning.replacements {
        let count = replacement.tokens.dim(1)?;
        let end = replacement.start_token + count;
        if end > base_token_count {
            anyhow::bail!(
                "conditioning replacement exceeds base token count: start={} count={} total={base_token_count}",
                replacement.start_token,
                count
            );
        }
        values[replacement.start_token..end].fill((1.0 - replacement.strength) as f32);
    }
    for condition in &conditioning.appended {
        values.extend(std::iter::repeat_n(
            (1.0 - condition.strength) as f32,
            condition.tokens.dim(1)?,
        ));
    }
    Tensor::from_vec(values.clone(), (1, values.len()), device).map_err(Into::into)
}

fn append_conditioning_attention_mask(
    existing_mask: Option<&Tensor>,
    num_noisy_tokens: usize,
    num_existing_tokens: usize,
    num_new_tokens: usize,
    batch_size: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let top_left = match existing_mask {
        Some(mask) => mask.to_device(device)?.to_dtype(DType::F32)?,
        None => Tensor::ones(
            (batch_size, num_existing_tokens, num_existing_tokens),
            DType::F32,
            device,
        )?,
    };
    let previous_ref_tokens = num_existing_tokens.saturating_sub(num_noisy_tokens);
    let noisy_to_new = Tensor::ones(
        (batch_size, num_noisy_tokens, num_new_tokens),
        DType::F32,
        device,
    )?;
    let prev_ref_to_new = Tensor::zeros(
        (batch_size, previous_ref_tokens, num_new_tokens),
        DType::F32,
        device,
    )?;
    let top_right = Tensor::cat(&[&noisy_to_new, &prev_ref_to_new], 1)?;

    let new_to_noisy = Tensor::ones(
        (batch_size, num_new_tokens, num_noisy_tokens),
        DType::F32,
        device,
    )?;
    let new_to_prev_ref = Tensor::zeros(
        (batch_size, num_new_tokens, previous_ref_tokens),
        DType::F32,
        device,
    )?;
    let bottom_left = Tensor::cat(&[&new_to_noisy, &new_to_prev_ref], 2)?;
    let bottom_right = Tensor::ones(
        (batch_size, num_new_tokens, num_new_tokens),
        DType::F32,
        device,
    )?;

    let top = Tensor::cat(&[&top_left, &top_right], 2)?;
    let bottom = Tensor::cat(&[&bottom_left, &bottom_right], 2)?;
    Tensor::cat(&[&top, &bottom], 1).map_err(Into::into)
}

fn build_video_conditioning_self_attention_mask(
    base_token_count: usize,
    conditioning: &StageVideoConditioning,
    device: &candle_core::Device,
) -> Result<Option<Tensor>> {
    if conditioning.appended.is_empty() {
        return Ok(None);
    }
    let batch_size = conditioning
        .appended
        .first()
        .context("appended conditioning unexpectedly empty")?
        .tokens
        .dim(0)?;
    let mut existing_mask = None;
    let mut existing_tokens = base_token_count;
    for condition in &conditioning.appended {
        existing_mask = Some(append_conditioning_attention_mask(
            existing_mask.as_ref(),
            base_token_count,
            existing_tokens,
            condition.tokens.dim(1)?,
            batch_size,
            device,
        )?);
        existing_tokens += condition.tokens.dim(1)?;
    }
    Ok(existing_mask)
}

fn maybe_apply_temporal_upsampler(
    plan: &Ltx2GeneratePlan,
    latents: &Tensor,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    if plan.temporal_upscale.is_none() {
        return Ok(latents.clone());
    }
    let temporal_upsampler_path = plan
        .temporal_upsampler_path
        .as_ref()
        .context("native LTX-2 temporal upscaling requires a temporal upsampler asset")?;
    let latent_stats = Ltx2VaeLatentStats::load(plan, device, dtype)?;
    let upsampler = LatentUpsampler::load(Path::new(temporal_upsampler_path), dtype, device)?;
    let upsampled = latent_stats
        .normalize(&upsampler.forward(&latent_stats.denormalize(&latents.to_dtype(dtype)?)?)?)?;
    drop(upsampler);
    drop(latent_stats);
    if device.is_cuda() {
        device.synchronize()?;
    }
    Ok(upsampled)
}

fn blend_conditioned_denoised(
    denoised: &Tensor,
    clean_latents: &Tensor,
    denoise_mask: &Tensor,
) -> Result<Tensor> {
    let mask = denoise_mask
        .to_device(denoised.device())?
        .to_dtype(denoised.dtype())?;
    let mask = mask.unsqueeze(2)?;
    let clean = clean_latents
        .to_device(denoised.device())?
        .to_dtype(denoised.dtype())?;
    let inverse = Tensor::ones_like(&mask)?.broadcast_sub(&mask)?;
    denoised
        .broadcast_mul(&mask)?
        .broadcast_add(&clean.broadcast_mul(&inverse)?)
        .map_err(Into::into)
}

fn is_placeholder_checkpoint_error(err: &anyhow::Error) -> bool {
    let message = err.to_string().to_ascii_lowercase();
    message.contains("header too small")
        || message.contains("invalid header")
        || message.contains("failed to parse safetensor")
}

fn render_real_distilled_av(
    plan: &Ltx2GeneratePlan,
    prepared: &NativePreparedRun,
    device: &candle_core::Device,
    progress: Option<&ProgressCallback>,
    tail_capture: Option<&std::sync::Arc<std::sync::Mutex<Option<Tensor>>>>,
) -> Result<NativeRenderedVideo> {
    let debug_enabled = ltx_debug_enabled();
    let prompt_inputs = prepare_render_prompt_inputs(
        prepared,
        device,
        RenderPromptInputOptions {
            include_unconditional: false,
            include_alt: true,
        },
    )?;
    let audio_shape = prompt_inputs.audio_shape;
    // Upstream LTX-2 diffusion stages pass connector outputs directly as the
    // text context and leave `context_mask=None` in the transformer modality
    // wrapper. The connector has already packed padded tokens into registers
    // and zeroed masked positions, so feeding the binary mask back into text
    // cross-attention here over-constrains the prompt path and does not match
    // the published inference stack.
    let cond_mask: Option<&Tensor> = None;
    let alt_mask: Option<&Tensor> = None;
    let stage1_video_noise = seeded_randn(
        plan.seed,
        &[
            prepared.video_latent_shape.batch,
            prepared.video_latent_shape.channels,
            prepared.video_latent_shape.frames,
            prepared.video_latent_shape.height,
            prepared.video_latent_shape.width,
        ],
        device,
        DType::F32,
    )?;
    let stage1_audio_noise = match audio_shape {
        Some(audio_shape) => Some(seeded_randn(
            plan.seed ^ 0x4155_4449_4f4c_5458,
            &[
                audio_shape.batch,
                audio_shape.channels,
                audio_shape.frames,
                audio_shape.mel_bins,
            ],
            device,
            DType::F32,
        )?),
        None => None,
    };

    if debug_enabled {
        log_tensor_stats("video_context", &prompt_inputs.cond_context)?;
        if let Some(audio_context) = prompt_inputs.audio_context.as_ref() {
            log_tensor_stats("audio_context", audio_context)?;
        }
        log_tensor_stats("initial_video_latents", &stage1_video_noise)?;
        if let Some(stage1_audio_noise) = stage1_audio_noise.as_ref() {
            log_tensor_stats("initial_audio_latents", stage1_audio_noise)?;
        }
    }

    let dtype = gpu_dtype(device);
    let stage1_guidance_scale = stage_guidance_scale(plan, 0)?;
    let latent_stats = Ltx2VaeLatentStats::load(plan, device, dtype)?;
    let stage1_video_conditioning = maybe_load_stage_video_conditioning(
        plan,
        prepared.video_pixel_shape,
        device,
        dtype,
        false,
    )?;
    if debug_enabled {
        eprintln!("[ltx2-debug] loading stage1 transformer");
    }
    let stage1_transformer_load_start = Instant::now();
    let stage1_transformer = load_ltx2_av_transformer(plan, device)?;
    log_timing(
        "distilled.stage1.transformer_load",
        stage1_transformer_load_start,
    );
    if debug_enabled {
        log_debug_vram("after_stage1_transformer_load");
    }
    let stage1_denoise_start = Instant::now();
    let (stage1_video_latents, stage1_audio_latents) = run_real_distilled_stage(
        &stage1_transformer,
        prepared.video_latent_shape,
        audio_shape,
        &stage1_video_noise,
        &stage1_video_conditioning,
        None,
        stage1_audio_noise.as_ref(),
        None,
        &prompt_inputs.video_positions,
        prompt_inputs.audio_positions.as_ref(),
        &prompt_inputs.cond_context,
        None,
        prompt_inputs.alt_context.as_ref(),
        prompt_inputs.audio_context.as_ref(),
        None,
        prompt_inputs.alt_audio_context.as_ref(),
        cond_mask,
        None,
        alt_mask,
        None,
        stage1_guidance_scale,
        DISTILLED_STAGE1_SIGMAS_NO_TERMINAL,
        stage_sampler_mode(plan, 0)?,
        Some(&stage1_video_noise),
        stage1_audio_noise.as_ref(),
        None,
        None,
        Some("distilled.stage1"),
        debug_enabled.then_some("stage1"),
        progress,
    )?;
    log_timing("distilled.stage1.denoise", stage1_denoise_start);
    if debug_enabled {
        log_debug_vram("after_stage1_denoise");
    }
    drop(stage1_transformer);
    device.synchronize()?;
    if debug_enabled {
        log_debug_vram("after_stage1_transformer_drop");
    }
    if env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
        let mut debug_vae = load_ltx2_video_vae(plan, device, dtype)?;
        debug_vae.use_tiling = false;
        debug_vae.use_framewise_decoding = false;
        maybe_write_debug_stage_video(
            "stage1",
            &debug_vae,
            &stage1_video_latents,
            prepared.video_pixel_shape,
            dtype,
        )?;
        drop(debug_vae);
        device.synchronize()?;
    }
    let spatial_upsampler_path = plan
        .spatial_upsampler_path
        .as_ref()
        .context("native distilled LTX-2 inference requires a spatial upsampler asset")?;
    let stage1_upsample_start = Instant::now();
    let upsampler = LatentUpsampler::load(Path::new(spatial_upsampler_path), dtype, device)?;
    let stage2_clean_video_latents = latent_stats.normalize(
        &upsampler.forward(&latent_stats.denormalize(&stage1_video_latents.to_dtype(dtype)?)?)?,
    )?;
    drop(upsampler);
    device.synchronize()?;
    log_timing("distilled.stage1.spatial_upsample", stage1_upsample_start);
    if debug_enabled {
        log_debug_vram("after_stage1_upsample");
    }
    let requested_pixel_shape = VideoPixelShape {
        batch: 1,
        frames: plan.num_frames as usize,
        height: plan.height as usize,
        width: plan.width as usize,
        fps: plan.frame_rate as f32,
    };
    let stage2_video_latent_shape = video_latent_shape_from_tensor(&stage2_clean_video_latents)?;
    let stage2_pixel_shape =
        pixel_shape_for_video_latents(stage2_video_latent_shape, plan.frame_rate);
    let stage2_video_conditioning =
        maybe_load_stage_video_conditioning(plan, stage2_pixel_shape, device, dtype, false)?;
    if env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
        let mut debug_vae = load_ltx2_video_vae(plan, device, dtype)?;
        debug_vae.use_tiling = false;
        debug_vae.use_framewise_decoding = false;
        maybe_write_debug_stage_video(
            "stage1-upscaled",
            &debug_vae,
            &stage2_clean_video_latents,
            stage2_pixel_shape,
            dtype,
        )?;
        drop(debug_vae);
        device.synchronize()?;
    }
    let stage2_video_positions = build_video_positions(stage2_pixel_shape, device)?;
    let stage2_video_noise = seeded_randn(
        plan.seed ^ 0x5354_4147_4532_4c54,
        &[
            stage2_video_latent_shape.batch,
            stage2_video_latent_shape.channels,
            stage2_video_latent_shape.frames,
            stage2_video_latent_shape.height,
            stage2_video_latent_shape.width,
        ],
        device,
        DType::F32,
    )?;
    let stage2_audio_noise = match audio_shape {
        Some(audio_shape) => Some(seeded_randn(
            plan.seed ^ 0x4155_4449_3254_4c58,
            &[
                audio_shape.batch,
                audio_shape.channels,
                audio_shape.frames,
                audio_shape.mel_bins,
            ],
            device,
            DType::F32,
        )?),
        None => None,
    };
    let stage2_sigma = DISTILLED_STAGE2_SIGMAS_NO_TERMINAL[0];
    let stage2_video_start = mix_clean_latents_with_noise(
        &stage2_clean_video_latents.to_dtype(DType::F32)?,
        &stage2_video_noise,
        stage2_sigma,
    )?;
    let stage2_audio_start = match (stage1_audio_latents.as_ref(), stage2_audio_noise.as_ref()) {
        (Some(stage1_audio_latents), Some(stage2_audio_noise)) => {
            Some(mix_clean_latents_with_noise(
                &stage1_audio_latents.to_dtype(DType::F32)?,
                stage2_audio_noise,
                stage2_sigma,
            )?)
        }
        _ => None,
    };
    if debug_enabled {
        eprintln!("[ltx2-debug] loading stage2 transformer");
    }
    let stage2_transformer_load_start = Instant::now();
    let stage2_transformer = load_ltx2_av_transformer(plan, device)?;
    log_timing(
        "distilled.stage2.transformer_load",
        stage2_transformer_load_start,
    );
    if debug_enabled {
        log_debug_vram("after_stage2_transformer_load");
    }
    let stage2_denoise_start = Instant::now();
    let (latents, audio_latents) = run_real_distilled_stage(
        &stage2_transformer,
        stage2_video_latent_shape,
        audio_shape,
        &stage2_video_start,
        &stage2_video_conditioning,
        None,
        stage2_audio_start.as_ref(),
        None,
        &stage2_video_positions,
        prompt_inputs.audio_positions.as_ref(),
        &prompt_inputs.cond_context,
        None,
        prompt_inputs.alt_context.as_ref(),
        prompt_inputs.audio_context.as_ref(),
        None,
        prompt_inputs.alt_audio_context.as_ref(),
        cond_mask,
        None,
        alt_mask,
        None,
        stage_guidance_scale(plan, 1)?,
        DISTILLED_STAGE2_SIGMAS_NO_TERMINAL,
        stage_sampler_mode(plan, 1)?,
        Some(&stage2_video_noise),
        stage2_audio_noise.as_ref(),
        None,
        None,
        Some("distilled.stage2"),
        debug_enabled.then_some("stage2"),
        progress,
    )?;
    log_timing("distilled.stage2.denoise", stage2_denoise_start);
    if debug_enabled {
        log_debug_vram("after_stage2_denoise");
    }
    drop(stage2_transformer);
    device.synchronize()?;
    if debug_enabled {
        log_debug_vram("after_stage2_transformer_drop");
    }
    let latents = maybe_apply_temporal_upsampler(plan, &latents, device, dtype)?;
    if debug_enabled && plan.temporal_upscale.is_some() {
        log_debug_vram("after_temporal_upsample");
    }
    if debug_enabled {
        log_tensor_stats("final_video_latents", &latents)?;
    }
    let mut vae = load_ltx2_video_vae(plan, device, dtype)?;
    vae.use_tiling = false;
    vae.use_framewise_decoding = false;
    let decode_start = Instant::now();
    // Chain-stage hook: capture the pre-decode F32 latents so
    // `Ltx2Engine::render_chain_stage` can narrow the tail off for the next
    // stage's conditioning. Cheap shallow clone (candle tensors are
    // Arc-backed). A poisoned mutex is ignored here — the outer caller
    // detects an empty slot and emits a clear error.
    if let Some(slot) = tail_capture {
        if let Ok(mut guard) = slot.lock() {
            *guard = Some(latents.clone());
        }
    }
    let (_dec_output, video) = vae.decode(&latents.to_dtype(dtype)?, None, false, false)?;
    if debug_enabled {
        log_tensor_stats("decoded_video", &video)?;
    }
    let frames = decoded_video_to_frames(&video, requested_pixel_shape)?;
    if device.is_cuda() {
        device.synchronize()?;
    }
    drop(video);
    drop(vae);
    log_timing("distilled.decode_video", decode_start);
    let audio_render_start = Instant::now();
    let audio_track = maybe_render_native_audio_track(plan, audio_latents.as_ref(), device, dtype)?;
    log_timing("distilled.render_audio", audio_render_start);
    drop(latents);
    drop(audio_latents);
    drop(stage2_audio_start);
    drop(stage2_video_start);
    drop(stage2_audio_noise);
    drop(stage2_video_noise);
    drop(stage2_video_positions);
    drop(stage2_clean_video_latents);
    drop(stage1_audio_latents);
    drop(stage1_video_latents);
    drop(stage1_audio_noise);
    drop(stage1_video_noise);
    let _ = cond_mask;
    let _ = alt_mask;
    drop(prompt_inputs);
    drop(latent_stats);
    if device.is_cuda() {
        device.synchronize()?;
    }

    let has_audio = audio_track.is_some();
    let audio_sample_rate = audio_track.as_ref().map(|track| track.sample_rate);
    let audio_channels = audio_track.as_ref().map(|track| u32::from(track.channels));

    Ok(NativeRenderedVideo {
        frames,
        audio_track,
        has_audio,
        audio_sample_rate,
        audio_channels,
    })
}

fn render_real_two_stage_av(
    plan: &Ltx2GeneratePlan,
    prepared: &NativePreparedRun,
    device: &candle_core::Device,
    progress: Option<&ProgressCallback>,
) -> Result<NativeRenderedVideo> {
    let debug_enabled = ltx_debug_enabled();
    let prompt_inputs = prepare_render_prompt_inputs(
        prepared,
        device,
        RenderPromptInputOptions {
            include_unconditional: true,
            include_alt: true,
        },
    )?;
    let audio_shape = prompt_inputs.audio_shape;
    let cond_mask: Option<&Tensor> = None;
    let uncond_mask: Option<&Tensor> = None;
    let alt_mask: Option<&Tensor> = None;
    let stage1_video_noise = seeded_randn(
        plan.seed,
        &[
            prepared.video_latent_shape.batch,
            prepared.video_latent_shape.channels,
            prepared.video_latent_shape.frames,
            prepared.video_latent_shape.height,
            prepared.video_latent_shape.width,
        ],
        device,
        DType::F32,
    )?;
    let dtype = gpu_dtype(device);
    let conditioned_audio = maybe_load_native_conditioning_audio(plan, audio_shape, device, dtype)?;
    let frozen_audio_denoise_mask = conditioned_audio
        .as_ref()
        .map(|_| {
            build_frozen_audio_denoise_mask(
                audio_shape.context("frozen audio conditioning requires an audio latent shape")?,
                device,
            )
        })
        .transpose()?;
    let stage1_audio_noise = if conditioned_audio.is_some() {
        None
    } else {
        match audio_shape {
            Some(audio_shape) => Some(seeded_randn(
                plan.seed ^ 0x4155_4449_4f4c_5458,
                &[
                    audio_shape.batch,
                    audio_shape.channels,
                    audio_shape.frames,
                    audio_shape.mel_bins,
                ],
                device,
                DType::F32,
            )?),
            None => None,
        }
    };
    let stage1_guidance_scale = stage_guidance_scale(plan, 0)?;
    let latent_stats = Ltx2VaeLatentStats::load(plan, device, dtype)?;
    let stage1_sigmas = stage_sigmas_no_terminal(plan, 0, device)?;
    let stage1_sampler = stage_sampler_mode(plan, 0)?;
    let stage1_loras = stage_lora_stack(plan, 0)?;
    let stage1_video_conditioning = maybe_load_stage_video_conditioning(
        plan,
        prepared.video_pixel_shape,
        device,
        dtype,
        matches!(plan.pipeline, PipelineKind::IcLora),
    )?;
    if debug_enabled {
        eprintln!("[ltx2-debug] loading stage1 transformer");
    }
    let stage1_transformer_load_start = Instant::now();
    let stage1_transformer = load_ltx2_av_transformer_with_loras(plan, device, &stage1_loras)?;
    log_timing(
        "two_stage.stage1.transformer_load",
        stage1_transformer_load_start,
    );
    let stage1_audio_start = conditioned_audio
        .as_ref()
        .map(|audio| &audio.latents)
        .or(stage1_audio_noise.as_ref());
    let stage1_denoise_start = Instant::now();
    let stage1_requires_uncond = stage_requires_unconditional_context(plan, 0)?;
    let (stage1_video_latents, stage1_audio_latents) = run_real_distilled_stage(
        &stage1_transformer,
        prepared.video_latent_shape,
        audio_shape,
        &stage1_video_noise,
        &stage1_video_conditioning,
        None,
        stage1_audio_start,
        None,
        &prompt_inputs.video_positions,
        prompt_inputs.audio_positions.as_ref(),
        &prompt_inputs.cond_context,
        stage1_requires_uncond
            .then_some(prompt_inputs.uncond_context.as_ref())
            .flatten(),
        prompt_inputs.alt_context.as_ref(),
        prompt_inputs.audio_context.as_ref(),
        stage1_requires_uncond
            .then_some(prompt_inputs.uncond_audio_context.as_ref())
            .flatten(),
        prompt_inputs.alt_audio_context.as_ref(),
        cond_mask,
        if stage1_requires_uncond {
            uncond_mask
        } else {
            None
        },
        alt_mask,
        stage_multimodal_guider_params(plan, 0),
        stage1_guidance_scale,
        &stage1_sigmas,
        stage1_sampler,
        Some(&stage1_video_noise),
        stage1_audio_noise.as_ref(),
        None,
        frozen_audio_denoise_mask.as_ref(),
        Some("two_stage.stage1"),
        debug_enabled.then_some("stage1"),
        progress,
    )?;
    log_timing("two_stage.stage1.denoise", stage1_denoise_start);
    drop(stage1_transformer);
    device.synchronize()?;
    if env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
        let mut debug_vae = load_ltx2_video_vae(plan, device, dtype)?;
        debug_vae.use_tiling = false;
        debug_vae.use_framewise_decoding = false;
        maybe_write_debug_stage_video(
            "stage1",
            &debug_vae,
            &stage1_video_latents,
            prepared.video_pixel_shape,
            dtype,
        )?;
        drop(debug_vae);
        device.synchronize()?;
    }

    let spatial_upsampler_path = plan
        .spatial_upsampler_path
        .as_ref()
        .context("native LTX-2 two-stage inference requires a spatial upsampler asset")?;
    let stage1_upsample_start = Instant::now();
    let upsampler = LatentUpsampler::load(Path::new(spatial_upsampler_path), dtype, device)?;
    let stage2_clean_video_latents = latent_stats.normalize(
        &upsampler.forward(&latent_stats.denormalize(&stage1_video_latents.to_dtype(dtype)?)?)?,
    )?;
    drop(upsampler);
    device.synchronize()?;
    log_timing("two_stage.stage1.spatial_upsample", stage1_upsample_start);

    let requested_pixel_shape = VideoPixelShape {
        batch: 1,
        frames: plan.num_frames as usize,
        height: plan.height as usize,
        width: plan.width as usize,
        fps: plan.frame_rate as f32,
    };
    let stage2_video_latent_shape = video_latent_shape_from_tensor(&stage2_clean_video_latents)?;
    let stage2_pixel_shape =
        pixel_shape_for_video_latents(stage2_video_latent_shape, plan.frame_rate);
    let stage2_video_conditioning =
        maybe_load_stage_video_conditioning(plan, stage2_pixel_shape, device, dtype, false)?;
    if env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
        let mut debug_vae = load_ltx2_video_vae(plan, device, dtype)?;
        debug_vae.use_tiling = false;
        debug_vae.use_framewise_decoding = false;
        maybe_write_debug_stage_video(
            "stage1-upscaled",
            &debug_vae,
            &stage2_clean_video_latents,
            stage2_pixel_shape,
            dtype,
        )?;
        drop(debug_vae);
        device.synchronize()?;
    }
    let stage2_video_positions = build_video_positions(stage2_pixel_shape, device)?;
    let stage2_video_noise = seeded_randn(
        plan.seed ^ 0x5354_4147_4532_4c54,
        &[
            stage2_video_latent_shape.batch,
            stage2_video_latent_shape.channels,
            stage2_video_latent_shape.frames,
            stage2_video_latent_shape.height,
            stage2_video_latent_shape.width,
        ],
        device,
        DType::F32,
    )?;
    let stage2_audio_noise = match audio_shape {
        Some(audio_shape) => Some(seeded_randn(
            plan.seed ^ 0x4155_4449_3254_4c58,
            &[
                audio_shape.batch,
                audio_shape.channels,
                audio_shape.frames,
                audio_shape.mel_bins,
            ],
            device,
            DType::F32,
        )?),
        None => None,
    };
    let stage2_sigmas = stage_sigmas_no_terminal(plan, 1, device)?;
    let stage2_sigma = *stage2_sigmas
        .first()
        .context("stage2 sigma schedule must contain at least one step")?;
    let stage2_video_start = mix_clean_latents_with_noise(
        &stage2_clean_video_latents.to_dtype(DType::F32)?,
        &stage2_video_noise,
        stage2_sigma,
    )?;
    let stage2_audio_start = match (stage1_audio_latents.as_ref(), stage2_audio_noise.as_ref()) {
        (Some(stage1_audio_latents), Some(stage2_audio_noise)) => {
            Some(mix_clean_latents_with_noise(
                &stage1_audio_latents.to_dtype(DType::F32)?,
                stage2_audio_noise,
                stage2_sigma,
            )?)
        }
        _ => None,
    };
    let stage2_sampler = stage_sampler_mode(plan, 1)?;
    let stage2_loras = stage_lora_stack(plan, 1)?;
    let stage2_guidance_scale = stage_guidance_scale(plan, 1)?;
    if debug_enabled {
        eprintln!("[ltx2-debug] loading stage2 transformer");
    }
    let stage2_transformer_load_start = Instant::now();
    let stage2_transformer = load_ltx2_av_transformer_with_loras(plan, device, &stage2_loras)?;
    log_timing(
        "two_stage.stage2.transformer_load",
        stage2_transformer_load_start,
    );
    let stage2_denoise_start = Instant::now();
    let stage2_requires_uncond = stage_requires_unconditional_context(plan, 1)?;
    let (latents, audio_latents) = run_real_distilled_stage(
        &stage2_transformer,
        stage2_video_latent_shape,
        audio_shape,
        &stage2_video_start,
        &stage2_video_conditioning,
        None,
        stage2_audio_start.as_ref(),
        None,
        &stage2_video_positions,
        prompt_inputs.audio_positions.as_ref(),
        &prompt_inputs.cond_context,
        stage2_requires_uncond
            .then_some(prompt_inputs.uncond_context.as_ref())
            .flatten(),
        prompt_inputs.alt_context.as_ref(),
        prompt_inputs.audio_context.as_ref(),
        stage2_requires_uncond
            .then_some(prompt_inputs.uncond_audio_context.as_ref())
            .flatten(),
        prompt_inputs.alt_audio_context.as_ref(),
        cond_mask,
        if stage2_requires_uncond {
            uncond_mask
        } else {
            None
        },
        alt_mask,
        stage_multimodal_guider_params(plan, 1),
        stage2_guidance_scale,
        &stage2_sigmas,
        stage2_sampler,
        Some(&stage2_video_noise),
        stage2_audio_noise.as_ref(),
        None,
        frozen_audio_denoise_mask.as_ref(),
        Some("two_stage.stage2"),
        debug_enabled.then_some("stage2"),
        progress,
    )?;
    log_timing("two_stage.stage2.denoise", stage2_denoise_start);
    drop(stage2_transformer);
    device.synchronize()?;
    let latents = maybe_apply_temporal_upsampler(plan, &latents, device, dtype)?;

    let mut vae = load_ltx2_video_vae(plan, device, dtype)?;
    vae.use_tiling = false;
    vae.use_framewise_decoding = false;
    let decode_start = Instant::now();
    let (_dec_output, video) = vae.decode(&latents.to_dtype(dtype)?, None, false, false)?;
    let frames = decoded_video_to_frames(&video, requested_pixel_shape)?;
    if device.is_cuda() {
        device.synchronize()?;
    }
    drop(video);
    drop(vae);
    log_timing("two_stage.decode_video", decode_start);
    let audio_render_start = Instant::now();
    let audio_track = if let Some(conditioned_audio) = conditioned_audio.as_ref() {
        conditioned_audio.original_track.clone()
    } else {
        maybe_render_native_audio_track(plan, audio_latents.as_ref(), device, dtype)?
    };
    log_timing("two_stage.render_audio", audio_render_start);
    drop(latents);
    drop(audio_latents);
    drop(stage2_audio_start);
    drop(stage2_video_start);
    drop(stage2_audio_noise);
    drop(stage2_video_noise);
    drop(stage2_video_positions);
    drop(stage2_clean_video_latents);
    drop(stage1_audio_latents);
    drop(stage1_video_latents);
    drop(stage1_audio_noise);
    drop(frozen_audio_denoise_mask);
    drop(conditioned_audio);
    drop(stage1_video_noise);
    let _ = cond_mask;
    let _ = uncond_mask;
    let _ = alt_mask;
    drop(prompt_inputs);
    drop(latent_stats);
    if device.is_cuda() {
        device.synchronize()?;
    }

    let has_audio = audio_track.is_some();
    let audio_sample_rate = audio_track.as_ref().map(|track| track.sample_rate);
    let audio_channels = audio_track.as_ref().map(|track| u32::from(track.channels));

    Ok(NativeRenderedVideo {
        frames,
        audio_track,
        has_audio,
        audio_sample_rate,
        audio_channels,
    })
}

fn render_real_one_stage_av(
    plan: &Ltx2GeneratePlan,
    prepared: &NativePreparedRun,
    device: &candle_core::Device,
    progress: Option<&ProgressCallback>,
) -> Result<NativeRenderedVideo> {
    let debug_enabled = ltx_debug_enabled();
    let prompt_inputs = prepare_render_prompt_inputs(
        prepared,
        device,
        RenderPromptInputOptions {
            include_unconditional: true,
            include_alt: true,
        },
    )?;
    let audio_shape = prompt_inputs.audio_shape;
    let cond_mask: Option<&Tensor> = None;
    let uncond_mask: Option<&Tensor> = None;
    let alt_mask: Option<&Tensor> = None;
    let stage1_video_noise = seeded_randn(
        plan.seed,
        &[
            prepared.video_latent_shape.batch,
            prepared.video_latent_shape.channels,
            prepared.video_latent_shape.frames,
            prepared.video_latent_shape.height,
            prepared.video_latent_shape.width,
        ],
        device,
        DType::F32,
    )?;
    let stage1_audio_noise = match audio_shape {
        Some(audio_shape) => Some(seeded_randn(
            plan.seed ^ 0x4155_4449_4f4c_5458,
            &[
                audio_shape.batch,
                audio_shape.channels,
                audio_shape.frames,
                audio_shape.mel_bins,
            ],
            device,
            DType::F32,
        )?),
        None => None,
    };

    if debug_enabled {
        log_tensor_stats("video_context", &prompt_inputs.cond_context)?;
        if let Some(audio_context) = prompt_inputs.audio_context.as_ref() {
            log_tensor_stats("audio_context", audio_context)?;
        }
        log_tensor_stats("initial_video_latents", &stage1_video_noise)?;
        if let Some(stage1_audio_noise) = stage1_audio_noise.as_ref() {
            log_tensor_stats("initial_audio_latents", stage1_audio_noise)?;
        }
    }

    let dtype = gpu_dtype(device);
    let stage1_guidance_scale = stage_guidance_scale(plan, 0)?;
    let stage1_video_conditioning = maybe_load_stage_video_conditioning(
        plan,
        prepared.video_pixel_shape,
        device,
        dtype,
        false,
    )?;
    if debug_enabled {
        eprintln!("[ltx2-debug] loading one-stage transformer");
    }
    let transformer = load_ltx2_av_transformer(plan, device)?;
    if debug_enabled {
        log_debug_vram("after_one_stage_transformer_load");
    }
    let stage1_requires_uncond = stage_requires_unconditional_context(plan, 0)?;
    let (latents, stage1_audio_latents) = run_real_distilled_stage(
        &transformer,
        prepared.video_latent_shape,
        audio_shape,
        &stage1_video_noise,
        &stage1_video_conditioning,
        None,
        stage1_audio_noise.as_ref(),
        None,
        &prompt_inputs.video_positions,
        prompt_inputs.audio_positions.as_ref(),
        &prompt_inputs.cond_context,
        stage1_requires_uncond
            .then_some(prompt_inputs.uncond_context.as_ref())
            .flatten(),
        prompt_inputs.alt_context.as_ref(),
        prompt_inputs.audio_context.as_ref(),
        stage1_requires_uncond
            .then_some(prompt_inputs.uncond_audio_context.as_ref())
            .flatten(),
        prompt_inputs.alt_audio_context.as_ref(),
        cond_mask,
        if stage1_requires_uncond {
            uncond_mask
        } else {
            None
        },
        alt_mask,
        None,
        stage1_guidance_scale,
        DISTILLED_STAGE1_SIGMAS_NO_TERMINAL,
        stage_sampler_mode(plan, 0)?,
        Some(&stage1_video_noise),
        stage1_audio_noise.as_ref(),
        None,
        None,
        Some("one_stage"),
        debug_enabled.then_some("one-stage"),
        progress,
    )?;
    if debug_enabled {
        log_debug_vram("after_one_stage_denoise");
        log_tensor_stats("final_video_latents", &latents)?;
    }
    drop(transformer);
    device.synchronize()?;
    if debug_enabled {
        log_debug_vram("after_one_stage_transformer_drop");
    }

    let mut vae = load_ltx2_video_vae(plan, device, dtype)?;
    vae.use_tiling = false;
    vae.use_framewise_decoding = false;
    let (_dec_output, video) = vae.decode(&latents.to_dtype(dtype)?, None, false, false)?;
    if debug_enabled {
        log_tensor_stats("decoded_video", &video)?;
    }
    let frames = decoded_video_to_frames(&video, prepared.video_pixel_shape)?;
    if device.is_cuda() {
        device.synchronize()?;
    }
    drop(video);
    drop(vae);
    let audio_track =
        maybe_render_native_audio_track(plan, stage1_audio_latents.as_ref(), device, dtype)?;
    drop(latents);
    drop(stage1_audio_latents);
    drop(stage1_audio_noise);
    drop(stage1_video_noise);
    let _ = cond_mask;
    let _ = uncond_mask;
    let _ = alt_mask;
    drop(prompt_inputs);
    if device.is_cuda() {
        device.synchronize()?;
    }

    let has_audio = audio_track.is_some();
    let audio_sample_rate = audio_track.as_ref().map(|track| track.sample_rate);
    let audio_channels = audio_track.as_ref().map(|track| u32::from(track.channels));

    Ok(NativeRenderedVideo {
        frames,
        audio_track,
        has_audio,
        audio_sample_rate,
        audio_channels,
    })
}

fn render_real_retake_av(
    plan: &Ltx2GeneratePlan,
    prepared: &NativePreparedRun,
    device: &candle_core::Device,
    progress: Option<&ProgressCallback>,
) -> Result<NativeRenderedVideo> {
    let debug_enabled = ltx_debug_enabled();
    let prompt_inputs = prepare_render_prompt_inputs(
        prepared,
        device,
        RenderPromptInputOptions {
            include_unconditional: false,
            include_alt: false,
        },
    )?;
    let audio_shape = prompt_inputs.audio_shape;
    let cond_mask: Option<&Tensor> = None;
    let dtype = gpu_dtype(device);
    let retake_range = plan
        .retake_range
        .as_ref()
        .context("native LTX-2 retake requires a retake_range")?;
    let source_video = maybe_load_native_conditioning_video(
        plan,
        prepared.video_pixel_shape,
        prepared.video_latent_shape,
        device,
        dtype,
    )?
    .context("native LTX-2 retake requires a source_video")?;
    let stage_video_conditioning = maybe_load_stage_video_conditioning(
        plan,
        prepared.video_pixel_shape,
        device,
        dtype,
        false,
    )?;
    let video_retake_mask =
        build_temporal_token_denoise_mask(retake_range, &prompt_inputs.video_positions, device)?;
    let stage1_video_noise = seeded_randn(
        plan.seed,
        &[
            prepared.video_latent_shape.batch,
            prepared.video_latent_shape.channels,
            prepared.video_latent_shape.frames,
            prepared.video_latent_shape.height,
            prepared.video_latent_shape.width,
        ],
        device,
        DType::F32,
    )?;
    let conditioned_audio = maybe_load_native_conditioning_audio(plan, audio_shape, device, dtype)?;
    let audio_retake_mask = match (
        retake_range,
        prompt_inputs.audio_positions.as_ref(),
        conditioned_audio.as_ref(),
    ) {
        (range, Some(audio_positions), Some(_)) => Some(build_temporal_token_denoise_mask(
            range,
            audio_positions,
            device,
        )?),
        _ => None,
    };
    let stage1_audio_noise = match audio_shape {
        Some(audio_shape) => Some(seeded_randn(
            plan.seed ^ 0x4155_4449_4f4c_5458,
            &[
                audio_shape.batch,
                audio_shape.channels,
                audio_shape.frames,
                audio_shape.mel_bins,
            ],
            device,
            DType::F32,
        )?),
        None => None,
    };

    if debug_enabled {
        eprintln!("[ltx2-debug] loading retake transformer");
    }
    let transformer = load_ltx2_av_transformer(plan, device)?;
    let (latents, audio_latents) = run_real_distilled_stage(
        &transformer,
        prepared.video_latent_shape,
        audio_shape,
        &stage1_video_noise,
        &stage_video_conditioning,
        Some(&source_video.latents),
        stage1_audio_noise.as_ref(),
        conditioned_audio.as_ref().map(|audio| &audio.latents),
        &prompt_inputs.video_positions,
        prompt_inputs.audio_positions.as_ref(),
        &prompt_inputs.cond_context,
        None,
        None,
        prompt_inputs.audio_context.as_ref(),
        None,
        None,
        cond_mask,
        None,
        None,
        None,
        stage_guidance_scale(plan, 0)?,
        DISTILLED_STAGE1_SIGMAS_NO_TERMINAL,
        stage_sampler_mode(plan, 0)?,
        Some(&stage1_video_noise),
        stage1_audio_noise.as_ref(),
        Some(&video_retake_mask),
        audio_retake_mask.as_ref(),
        Some("retake.stage1"),
        debug_enabled.then_some("retake"),
        progress,
    )?;
    drop(transformer);
    if device.is_cuda() {
        device.synchronize()?;
    }

    let mut vae = load_ltx2_video_vae(plan, device, dtype)?;
    vae.use_tiling = false;
    vae.use_framewise_decoding = false;
    let (_dec_output, video) = vae.decode(&latents.to_dtype(dtype)?, None, false, false)?;
    let frames = decoded_video_to_frames(&video, prepared.video_pixel_shape)?;
    if device.is_cuda() {
        device.synchronize()?;
    }
    drop(video);
    drop(vae);
    let audio_track = maybe_render_native_audio_track(plan, audio_latents.as_ref(), device, dtype)?;
    drop(latents);
    drop(audio_latents);
    drop(stage1_audio_noise);
    drop(stage1_video_noise);
    drop(audio_retake_mask);
    drop(video_retake_mask);
    drop(conditioned_audio);
    drop(source_video);
    let _ = cond_mask;
    drop(prompt_inputs);
    if device.is_cuda() {
        device.synchronize()?;
    }

    let has_audio = audio_track.is_some();
    let audio_sample_rate = audio_track.as_ref().map(|track| track.sample_rate);
    let audio_channels = audio_track.as_ref().map(|track| u32::from(track.channels));

    Ok(NativeRenderedVideo {
        frames,
        audio_track,
        has_audio,
        audio_sample_rate,
        audio_channels,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_real_distilled_stage(
    transformer: &Ltx2AvTransformer3DModel,
    video_shape: VideoLatentShape,
    audio_shape: Option<AudioLatentShape>,
    video_start_latents: &Tensor,
    video_conditioning: &StageVideoConditioning,
    video_clean_latents: Option<&Tensor>,
    audio_start_latents: Option<&Tensor>,
    audio_clean_latents: Option<&Tensor>,
    video_positions: &Tensor,
    audio_positions: Option<&Tensor>,
    cond_context: &Tensor,
    uncond_context: Option<&Tensor>,
    alt_context: Option<&Tensor>,
    audio_context: Option<&Tensor>,
    uncond_audio_context: Option<&Tensor>,
    alt_audio_context: Option<&Tensor>,
    cond_mask: Option<&Tensor>,
    uncond_mask: Option<&Tensor>,
    alt_mask: Option<&Tensor>,
    multimodal_guidance: Option<(MultiModalGuiderParams, MultiModalGuiderParams)>,
    guidance_scale: f64,
    sigmas_no_terminal: &[f32],
    sampler_mode: SamplerMode,
    video_sampler_noise: Option<&Tensor>,
    audio_sampler_noise: Option<&Tensor>,
    video_denoise_mask: Option<&Tensor>,
    audio_denoise_mask: Option<&Tensor>,
    timing_label: Option<&str>,
    debug_stage: Option<&str>,
    progress: Option<&ProgressCallback>,
) -> Result<(Tensor, Option<Tensor>)> {
    let device = video_start_latents.device().clone();
    let video_patchifier = VideoLatentPatchifier::new(1);
    let audio_patchifier = AudioPatchifier::new(
        LTX2_AUDIO_SAMPLE_RATE,
        LTX2_AUDIO_HOP_LENGTH,
        LTX2_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        true,
        0,
    );
    let mut run_sigmas = sigmas_no_terminal.to_vec();
    run_sigmas.push(0.0);
    let base_video_token_count = video_patchifier.get_token_count(video_shape);
    let (mut video_latents, conditioned_video_positions) = apply_stage_video_conditioning(
        &video_patchifier.patchify(video_start_latents)?,
        video_positions,
        video_conditioning,
    )?;
    let clean_video_latents = match video_clean_latents {
        Some(latents) => video_patchifier.patchify(latents)?,
        None => clean_latents_for_conditioning(&video_latents, video_conditioning)?,
    };
    let video_denoise_mask = match video_denoise_mask {
        Some(mask) => mask.to_device(&device)?.to_dtype(DType::F32)?,
        None => build_video_conditioning_denoise_mask(
            base_video_token_count,
            video_conditioning,
            &device,
        )?,
    };
    let video_self_attention_mask = build_video_conditioning_self_attention_mask(
        base_video_token_count,
        video_conditioning,
        &device,
    )?;
    let video_positions = &conditioned_video_positions;
    let uses_video_freeze_mask = video_clean_latents.is_some() || !video_conditioning.is_empty();
    let video_sampler_noise = video_sampler_noise
        .map(|noise| video_patchifier.patchify(noise))
        .transpose()?;
    let mut audio_latents = match (audio_shape, audio_start_latents) {
        (Some(_), Some(latents)) => Some(audio_patchifier.patchify(latents)?),
        _ => None,
    };
    let clean_audio_latents = match (audio_shape, audio_clean_latents) {
        (Some(_), Some(latents)) => Some(audio_patchifier.patchify(latents)?),
        _ => audio_latents.clone(),
    };
    let audio_sampler_noise = match (audio_shape, audio_sampler_noise) {
        (Some(_), Some(noise)) => Some(audio_patchifier.patchify(noise)?),
        _ => None,
    };
    let audio_denoise_mask = audio_denoise_mask
        .map(|mask| mask.to_device(&device)?.to_dtype(DType::F32))
        .transpose()?;
    if uses_video_freeze_mask {
        video_latents =
            blend_conditioned_denoised(&video_latents, &clean_video_latents, &video_denoise_mask)?;
    }
    if let Some(blended_audio_latents) = match (
        audio_latents.as_ref(),
        clean_audio_latents.as_ref(),
        audio_denoise_mask.as_ref(),
    ) {
        (Some(audio_latents), Some(clean_audio_latents), Some(audio_denoise_mask)) => Some(
            blend_conditioned_denoised(audio_latents, clean_audio_latents, audio_denoise_mask)?,
        ),
        _ => None,
    } {
        audio_latents = Some(blended_audio_latents);
    }
    let use_cfg = guidance_scale > 1.0;
    let multimodal_guiders = multimodal_guidance.map(|(video_params, audio_params)| {
        (
            MultiModalGuider::new(video_params, uncond_context.cloned()),
            MultiModalGuider::new(audio_params, uncond_audio_context.cloned()),
        )
    });
    let cond_static_inputs = if multimodal_guiders.is_none() {
        Some(transformer.prepare_static_inputs(
            cond_context,
            audio_context,
            cond_mask,
            cond_mask,
            video_self_attention_mask.as_ref(),
            None,
            video_positions,
            audio_positions,
        )?)
    } else {
        None
    };
    let uncond_static_inputs = if multimodal_guiders.is_none() {
        match (uncond_context, uncond_audio_context) {
            (Some(uncond_context), uncond_audio_context) => {
                Some(transformer.prepare_static_inputs(
                    uncond_context,
                    uncond_audio_context,
                    uncond_mask,
                    uncond_mask,
                    video_self_attention_mask.as_ref(),
                    None,
                    video_positions,
                    audio_positions,
                )?)
            }
            (None, _) => None,
        }
    } else {
        None
    };
    let alt_static_inputs = if multimodal_guiders.is_none() {
        match (alt_context, alt_audio_context) {
            (Some(alt_context), alt_audio_context) => Some(transformer.prepare_static_inputs(
                alt_context,
                alt_audio_context,
                alt_mask,
                alt_mask,
                video_self_attention_mask.as_ref(),
                None,
                video_positions,
                audio_positions,
            )?),
            (None, _) => None,
        }
    } else {
        None
    };
    let mut step_setup_secs = 0.0;
    let mut transformer_secs = 0.0;
    let mut update_secs = 0.0;

    for (step_idx, sigma) in run_sigmas
        .iter()
        .copied()
        .take(run_sigmas.len().saturating_sub(1))
        .enumerate()
    {
        let step_start = Instant::now();
        if let Some(stage) = debug_stage {
            eprintln!("[ltx2-debug] {stage} step={step_idx} sigma={sigma:.6} entering");
        }
        let step_setup_start = Instant::now();
        let video_sigma = Tensor::full(sigma, (video_latents.dim(0)?,), &device)?;
        let video_timestep = timestep_from_sigma_and_mask(
            sigma,
            video_latents.dim(0)?,
            uses_video_freeze_mask.then_some(&video_denoise_mask),
            &device,
        )?;
        let audio_sigma = if let Some(audio_latents_ref) = audio_latents.as_ref() {
            Some(Tensor::full(sigma, (audio_latents_ref.dim(0)?,), &device)?)
        } else {
            None
        };
        let audio_timestep = if let Some(audio_latents_ref) = audio_latents.as_ref() {
            Some(timestep_from_sigma_and_mask(
                sigma,
                audio_latents_ref.dim(0)?,
                audio_denoise_mask.as_ref(),
                &device,
            )?)
        } else {
            None
        };
        step_setup_secs += step_setup_start.elapsed().as_secs_f64();
        let transformer_start = Instant::now();
        let (mut video_denoised, audio_denoised, video_velocity): (
            Tensor,
            Option<Tensor>,
            Option<Tensor>,
        ) = if let Some((video_guider, audio_guider)) = multimodal_guiders.as_ref() {
            let (video_denoised, audio_denoised) = multimodal_guided_denoise_step(
                transformer,
                &video_latents,
                audio_latents.as_ref(),
                cond_context,
                audio_context,
                cond_mask,
                uncond_mask,
                &video_sigma,
                &video_timestep,
                audio_sigma.as_ref(),
                audio_timestep.as_ref(),
                video_self_attention_mask.as_ref(),
                video_positions,
                audio_positions,
                video_guider,
                audio_guider,
                step_idx,
            )?;
            (video_denoised, audio_denoised, None)
        } else if let Some(audio_latents_ref) = audio_latents.as_ref() {
            if use_cfg {
                let uncond_static_inputs = uncond_static_inputs
                    .as_ref()
                    .context("missing unconditional static inputs for CFG")?;
                let cond_static_inputs = cond_static_inputs
                    .as_ref()
                    .context("missing conditional static inputs for multimodal stage")?;
                let (uncond_video_velocity, uncond_audio_velocity) = transformer
                    .forward_with_static_inputs(
                        &video_latents,
                        Some(audio_latents_ref),
                        &video_sigma,
                        &video_timestep,
                        audio_sigma.as_ref(),
                        audio_timestep.as_ref(),
                        uncond_static_inputs,
                        None,
                    )?;
                let (cond_video_velocity, cond_audio_velocity) = transformer
                    .forward_with_static_inputs(
                        &video_latents,
                        Some(audio_latents_ref),
                        &video_sigma,
                        &video_timestep,
                        audio_sigma.as_ref(),
                        audio_timestep.as_ref(),
                        cond_static_inputs,
                        None,
                    )?;
                let uncond_audio_velocity = uncond_audio_velocity
                    .context("audio branch unexpectedly returned no unconditional output")?;
                let cond_audio_velocity = cond_audio_velocity
                    .context("audio branch unexpectedly returned no conditional output")?;
                (
                    denoised_from_velocity(
                        &video_latents,
                        &guided_velocity_from_cfg(
                            &video_latents,
                            &cond_video_velocity,
                            &uncond_video_velocity,
                            sigma,
                            guidance_scale,
                        )?,
                        sigma,
                    )?,
                    Some(denoised_from_velocity(
                        audio_latents_ref,
                        &guided_velocity_from_cfg(
                            audio_latents_ref,
                            &cond_audio_velocity,
                            &uncond_audio_velocity,
                            sigma,
                            guidance_scale,
                        )?,
                        sigma,
                    )?),
                    Some(cond_video_velocity),
                )
            } else {
                let cond_static_inputs = cond_static_inputs
                    .as_ref()
                    .context("missing conditional static inputs for multimodal stage")?;
                let (cond_video_velocity, cond_audio_velocity) = transformer
                    .forward_with_static_inputs(
                        &video_latents,
                        Some(audio_latents_ref),
                        &video_sigma,
                        &video_timestep,
                        audio_sigma.as_ref(),
                        audio_timestep.as_ref(),
                        cond_static_inputs,
                        None,
                    )?;
                if ltx_debug_compare_uncond_enabled() && step_idx == 0 {
                    if let Some(uncond_static_inputs) = uncond_static_inputs.as_ref() {
                        let (uncond_video_velocity, uncond_audio_velocity) = transformer
                            .forward_with_static_inputs(
                                &video_latents,
                                Some(audio_latents_ref),
                                &video_sigma,
                                &video_timestep,
                                audio_sigma.as_ref(),
                                audio_timestep.as_ref(),
                                uncond_static_inputs,
                                None,
                            )?;
                        log_distilled_prompt_sensitivity(
                            debug_stage,
                            step_idx,
                            sigma,
                            &video_latents,
                            &cond_video_velocity,
                            &uncond_video_velocity,
                            Some(audio_latents_ref),
                            cond_audio_velocity.as_ref(),
                            uncond_audio_velocity.as_ref(),
                        )?;
                    }
                }
                if step_idx == 0 {
                    if let Some(alt_static_inputs) = alt_static_inputs.as_ref() {
                        let (alt_video_velocity, alt_audio_velocity) = transformer
                            .forward_with_static_inputs(
                                &video_latents,
                                Some(audio_latents_ref),
                                &video_sigma,
                                &video_timestep,
                                audio_sigma.as_ref(),
                                audio_timestep.as_ref(),
                                alt_static_inputs,
                                None,
                            )?;
                        log_distilled_alternate_prompt_sensitivity(
                            debug_stage,
                            step_idx,
                            sigma,
                            &video_latents,
                            &cond_video_velocity,
                            &alt_video_velocity,
                            Some(audio_latents_ref),
                            cond_audio_velocity.as_ref(),
                            alt_audio_velocity.as_ref(),
                        )?;
                    }
                }
                (
                    denoised_from_velocity(&video_latents, &cond_video_velocity, sigma)?,
                    cond_audio_velocity
                        .as_ref()
                        .map(|velocity| denoised_from_velocity(audio_latents_ref, velocity, sigma))
                        .transpose()?,
                    Some(cond_video_velocity),
                )
            }
        } else if use_cfg {
            let uncond_static_inputs = uncond_static_inputs
                .as_ref()
                .context("missing unconditional static inputs for CFG")?;
            let cond_static_inputs = cond_static_inputs
                .as_ref()
                .context("missing conditional static inputs for video stage")?;
            let (uncond_video_velocity, _) = transformer.forward_with_static_inputs(
                &video_latents,
                None,
                &video_sigma,
                &video_timestep,
                None,
                None,
                uncond_static_inputs,
                None,
            )?;
            let (cond_video_velocity, _) = transformer.forward_with_static_inputs(
                &video_latents,
                None,
                &video_sigma,
                &video_timestep,
                None,
                None,
                cond_static_inputs,
                None,
            )?;
            (
                denoised_from_velocity(
                    &video_latents,
                    &guided_velocity_from_cfg(
                        &video_latents,
                        &cond_video_velocity,
                        &uncond_video_velocity,
                        sigma,
                        guidance_scale,
                    )?,
                    sigma,
                )?,
                None,
                Some(cond_video_velocity),
            )
        } else {
            let cond_static_inputs = cond_static_inputs
                .as_ref()
                .context("missing conditional static inputs for video stage")?;
            let (cond_video_velocity, _cond_audio_velocity) = transformer
                .forward_with_static_inputs(
                    &video_latents,
                    None,
                    &video_sigma,
                    &video_timestep,
                    None,
                    None,
                    cond_static_inputs,
                    None,
                )?;
            if ltx_debug_compare_uncond_enabled() && step_idx == 0 {
                if let Some(uncond_static_inputs) = uncond_static_inputs.as_ref() {
                    let (uncond_video_velocity, _) = transformer.forward_with_static_inputs(
                        &video_latents,
                        None,
                        &video_sigma,
                        &video_timestep,
                        None,
                        None,
                        uncond_static_inputs,
                        None,
                    )?;
                    log_distilled_prompt_sensitivity(
                        debug_stage,
                        step_idx,
                        sigma,
                        &video_latents,
                        &cond_video_velocity,
                        &uncond_video_velocity,
                        None,
                        None,
                        None,
                    )?;
                }
            }
            if step_idx == 0 {
                if let Some(alt_static_inputs) = alt_static_inputs.as_ref() {
                    let (alt_video_velocity, _) = transformer.forward_with_static_inputs(
                        &video_latents,
                        None,
                        &video_sigma,
                        &video_timestep,
                        None,
                        None,
                        alt_static_inputs,
                        None,
                    )?;
                    log_distilled_alternate_prompt_sensitivity(
                        debug_stage,
                        step_idx,
                        sigma,
                        &video_latents,
                        &cond_video_velocity,
                        &alt_video_velocity,
                        None,
                        None,
                        None,
                    )?;
                }
            }
            (
                denoised_from_velocity(&video_latents, &cond_video_velocity, sigma)?,
                None,
                Some(cond_video_velocity),
            )
        };
        transformer_secs += transformer_start.elapsed().as_secs_f64();
        let update_start = Instant::now();
        // Keep the hot denoise loop fully device-side unless step-level debug
        // inspection is explicitly enabled.
        if should_inspect_step_velocity(debug_stage) {
            let stage =
                debug_stage.expect("debug stage should be present when inspection is enabled");
            let video_velocity = video_velocity
                .as_ref()
                .context("video velocity missing for debug inspection")?;
            let video_velocity = video_velocity.to_dtype(DType::F32)?;
            log_tensor_stats("video_velocity", &video_velocity)?;
            eprintln!("[ltx2-debug] {stage} step={step_idx} sigma={sigma:.6}");
        }
        if uses_video_freeze_mask {
            video_denoised = blend_conditioned_denoised(
                &video_denoised,
                &clean_video_latents,
                &video_denoise_mask,
            )?;
        }
        video_latents = match sampler_mode {
            SamplerMode::Euler => {
                euler_step(&video_latents, &video_denoised, &run_sigmas, step_idx)?
            }
            SamplerMode::Res2S => res2s_step(
                &video_latents,
                &video_denoised,
                sigma as f64,
                run_sigmas[step_idx + 1] as f64,
                video_sampler_noise
                    .as_ref()
                    .context("video sampler noise missing for Res2S stage")?,
                0.5,
            )?,
        };
        if !video_conditioning.is_empty() {
            video_latents = reapply_stage_video_conditioning(
                &video_latents,
                base_video_token_count,
                video_conditioning,
            )?;
        }

        if let (Some(audio_latents), Some(audio_velocity)) =
            (audio_latents.as_mut(), audio_denoised.as_ref())
        {
            let audio_velocity = if let (Some(clean_audio_latents), Some(audio_denoise_mask)) =
                (clean_audio_latents.as_ref(), audio_denoise_mask.as_ref())
            {
                blend_conditioned_denoised(audio_velocity, clean_audio_latents, audio_denoise_mask)?
            } else {
                audio_velocity.clone()
            };
            *audio_latents = match sampler_mode {
                SamplerMode::Euler => {
                    euler_step(audio_latents, &audio_velocity, &run_sigmas, step_idx)?
                }
                SamplerMode::Res2S => res2s_step(
                    audio_latents,
                    &audio_velocity,
                    sigma as f64,
                    run_sigmas[step_idx + 1] as f64,
                    audio_sampler_noise
                        .as_ref()
                        .context("audio sampler noise missing for Res2S stage")?,
                    0.5,
                )?,
            };
        }
        update_secs += update_start.elapsed().as_secs_f64();
        emit_denoise_progress(
            progress,
            step_idx + 1,
            run_sigmas.len() - 1,
            step_start.elapsed(),
        );

        if let Some(stage) = debug_stage {
            eprintln!("[ltx2-debug] {stage} step={step_idx} sigma={sigma:.6}");
            log_tensor_stats("step_video_latents", &video_latents)?;
            if let Some(audio_latents) = audio_latents.as_ref() {
                log_tensor_stats("step_audio_latents", audio_latents)?;
            }
            log_tensor_stats("video_x0", &video_denoised)?;
            if let (Some(audio_latents), Some(audio_denoised)) =
                (audio_latents.as_ref(), audio_denoised.as_ref())
            {
                log_tensor_stats("audio_x0", audio_denoised)?;
                let audio_velocity = velocity_from_denoised(audio_latents, audio_denoised, sigma)?;
                log_tensor_stats("audio_velocity", &audio_velocity)?;
            }
        }
    }

    let video_latents = strip_appended_video_conditioning(&video_latents, base_video_token_count)?;
    let video_latents = video_patchifier.unpatchify(&video_latents, video_shape)?;
    let audio_latents = match (audio_latents, audio_shape) {
        (Some(latents), Some(shape)) => Some(audio_patchifier.unpatchify(&latents, shape)?),
        _ => None,
    };
    if debug_stage.is_some() {
        log_tensor_stats("final_patched_latents", &video_latents)?;
    }
    if device.is_cuda() {
        device.synchronize()?;
    }
    if let Some(timing_label) = timing_label {
        log_elapsed_secs(&format!("{timing_label}.step_setup_total"), step_setup_secs);
        log_elapsed_secs(
            &format!("{timing_label}.transformer_total"),
            transformer_secs,
        );
        log_elapsed_secs(&format!("{timing_label}.update_total"), update_secs);
    }
    Ok((video_latents, audio_latents))
}

fn build_video_positions(
    pixel_shape: VideoPixelShape,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let scale_factors = SpatioTemporalScaleFactors::default();
    let latent_shape =
        VideoLatentShape::from_pixel_shape(pixel_shape, LTX2_VIDEO_LATENT_CHANNELS, scale_factors);
    let video_patchifier = VideoLatentPatchifier::new(1);
    scale_video_time_to_seconds(
        &get_pixel_coords(
            &video_token_positions(video_patchifier, latent_shape, device)?,
            scale_factors,
            true,
        )?,
        pixel_shape.fps,
    )
}

fn mix_clean_latents_with_noise(
    clean_latents: &Tensor,
    noise: &Tensor,
    noise_scale: f32,
) -> Result<Tensor> {
    let noise_scale = noise_scale as f64;
    let clean_scale = 1.0 - noise_scale;
    clean_latents
        .affine(clean_scale, 0.0)?
        .broadcast_add(&noise.affine(noise_scale, 0.0)?)
        .map_err(Into::into)
}

fn should_inspect_step_velocity(debug_stage: Option<&str>) -> bool {
    debug_stage.is_some()
}

fn decoded_video_to_frames(video: &Tensor, pixel_shape: VideoPixelShape) -> Result<Vec<RgbImage>> {
    let video =
        ((video.to_dtype(DType::F32)?.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let video = video.i(0)?;

    let mut frames = Vec::with_capacity(video.dim(1)?);
    for index in 0..video.dim(1)? {
        let frame = video
            .i((.., index, .., ..))?
            .permute((1, 2, 0))?
            .contiguous()?;
        let (decoded_height, decoded_width, decoded_channels) = frame.dims3()?;
        if decoded_channels != 3 {
            anyhow::bail!(
                "expected decoded LTX-2 frame to have 3 channels, got {decoded_channels}"
            );
        }
        let data: Vec<u8> = frame.flatten_all()?.to_vec1()?;
        let rgb = RgbImage::from_raw(decoded_width as u32, decoded_height as u32, data)
            .context("failed to build an RGB frame from the decoded LTX-2 tensor")?;
        let rgb = if decoded_width != pixel_shape.width || decoded_height != pixel_shape.height {
            imageops::resize(
                &rgb,
                pixel_shape.width as u32,
                pixel_shape.height as u32,
                imageops::FilterType::Triangle,
            )
        } else {
            rgb
        };
        frames.push(rgb);
    }
    Ok(frames)
}

fn maybe_render_native_audio_track(
    plan: &Ltx2GeneratePlan,
    audio_latents: Option<&Tensor>,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Option<NativeAudioTrack>> {
    if !plan.execution_graph.wants_audio_output {
        return Ok(None);
    }
    let audio_latents = audio_latents.context(
        "native LTX-2 audio output requested but the denoiser produced no audio latents",
    )?;
    let decoder =
        Ltx2AudioDecoder::load_from_checkpoint(Path::new(&plan.checkpoint_path), dtype, device)?;
    let mel_spec = decoder.decode(&audio_latents.to_dtype(dtype)?)?;
    drop(decoder);
    if device.is_cuda() {
        device.synchronize()?;
    }

    let vocoder =
        Ltx2VocoderWithBwe::load_from_checkpoint(Path::new(&plan.checkpoint_path), device)?;
    let output_sample_rate = vocoder.config.output_sample_rate as u32;
    let waveform = vocoder.forward(&mel_spec.to_dtype(DType::F32)?)?;
    drop(vocoder);
    drop(mel_spec);
    if device.is_cuda() {
        device.synchronize()?;
    }
    waveform_to_audio_track(&waveform, output_sample_rate)
}

fn video_tensor_from_frames(
    frames: &[RgbImage],
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    let first = frames
        .first()
        .context("native LTX-2 source video conditioning requires at least one frame")?;
    let width = first.width() as usize;
    let height = first.height() as usize;
    let frame_count = frames.len();
    let mut data = Vec::with_capacity(frame_count * width * height * 3);

    for channel in 0..3usize {
        for frame in frames {
            if frame.width() as usize != width || frame.height() as usize != height {
                anyhow::bail!("native LTX-2 source video frames do not share a common size");
            }
            for pixel in frame.pixels() {
                data.push((pixel[channel] as f32 / 127.5) - 1.0);
            }
        }
    }

    Tensor::from_vec(data, (1, 3, frame_count, height, width), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

fn conform_video_latent_length(
    latents: &Tensor,
    expected_shape: VideoLatentShape,
) -> Result<Tensor> {
    let (batch, channels, frames, height, width) = latents.dims5()?;
    if batch != expected_shape.batch
        || channels != expected_shape.channels
        || height != expected_shape.height
        || width != expected_shape.width
    {
        anyhow::bail!(
            "native LTX-2 source video latent shape mismatch: got [{batch}, {channels}, {frames}, {height}, {width}], expected [{}, {}, {}, {}, {}]",
            expected_shape.batch,
            expected_shape.channels,
            expected_shape.frames,
            expected_shape.height,
            expected_shape.width
        );
    }
    if frames == expected_shape.frames {
        return Ok(latents.clone());
    }
    if frames > expected_shape.frames {
        return latents
            .narrow(2, 0, expected_shape.frames)
            .map_err(Into::into);
    }
    let pad_frames = expected_shape.frames - frames;
    let pad = Tensor::zeros(
        (batch, channels, pad_frames, height, width),
        latents.dtype(),
        latents.device(),
    )?;
    Tensor::cat(&[latents, &pad], 2).map_err(Into::into)
}

fn maybe_load_native_conditioning_video(
    plan: &Ltx2GeneratePlan,
    pixel_shape: VideoPixelShape,
    latent_shape: VideoLatentShape,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Option<NativeConditioningVideo>> {
    let Some(video_path) = plan.conditioning.video_path.as_ref() else {
        return Ok(None);
    };
    let (metadata, frames) = media::decode_video_frames(Path::new(video_path))?;
    if metadata.fps != pixel_shape.fps.round() as u32 {
        anyhow::bail!(
            "native LTX-2 source video FPS mismatch: source={} expected={}",
            metadata.fps,
            pixel_shape.fps.round() as u32
        );
    }
    let resized = frames
        .into_iter()
        .map(|frame| {
            if frame.width() == pixel_shape.width as u32
                && frame.height() == pixel_shape.height as u32
            {
                frame
            } else {
                imageops::resize(
                    &frame,
                    pixel_shape.width as u32,
                    pixel_shape.height as u32,
                    imageops::FilterType::Lanczos3,
                )
            }
        })
        .collect::<Vec<_>>();
    let video = video_tensor_from_frames(&resized, device, dtype)?;
    let mut vae = load_ltx2_video_vae(plan, device, dtype)?;
    vae.use_tiling = false;
    vae.use_framewise_decoding = false;
    let latents = conform_video_latent_length(&vae.encode(&video)?, latent_shape)?;
    drop(vae);
    if device.is_cuda() {
        device.synchronize()?;
    }
    Ok(Some(NativeConditioningVideo { latents }))
}

fn maybe_load_native_conditioning_audio(
    plan: &Ltx2GeneratePlan,
    audio_shape: Option<AudioLatentShape>,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Option<NativeConditioningAudio>> {
    let explicit_audio_path = plan.conditioning.audio_path.as_ref();
    let audio_path = explicit_audio_path.or({
        if plan.execution_graph.uses_retake_masking {
            plan.conditioning.video_path.as_ref()
        } else {
            None
        }
    });
    let Some(audio_path) = audio_path else {
        return Ok(None);
    };
    let audio_shape = audio_shape.context(
        "native LTX-2 audio conditioning requested but the prepared run has no audio latent shape",
    )?;
    let max_duration = plan.num_frames as f32 / plan.frame_rate.max(1) as f32;
    let decoded_audio = match DecodedAudio::from_file(Path::new(audio_path), Some(max_duration))? {
        Some(decoded_audio) => decoded_audio,
        None if explicit_audio_path.is_none() && plan.execution_graph.uses_retake_masking => {
            return Ok(None);
        }
        None => {
            return Err(anyhow::anyhow!(
                "source audio '{}' did not contain a decodable audio stream",
                audio_path
            ));
        }
    };
    let encoder =
        Ltx2AudioEncoder::load_from_checkpoint(Path::new(&plan.checkpoint_path), dtype, device)?;
    let latents = conform_audio_latent_length(&encoder.encode_audio(&decoded_audio)?, audio_shape)?;
    drop(encoder);
    if device.is_cuda() {
        device.synchronize()?;
    }
    let original_track = if plan.execution_graph.wants_audio_output {
        native_audio_track_from_decoded_audio(&decoded_audio)?
    } else {
        None
    };
    Ok(Some(NativeConditioningAudio {
        latents,
        original_track,
    }))
}

fn build_temporal_token_denoise_mask(
    range: &TimeRange,
    positions: &Tensor,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let temporal = positions
        .i((.., 0, .., ..))?
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?;
    let (batch, tokens, _) = temporal.dims3()?;
    let mut values = Vec::with_capacity(batch * tokens);
    for batch_item in temporal.to_vec3::<f32>()? {
        for bounds in batch_item {
            let start = bounds.first().copied().unwrap_or_default();
            let end = bounds.get(1).copied().unwrap_or(start);
            let active = end > range.start_seconds && start < range.end_seconds;
            values.push(if active { 1.0f32 } else { 0.0f32 });
        }
    }
    Tensor::from_vec(values, (batch, tokens), device).map_err(Into::into)
}

fn conform_audio_latent_length(
    latents: &Tensor,
    expected_shape: AudioLatentShape,
) -> Result<Tensor> {
    let (batch, channels, frames, mel_bins) = latents.dims4()?;
    if batch != expected_shape.batch
        || channels != expected_shape.channels
        || mel_bins != expected_shape.mel_bins
    {
        anyhow::bail!(
            "native LTX-2 source audio latent shape mismatch: got [{batch}, {channels}, {frames}, {mel_bins}], expected [{}, {}, {}, {}]",
            expected_shape.batch,
            expected_shape.channels,
            expected_shape.frames,
            expected_shape.mel_bins
        );
    }
    if frames == expected_shape.frames {
        return Ok(latents.clone());
    }
    if frames > expected_shape.frames {
        return latents
            .narrow(2, 0, expected_shape.frames)
            .map_err(Into::into);
    }
    let pad_frames = expected_shape.frames - frames;
    let pad = Tensor::zeros(
        (batch, channels, pad_frames, mel_bins),
        latents.dtype(),
        latents.device(),
    )?;
    Tensor::cat(&[latents, &pad], 2).map_err(Into::into)
}

fn build_frozen_audio_denoise_mask(
    audio_shape: AudioLatentShape,
    device: &candle_core::Device,
) -> Result<Tensor> {
    Tensor::zeros((audio_shape.batch, audio_shape.frames), DType::F32, device).map_err(Into::into)
}

fn timestep_from_sigma_and_mask(
    sigma: f32,
    batch_size: usize,
    denoise_mask: Option<&Tensor>,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let sigma_tensor = Tensor::full(sigma, (batch_size,), device)?;
    match denoise_mask {
        Some(mask) => mask.affine(sigma as f64, 0.0).map_err(Into::into),
        None => Ok(sigma_tensor),
    }
}

fn native_audio_track_from_decoded_audio(
    decoded_audio: &DecodedAudio,
) -> Result<Option<NativeAudioTrack>> {
    let channels = decoded_audio.channel_count();
    let samples_per_channel = decoded_audio.sample_count();
    if channels == 0 || samples_per_channel == 0 {
        return Ok(None);
    }
    let mut interleaved_samples = Vec::with_capacity(channels * samples_per_channel);
    for sample_idx in 0..samples_per_channel {
        for channel in &decoded_audio.channels {
            interleaved_samples.push(channel[sample_idx]);
        }
    }
    Ok(Some(NativeAudioTrack {
        interleaved_samples,
        sample_rate: decoded_audio.sample_rate as u32,
        channels: channels as u16,
    }))
}

fn waveform_to_audio_track(
    waveform: &Tensor,
    sample_rate: u32,
) -> Result<Option<NativeAudioTrack>> {
    let waveform = waveform
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?;
    let (batch, channels, samples_per_channel) = waveform.dims3()?;
    if batch == 0 || channels == 0 || samples_per_channel == 0 {
        return Ok(None);
    }
    let channel_vectors = waveform.i(0)?.to_vec2::<f32>()?;
    let mut interleaved_samples = Vec::with_capacity(channels * samples_per_channel);
    for sample_idx in 0..samples_per_channel {
        for channel in &channel_vectors {
            interleaved_samples.push(channel[sample_idx]);
        }
    }
    Ok(Some(NativeAudioTrack {
        interleaved_samples,
        sample_rate,
        channels: channels as u16,
    }))
}

fn maybe_write_debug_stage_video(
    stage: &str,
    vae: &AutoencoderKLLtx2Video,
    latents: &Tensor,
    pixel_shape: VideoPixelShape,
    dtype: DType,
) -> Result<()> {
    let Some(prefix) = env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX") else {
        return Ok(());
    };

    let (_decoded, video) = vae.decode(&latents.to_dtype(dtype)?, None, false, false)?;
    let frames = decoded_video_to_frames(&video, pixel_shape)?;
    let prefix = prefix.to_string_lossy();
    let first_frame_path = std::path::PathBuf::from(format!("{prefix}-{stage}-first-frame.png"));
    let contact_sheet_path =
        std::path::PathBuf::from(format!("{prefix}-{stage}-contact-sheet.png"));
    if let Some(first) = frames.first() {
        first.save(&first_frame_path)?;
    }
    write_contact_sheet_from_frames(&frames, &contact_sheet_path)?;
    eprintln!(
        "[ltx2-debug] wrote stage video: stage={stage} first_frame={} contact_sheet={}",
        first_frame_path.display(),
        contact_sheet_path.display()
    );
    Ok(())
}

fn write_contact_sheet_from_frames(
    frames: &[RgbImage],
    output_png: &std::path::Path,
) -> Result<()> {
    if frames.is_empty() {
        return Ok(());
    }

    let columns = 3usize;
    let rows = frames.len().div_ceil(columns);
    let frame_width = frames[0].width();
    let frame_height = frames[0].height();
    let mut sheet = RgbImage::from_pixel(
        frame_width * columns as u32,
        frame_height * rows as u32,
        Rgb([0, 0, 0]),
    );

    for (index, frame) in frames.iter().enumerate() {
        let x = (index % columns) as u32 * frame_width;
        let y = (index / columns) as u32 * frame_height;
        sheet.copy_from(frame, x, y)?;
    }

    sheet.save(output_png)?;
    Ok(())
}

fn repeat_batch(tensor: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(tensor.clone());
    }
    let parts = (0..repeats).map(|_| tensor.clone()).collect::<Vec<_>>();
    let refs = parts.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 0).map_err(Into::into)
}

fn cat_optional_batches(parts: &[Option<Tensor>]) -> Result<Option<Tensor>> {
    if parts.iter().all(Option::is_none) {
        return Ok(None);
    }
    if !parts.iter().all(Option::is_some) {
        anyhow::bail!("batched optional tensors must be either all present or all absent");
    }
    let tensors = parts.iter().flatten().collect::<Vec<_>>();
    Tensor::cat(&tensors, 0).map(Some).map_err(Into::into)
}

fn split_batch_chunk(tensor: &Tensor, index: usize, chunk: usize) -> Result<Tensor> {
    tensor.narrow(0, index * chunk, chunk).map_err(Into::into)
}

fn sigma_scale_for_sample(sample: &Tensor, sigma: &Tensor) -> Result<Tensor> {
    match sigma.rank() {
        1 => sigma
            .reshape((sample.dim(0)?, 1, 1))?
            .to_device(sample.device())?
            .to_dtype(sample.dtype())
            .map_err(Into::into),
        2 => sigma
            .reshape((sample.dim(0)?, sample.dim(1)?, 1))?
            .to_device(sample.device())?
            .to_dtype(sample.dtype())
            .map_err(Into::into),
        other => anyhow::bail!("expected sigma rank 1 or 2, got rank {other}"),
    }
}

fn denoised_from_velocity_with_sigma(
    sample: &Tensor,
    velocity: &Tensor,
    sigma: &Tensor,
) -> Result<Tensor> {
    let sigma = sigma_scale_for_sample(sample, sigma)?;
    let velocity = if velocity.dtype() == sample.dtype() {
        velocity.clone()
    } else {
        velocity.to_dtype(sample.dtype())?
    };
    sample
        .broadcast_sub(&velocity.broadcast_mul(&sigma)?)
        .map_err(Into::into)
}

fn multimodal_guided_denoise_step(
    transformer: &Ltx2AvTransformer3DModel,
    video_latents: &Tensor,
    audio_latents: Option<&Tensor>,
    cond_context: &Tensor,
    audio_context: Option<&Tensor>,
    cond_mask: Option<&Tensor>,
    uncond_mask: Option<&Tensor>,
    video_sigma: &Tensor,
    video_timestep: &Tensor,
    audio_sigma: Option<&Tensor>,
    audio_timestep: Option<&Tensor>,
    video_self_attention_mask: Option<&Tensor>,
    video_positions: &Tensor,
    audio_positions: Option<&Tensor>,
    video_guider: &MultiModalGuider,
    audio_guider: &MultiModalGuider,
    step_idx: usize,
) -> Result<(Tensor, Option<Tensor>)> {
    let video_skip = video_guider.should_skip_step(step_idx);
    let audio_skip = audio_guider.should_skip_step(step_idx);

    let mut video_contexts = vec![cond_context.clone()];
    let mut audio_contexts = vec![audio_context.cloned()];
    let mut video_masks = vec![cond_mask.cloned()];
    let mut audio_masks = vec![cond_mask.cloned()];
    let mut perturbations = vec![PerturbationConfig::empty()];
    let cond_index = 0usize;
    let mut uncond_index = None;
    let mut perturbed_index = None;
    let mut modality_index = None;

    if video_guider.do_unconditional_generation() || audio_guider.do_unconditional_generation() {
        let negative_video_context = video_guider
            .negative_context
            .as_ref()
            .context("missing unconditional video context for multimodal guidance")?;
        video_contexts.push(negative_video_context.clone());
        audio_contexts.push(
            audio_guider
                .negative_context
                .clone()
                .or_else(|| audio_context.cloned()),
        );
        video_masks.push(uncond_mask.cloned());
        audio_masks.push(uncond_mask.cloned());
        perturbations.push(PerturbationConfig::empty());
        uncond_index = Some(perturbations.len() - 1);
    }

    if video_guider.do_perturbed_generation() || audio_guider.do_perturbed_generation() {
        let mut stg_perturbations = Vec::new();
        if video_guider.do_perturbed_generation() {
            stg_perturbations.push(Perturbation::new(
                PerturbationType::SkipVideoSelfAttention,
                Some(video_guider.params.stg_blocks.clone()),
            ));
        }
        if audio_guider.do_perturbed_generation() {
            stg_perturbations.push(Perturbation::new(
                PerturbationType::SkipAudioSelfAttention,
                Some(audio_guider.params.stg_blocks.clone()),
            ));
        }
        video_contexts.push(cond_context.clone());
        audio_contexts.push(audio_context.cloned());
        video_masks.push(cond_mask.cloned());
        audio_masks.push(cond_mask.cloned());
        perturbations.push(PerturbationConfig::new(stg_perturbations));
        perturbed_index = Some(perturbations.len() - 1);
    }
    if video_guider.do_isolated_modality_generation()
        || audio_guider.do_isolated_modality_generation()
    {
        video_contexts.push(cond_context.clone());
        audio_contexts.push(audio_context.cloned());
        video_masks.push(cond_mask.cloned());
        audio_masks.push(cond_mask.cloned());
        perturbations.push(PerturbationConfig::new(vec![
            Perturbation::new(PerturbationType::SkipA2VCrossAttention, None),
            Perturbation::new(PerturbationType::SkipV2ACrossAttention, None),
        ]));
        modality_index = Some(perturbations.len() - 1);
    }

    let repeat_count = perturbations.len();
    let batch = video_latents.dim(0)?;
    let batched_video_context = Tensor::cat(&video_contexts.iter().collect::<Vec<_>>(), 0)?;
    let batched_audio_context = cat_optional_batches(&audio_contexts)?;
    let batched_video_mask = cat_optional_batches(&video_masks)?;
    let batched_audio_mask = cat_optional_batches(&audio_masks)?;
    let batched_video_latents = repeat_batch(video_latents, repeat_count)?;
    let batched_video_sigma = repeat_batch(video_sigma, repeat_count)?;
    let batched_video_timestep = repeat_batch(video_timestep, repeat_count)?;
    let batched_video_positions = repeat_batch(video_positions, repeat_count)?;
    let batched_video_self_attention_mask = video_self_attention_mask
        .map(|mask| repeat_batch(mask, repeat_count))
        .transpose()?;
    let batched_audio_latents = audio_latents
        .map(|latents| repeat_batch(latents, repeat_count))
        .transpose()?;
    let batched_audio_sigma = audio_sigma
        .map(|sigma| repeat_batch(sigma, repeat_count))
        .transpose()?;
    let batched_audio_timestep = audio_timestep
        .map(|timestep| repeat_batch(timestep, repeat_count))
        .transpose()?;
    let batched_audio_positions = audio_positions
        .map(|positions| repeat_batch(positions, repeat_count))
        .transpose()?;

    let (all_video_velocity, all_audio_velocity) = transformer.forward(
        &batched_video_latents,
        batched_audio_latents.as_ref(),
        &batched_video_context,
        batched_audio_context.as_ref(),
        &batched_video_sigma,
        &batched_video_timestep,
        batched_audio_sigma.as_ref(),
        batched_audio_timestep.as_ref(),
        batched_video_mask.as_ref(),
        batched_audio_mask.as_ref(),
        batched_video_self_attention_mask.as_ref(),
        None,
        &batched_video_positions,
        batched_audio_positions.as_ref(),
        Some(&BatchedPerturbationConfig::new(perturbations)),
    )?;

    let cond_video = denoised_from_velocity_with_sigma(
        video_latents,
        &split_batch_chunk(&all_video_velocity, cond_index, batch)?,
        video_timestep,
    )?;
    let uncond_video = if let Some(index) = uncond_index {
        denoised_from_velocity_with_sigma(
            video_latents,
            &split_batch_chunk(&all_video_velocity, index, batch)?,
            video_timestep,
        )?
    } else {
        cond_video.clone()
    };
    let perturbed_video = if let Some(index) = perturbed_index {
        denoised_from_velocity_with_sigma(
            video_latents,
            &split_batch_chunk(&all_video_velocity, index, batch)?,
            video_timestep,
        )?
    } else {
        cond_video.clone()
    };
    let modality_video = if let Some(index) = modality_index {
        denoised_from_velocity_with_sigma(
            video_latents,
            &split_batch_chunk(&all_video_velocity, index, batch)?,
            video_timestep,
        )?
    } else {
        cond_video.clone()
    };
    let video_denoised = if video_skip {
        cond_video.clone()
    } else {
        video_guider.calculate(
            &cond_video,
            &uncond_video,
            &perturbed_video,
            &modality_video,
        )?
    };

    let audio_denoised = match (
        audio_latents,
        all_audio_velocity.as_ref(),
        audio_timestep,
        batched_audio_positions.as_ref(),
    ) {
        (Some(audio_latents), Some(all_audio_velocity), Some(audio_timestep), Some(_)) => {
            let cond_audio = denoised_from_velocity_with_sigma(
                audio_latents,
                &split_batch_chunk(all_audio_velocity, cond_index, batch)?,
                audio_timestep,
            )?;
            let uncond_audio = if let Some(index) = uncond_index {
                denoised_from_velocity_with_sigma(
                    audio_latents,
                    &split_batch_chunk(all_audio_velocity, index, batch)?,
                    audio_timestep,
                )?
            } else {
                cond_audio.clone()
            };
            let perturbed_audio = if let Some(index) = perturbed_index {
                denoised_from_velocity_with_sigma(
                    audio_latents,
                    &split_batch_chunk(all_audio_velocity, index, batch)?,
                    audio_timestep,
                )?
            } else {
                cond_audio.clone()
            };
            let modality_audio = if let Some(index) = modality_index {
                denoised_from_velocity_with_sigma(
                    audio_latents,
                    &split_batch_chunk(all_audio_velocity, index, batch)?,
                    audio_timestep,
                )?
            } else {
                cond_audio.clone()
            };
            Some(if audio_skip {
                cond_audio
            } else {
                audio_guider.calculate(
                    &cond_audio,
                    &uncond_audio,
                    &perturbed_audio,
                    &modality_audio,
                )?
            })
        }
        _ => None,
    };

    Ok((video_denoised, audio_denoised))
}

fn convert_velocity_to_x0(sample: &Tensor, velocity: &Tensor, sigma: f32) -> Result<Tensor> {
    sample
        .to_dtype(DType::F32)?
        .broadcast_sub(&velocity.to_dtype(DType::F32)?.affine(sigma as f64, 0.0)?)
        .map_err(Into::into)
}

fn convert_x0_to_velocity(sample: &Tensor, denoised: &Tensor, sigma: f32) -> Result<Tensor> {
    if sigma.abs() <= f32::EPSILON {
        anyhow::bail!("cannot convert x0 to velocity at zero sigma");
    }
    sample
        .to_dtype(DType::F32)?
        .broadcast_sub(&denoised.to_dtype(DType::F32)?)?
        .affine(1.0 / sigma as f64, 0.0)
        .map_err(Into::into)
}

fn guided_velocity_from_cfg(
    sample: &Tensor,
    conditional_velocity: &Tensor,
    unconditional_velocity: &Tensor,
    sigma: f32,
    guidance_scale: f64,
) -> Result<Tensor> {
    if guidance_scale <= 1.0 {
        return Ok(conditional_velocity.clone());
    }
    let conditional_x0 = convert_velocity_to_x0(sample, conditional_velocity, sigma)?;
    let unconditional_x0 = convert_velocity_to_x0(sample, unconditional_velocity, sigma)?;
    let guidance_delta = conditional_x0
        .broadcast_sub(&unconditional_x0)?
        .affine(guidance_scale - 1.0, 0.0)?;
    let guided_x0 = conditional_x0.broadcast_add(&guidance_delta)?;
    convert_x0_to_velocity(sample, &guided_x0, sigma)
}

fn load_ltx2_av_transformer(
    plan: &Ltx2GeneratePlan,
    device: &candle_core::Device,
) -> Result<Ltx2AvTransformer3DModel> {
    load_ltx2_av_transformer_with_loras(plan, device, &[])
}

fn load_ltx2_av_transformer_with_loras(
    plan: &Ltx2GeneratePlan,
    device: &candle_core::Device,
    loras: &[LoraWeight],
) -> Result<Ltx2AvTransformer3DModel> {
    let force_streaming = std::env::var_os("MOLD_LTX2_FORCE_STREAMING").is_some();
    let force_eager = std::env::var_os("MOLD_LTX2_FORCE_EAGER").is_some();
    let config = ltx2_video_transformer_config(plan);
    let lora_registry = super::lora::load_lora_registry(loras)?;
    let vb = if ltx2_checkpoint_is_fp8(plan) {
        load_fp8_safetensors(
            std::slice::from_ref(&Path::new(&plan.checkpoint_path)),
            device,
            "LTX-2 transformer",
            &ProgressReporter::default(),
        )?
    } else {
        let dtype = transformer_weight_dtype(plan, device);
        unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&Path::new(&plan.checkpoint_path)),
                dtype,
                device,
            )?
        }
    };
    let vb = vb.rename_f(remap_ltx2_transformer_key);
    if device.is_cuda() && ltx2_checkpoint_is_fp8(plan) && force_eager && !force_streaming {
        Ok(Ltx2AvTransformer3DModel::new(&config, vb, lora_registry)?)
    } else {
        Ok(Ltx2AvTransformer3DModel::new_streaming(
            &config,
            vb,
            lora_registry,
        )?)
    }
}

fn load_ltx2_video_vae(
    plan: &Ltx2GeneratePlan,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<AutoencoderKLLtx2Video> {
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&Path::new(&plan.checkpoint_path)),
            dtype,
            device,
        )?
    };
    Ok(AutoencoderKLLtx2Video::new(
        ltx2_video_vae_config(plan),
        vb.pp("vae"),
    )?)
}

fn ltx2_video_transformer_config(plan: &Ltx2GeneratePlan) -> Ltx2VideoTransformer3DModelConfig {
    let cross_attention_adaln = plan.preset.transformer.cross_attention_adaln
        && !ltx_debug_disable_cross_attention_adaln_enabled();
    let apply_gated_attention = plan.preset.transformer.apply_gated_attention
        && !ltx_debug_disable_transformer_gated_attention_enabled();
    Ltx2VideoTransformer3DModelConfig {
        in_channels: plan.preset.transformer.in_channels,
        out_channels: plan.preset.transformer.out_channels,
        patch_size: 1,
        patch_size_t: 1,
        num_attention_heads: plan.preset.transformer.num_attention_heads,
        attention_head_dim: plan.preset.transformer.attention_head_dim,
        cross_attention_dim: plan.preset.transformer.cross_attention_dim,
        num_layers: plan.preset.transformer.num_layers,
        qk_norm: "rms_norm".to_string(),
        norm_elementwise_affine: false,
        norm_eps: 1e-6,
        caption_channels: plan.preset.video_connector_inner_dim(),
        caption_projection_in_transformer: matches!(
            plan.preset.caption_projection,
            crate::ltx2::preset::CaptionProjectionPlacement::Transformer
        ),
        attention_bias: true,
        attention_out_bias: true,
        positional_embedding_theta: 10_000.0,
        positional_embedding_max_pos: vec![20, 2048, 2048],
        use_middle_indices_grid: true,
        rope_type: crate::ltx2::model::LtxRopeType::Split,
        double_precision_rope: true,
        audio_num_attention_heads: plan.preset.transformer.audio_num_attention_heads,
        audio_attention_head_dim: plan.preset.transformer.audio_attention_head_dim,
        audio_in_channels: plan.preset.transformer.audio_in_channels,
        audio_out_channels: plan.preset.transformer.audio_out_channels,
        audio_cross_attention_dim: plan.preset.transformer.audio_cross_attention_dim,
        audio_positional_embedding_max_pos: vec![20],
        apply_gated_attention,
        // Public LTX-2 checkpoints set this to 1000.0, which keeps the AV gate
        // branch on the same sigma*1000 scale as the main timestep embedding.
        av_ca_timestep_scale_multiplier: 1000.0,
        cross_attention_adaln,
        streaming_prefetch_count: plan.streaming_prefetch_count.unwrap_or(1) as usize,
    }
}

fn transformer_weight_dtype(_plan: &Ltx2GeneratePlan, device: &candle_core::Device) -> DType {
    // Public LTX-2 FP8 manifests keep transformer weights in float8 storage but
    // run the native Rust matmuls in the normal compute dtype after applying the
    // checkpoint-provided per-tensor weight scales.
    gpu_dtype(device)
}

fn ltx2_checkpoint_is_fp8(plan: &Ltx2GeneratePlan) -> bool {
    if plan.checkpoint_path.to_ascii_lowercase().contains("fp8") {
        return true;
    }
    let Ok(tensors) = (unsafe {
        candle_core::safetensors::MmapedSafetensors::multi(&[Path::new(&plan.checkpoint_path)])
    }) else {
        return false;
    };
    for key in [
        "model.diffusion_model.transformer_blocks.1.attn1.to_q.weight",
        "model.diffusion_model.transformer_blocks.1.ff.net.0.proj.weight",
    ] {
        if let Ok(tensor) = tensors.load(key, &candle_core::Device::Cpu) {
            return tensor.dtype() == DType::F8E4M3;
        }
    }
    false
}

fn ltx2_video_vae_config(plan: &Ltx2GeneratePlan) -> AutoencoderKLLtx2VideoConfig {
    if plan.preset.name == "ltx-2.3-22b" {
        AutoencoderKLLtx2VideoConfig::ltx2_22b()
    } else {
        AutoencoderKLLtx2VideoConfig::default()
    }
}

fn ltx2_scheduler_config() -> FlowMatchEulerDiscreteSchedulerConfig {
    FlowMatchEulerDiscreteSchedulerConfig {
        num_train_timesteps: 1000,
        shift: 1.0,
        use_dynamic_shifting: false,
        base_shift: Some(0.5),
        max_shift: Some(1.15),
        base_image_seq_len: Some(256),
        max_image_seq_len: Some(4096),
        invert_sigmas: false,
        shift_terminal: None,
        use_karras_sigmas: false,
        use_exponential_sigmas: false,
        use_beta_sigmas: false,
        time_shift_type: TimeShiftType::Exponential,
        stochastic_sampling: false,
    }
}

fn remap_ltx2_transformer_key(name: &str) -> String {
    let mapped = name
        .split('.')
        .map(|component| match component {
            "proj_in" => "patchify_proj",
            "time_embed" => "adaln_single",
            "norm_q" => "q_norm",
            "norm_k" => "k_norm",
            _ => component,
        })
        .collect::<Vec<_>>()
        .join(".");
    format!("model.diffusion_model.{mapped}")
}

fn denoised_from_velocity(sample: &Tensor, velocity: &Tensor, sigma: f32) -> Result<Tensor> {
    let velocity = if velocity.dtype() == sample.dtype() {
        velocity.clone()
    } else {
        velocity.to_dtype(sample.dtype())?
    };
    sample
        .broadcast_sub(&velocity.affine(sigma as f64, 0.0)?)
        .map_err(Into::into)
}

fn velocity_from_denoised(sample: &Tensor, denoised: &Tensor, sigma: f32) -> Result<Tensor> {
    if sigma == 0.0 {
        return Tensor::zeros_like(sample).map_err(Into::into);
    }
    let denoised = if denoised.dtype() == sample.dtype() {
        denoised.clone()
    } else {
        denoised.to_dtype(sample.dtype())?
    };
    sample
        .broadcast_sub(&denoised)?
        .affine(1.0 / sigma as f64, 0.0)
        .map_err(Into::into)
}

fn ltx_debug_enabled() -> bool {
    env::var_os("MOLD_LTX_DEBUG").is_some()
}

fn ltx_debug_timings_enabled() -> bool {
    env::var_os("MOLD_LTX2_DEBUG_TIMINGS").is_some()
}

fn log_debug_vram(label: &str) {
    if let Some(free) = free_vram_bytes(0) {
        eprintln!("[ltx2-debug] {label} free_vram={}", fmt_gb(free));
    } else {
        eprintln!("[ltx2-debug] {label} free_vram=unavailable");
    }
}

fn ltx_debug_compare_uncond_enabled() -> bool {
    env::var_os("MOLD_LTX_DEBUG_COMPARE_UNCOND").is_some()
}

fn ltx_debug_alt_prompt() -> Option<String> {
    env::var("MOLD_LTX_DEBUG_ALT_PROMPT")
        .ok()
        .map(|prompt| prompt.trim().to_string())
        .filter(|prompt| !prompt.is_empty())
}

fn ltx_debug_disable_audio_branch_enabled() -> bool {
    env::var_os("MOLD_LTX_DEBUG_DISABLE_AUDIO_BRANCH").is_some()
}

fn ltx_debug_disable_cross_attention_adaln_enabled() -> bool {
    env::var_os("MOLD_LTX_DEBUG_DISABLE_CROSS_ATTENTION_ADALN").is_some()
}

fn ltx_debug_disable_transformer_gated_attention_enabled() -> bool {
    env::var_os("MOLD_LTX2_DEBUG_DISABLE_TRANSFORMER_GATED_ATTENTION").is_some()
}

fn ltx_debug_log_file() -> &'static Mutex<Option<std::fs::File>> {
    static LOG_FILE: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();
    LOG_FILE.get_or_init(|| {
        let path = env::var_os("MOLD_LTX_DEBUG_FILE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp/mold-ltx2-debug.log"));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok();
        Mutex::new(file)
    })
}

fn log_tensor_stats(name: &str, tensor: &Tensor) -> Result<()> {
    let tensor = tensor.to_device(&candle_core::Device::Cpu)?;
    let tensor = tensor.to_dtype(DType::F32)?;
    let mean = tensor.flatten_all()?.mean_all()?.to_scalar::<f32>()?;
    let abs_mean = tensor
        .flatten_all()?
        .abs()?
        .mean_all()?
        .to_scalar::<f32>()?;
    let sq_mean = tensor
        .flatten_all()?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()?;
    let std = (sq_mean - mean * mean).max(0.0).sqrt();
    let line = format!(
        "[ltx2-debug] {name}: shape={:?} mean={mean:.6} abs_mean={abs_mean:.6} rms={:.6} std={std:.6}",
        tensor.dims(),
        sq_mean.sqrt(),
    );
    eprintln!("{line}");
    if let Ok(mut guard) = ltx_debug_log_file().lock() {
        if let Some(file) = guard.as_mut() {
            use std::io::Write;
            let _ = writeln!(file, "{line}");
        }
    }
    Ok(())
}

fn log_timing(label: &str, start: Instant) {
    if !ltx_debug_timings_enabled() {
        return;
    }
    eprintln!(
        "[ltx2-timing] {label} {:.3}s",
        start.elapsed().as_secs_f64()
    );
}

fn log_elapsed_secs(label: &str, elapsed_secs: f64) {
    if !ltx_debug_timings_enabled() {
        return;
    }
    eprintln!("[ltx2-timing] {label} {elapsed_secs:.3}s");
}

fn log_prompt_debug_stats(plan: &Ltx2GeneratePlan, prompt: &NativePromptEncoding) -> Result<()> {
    let cond = &plan.prompt_tokens.conditional;
    let uncond = &plan.prompt_tokens.unconditional;
    let cond_tail = cond
        .input_ids
        .iter()
        .rev()
        .take(12)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let cond_tail_mask = cond
        .attention_mask
        .iter()
        .rev()
        .take(12)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let uncond_tail = uncond
        .input_ids
        .iter()
        .rev()
        .take(12)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let uncond_tail_mask = uncond
        .attention_mask
        .iter()
        .rev()
        .take(12)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let token_line = format!(
        "[ltx2-debug] prompt_tokens cond_valid={} uncond_valid={} cond_tail_ids={cond_tail:?} cond_tail_mask={cond_tail_mask:?} uncond_tail_ids={uncond_tail:?} uncond_tail_mask={uncond_tail_mask:?}",
        cond.valid_len(),
        uncond.valid_len(),
    );
    eprintln!("{token_line}");
    if let Ok(mut guard) = ltx_debug_log_file().lock() {
        if let Some(file) = guard.as_mut() {
            use std::io::Write;
            let _ = writeln!(file, "{token_line}");
        }
    }

    log_tensor_stats("cond_video_context", &prompt.conditional.video_encoding)?;
    log_tensor_stats("uncond_video_context", &prompt.unconditional.video_encoding)?;
    log_tensor_pair_stats(
        "video_context",
        &prompt.conditional.video_encoding,
        &prompt.unconditional.video_encoding,
    )?;

    let cond_video_mask_valid = prompt
        .conditional
        .attention_mask
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let uncond_video_mask_valid = prompt
        .unconditional
        .attention_mask
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let mask_line = format!(
        "[ltx2-debug] prompt_masks cond_valid_tokens={cond_video_mask_valid:.0} uncond_valid_tokens={uncond_video_mask_valid:.0}"
    );
    eprintln!("{mask_line}");
    if let Ok(mut guard) = ltx_debug_log_file().lock() {
        if let Some(file) = guard.as_mut() {
            use std::io::Write;
            let _ = writeln!(file, "{mask_line}");
        }
    }

    if let (Some(cond_audio), Some(uncond_audio)) = (
        prompt.conditional.audio_encoding.as_ref(),
        prompt.unconditional.audio_encoding.as_ref(),
    ) {
        log_tensor_stats("cond_audio_context", cond_audio)?;
        log_tensor_stats("uncond_audio_context", uncond_audio)?;
        log_tensor_pair_stats("audio_context", cond_audio, uncond_audio)?;
    }

    Ok(())
}

fn log_alt_prompt_debug_stats(
    plan: &Ltx2GeneratePlan,
    primary: &EmbeddingsProcessorOutput,
    alternate: &EmbeddingsProcessorOutput,
) -> Result<()> {
    if !ltx_debug_enabled() {
        return Ok(());
    }
    let alt_prompt = ltx_debug_alt_prompt().unwrap_or_else(|| "<unset>".to_string());
    let line = format!(
        "[ltx2-debug] alternate_prompt primary={:?} alternate={alt_prompt:?}",
        plan.prompt
    );
    eprintln!("{line}");
    if let Ok(mut guard) = ltx_debug_log_file().lock() {
        if let Some(file) = guard.as_mut() {
            use std::io::Write;
            let _ = writeln!(file, "{line}");
        }
    }
    log_tensor_pair_stats(
        "alt_prompt_video_context",
        &primary.video_encoding,
        &alternate.video_encoding,
    )?;
    if let (Some(primary_audio), Some(alternate_audio)) = (
        primary.audio_encoding.as_ref(),
        alternate.audio_encoding.as_ref(),
    ) {
        log_tensor_pair_stats("alt_prompt_audio_context", primary_audio, alternate_audio)?;
    }
    Ok(())
}

fn log_tensor_pair_stats(name: &str, lhs: &Tensor, rhs: &Tensor) -> Result<()> {
    let delta = lhs.broadcast_sub(rhs)?;
    log_tensor_stats(&format!("{name}_delta"), &delta)?;
    let cosine = tensor_cosine_similarity(lhs, rhs)?;
    let l2 = tensor_l2_distance(lhs, rhs)?;
    let line = format!("[ltx2-debug] {name}_pair cosine={cosine:.6} l2={l2:.6}");
    eprintln!("{line}");
    if let Ok(mut guard) = ltx_debug_log_file().lock() {
        if let Some(file) = guard.as_mut() {
            use std::io::Write;
            let _ = writeln!(file, "{line}");
        }
    }
    Ok(())
}

fn log_distilled_prompt_sensitivity(
    stage: Option<&str>,
    step_idx: usize,
    sigma: f32,
    video_sample: &Tensor,
    conditional_video_velocity: &Tensor,
    unconditional_video_velocity: &Tensor,
    audio_sample: Option<&Tensor>,
    conditional_audio_velocity: Option<&Tensor>,
    unconditional_audio_velocity: Option<&Tensor>,
) -> Result<()> {
    if !ltx_debug_enabled() {
        return Ok(());
    }
    let prefix = format!(
        "{}_step{step_idx}_sigma{sigma:.6}",
        stage.unwrap_or("stage")
    );
    log_tensor_pair_stats(
        &format!("{prefix}_video_velocity_cond_vs_uncond"),
        conditional_video_velocity,
        unconditional_video_velocity,
    )?;
    let conditional_video_x0 =
        convert_velocity_to_x0(video_sample, conditional_video_velocity, sigma)?;
    let unconditional_video_x0 =
        convert_velocity_to_x0(video_sample, unconditional_video_velocity, sigma)?;
    log_tensor_pair_stats(
        &format!("{prefix}_video_x0_cond_vs_uncond"),
        &conditional_video_x0,
        &unconditional_video_x0,
    )?;

    if let (
        Some(audio_sample),
        Some(conditional_audio_velocity),
        Some(unconditional_audio_velocity),
    ) = (
        audio_sample,
        conditional_audio_velocity,
        unconditional_audio_velocity,
    ) {
        log_tensor_pair_stats(
            &format!("{prefix}_audio_velocity_cond_vs_uncond"),
            conditional_audio_velocity,
            unconditional_audio_velocity,
        )?;
        let conditional_audio_x0 =
            convert_velocity_to_x0(audio_sample, conditional_audio_velocity, sigma)?;
        let unconditional_audio_x0 =
            convert_velocity_to_x0(audio_sample, unconditional_audio_velocity, sigma)?;
        log_tensor_pair_stats(
            &format!("{prefix}_audio_x0_cond_vs_uncond"),
            &conditional_audio_x0,
            &unconditional_audio_x0,
        )?;
    }

    Ok(())
}

fn log_distilled_alternate_prompt_sensitivity(
    stage: Option<&str>,
    step_idx: usize,
    sigma: f32,
    video_sample: &Tensor,
    primary_video_velocity: &Tensor,
    alternate_video_velocity: &Tensor,
    audio_sample: Option<&Tensor>,
    primary_audio_velocity: Option<&Tensor>,
    alternate_audio_velocity: Option<&Tensor>,
) -> Result<()> {
    if !ltx_debug_enabled() {
        return Ok(());
    }
    let prefix = format!(
        "{}_step{step_idx}_sigma{sigma:.6}",
        stage.unwrap_or("stage")
    );
    log_tensor_pair_stats(
        &format!("{prefix}_video_velocity_prompt_vs_alt"),
        primary_video_velocity,
        alternate_video_velocity,
    )?;
    let primary_video_x0 = convert_velocity_to_x0(video_sample, primary_video_velocity, sigma)?;
    let alternate_video_x0 = convert_velocity_to_x0(video_sample, alternate_video_velocity, sigma)?;
    log_tensor_pair_stats(
        &format!("{prefix}_video_x0_prompt_vs_alt"),
        &primary_video_x0,
        &alternate_video_x0,
    )?;

    if let (Some(audio_sample), Some(primary_audio_velocity), Some(alternate_audio_velocity)) = (
        audio_sample,
        primary_audio_velocity,
        alternate_audio_velocity,
    ) {
        log_tensor_pair_stats(
            &format!("{prefix}_audio_velocity_prompt_vs_alt"),
            primary_audio_velocity,
            alternate_audio_velocity,
        )?;
        let primary_audio_x0 = convert_velocity_to_x0(audio_sample, primary_audio_velocity, sigma)?;
        let alternate_audio_x0 =
            convert_velocity_to_x0(audio_sample, alternate_audio_velocity, sigma)?;
        log_tensor_pair_stats(
            &format!("{prefix}_audio_x0_prompt_vs_alt"),
            &primary_audio_x0,
            &alternate_audio_x0,
        )?;
    }

    Ok(())
}

fn tensor_cosine_similarity(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    let lhs = lhs
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?;
    let rhs = rhs
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?;
    let lhs_flat = lhs.flatten_all()?;
    let rhs_flat = rhs.flatten_all()?;
    let dot = lhs_flat
        .broadcast_mul(&rhs_flat)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let lhs_norm = lhs_flat
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?
        .sqrt()
        .max(1e-12);
    let rhs_norm = rhs_flat
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?
        .sqrt()
        .max(1e-12);
    Ok(dot / (lhs_norm * rhs_norm))
}

fn tensor_l2_distance(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    Ok(lhs
        .broadcast_sub(rhs)?
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?
        .sqrt())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use mold_core::{
        GenerateRequest, LoraWeight, Ltx2SpatialUpscale, Ltx2TemporalUpscale, OutputFormat,
        TimeRange,
    };

    use super::{
        apply_stage_video_conditioning, apply_video_token_replacements,
        build_video_conditioning_self_attention_mask, clean_latents_for_conditioning,
        convert_velocity_to_x0, convert_x0_to_velocity, decoded_video_to_frames,
        effective_native_guidance_scale, emit_denoise_progress, guided_velocity_from_cfg,
        keyframe_only_conditioning, ltx2_video_transformer_config,
        reapply_stage_video_conditioning, should_inspect_step_velocity,
        source_image_only_conditioning, strip_appended_video_conditioning, Ltx2RuntimeSession,
        StageVideoConditioning, VideoTokenAppendCondition, VideoTokenReplacement,
        LTX2_AUDIO_LATENT_CHANNELS, LTX2_VIDEO_LATENT_CHANNELS,
    };
    use crate::ltx2::conditioning::{self, StagedConditioning};
    use crate::ltx2::model::VideoPixelShape;
    use crate::ltx2::plan::{Ltx2GeneratePlan, PipelineKind};
    use crate::ltx2::preset::preset_for_model;
    use crate::ltx2::text::connectors::PaddingSide;
    use crate::ltx2::text::encoder::{GemmaConfig, GemmaHiddenStateEncoder};
    use crate::ltx2::text::gemma::{EncodedPromptPair, PromptTokens};
    use crate::ltx2::text::prompt_encoder::{
        build_embeddings_processor, ConnectorSpec, NativePromptEncoder,
    };
    use crate::progress::{ProgressCallback, ProgressEvent};

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
            placement: None,
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
        Ltx2RuntimeSession::new(candle_core::Device::Cpu, prompt_encoder, 0)
    }

    fn build_plan(
        req: &GenerateRequest,
        preset: crate::ltx2::preset::Ltx2ModelPreset,
        conditioning: StagedConditioning,
    ) -> Ltx2GeneratePlan {
        let loras = crate::ltx2::lora::normalize_loras(req);
        let graph = crate::ltx2::execution::build_execution_graph(
            req,
            PipelineKind::Distilled,
            &conditioning,
            &preset,
            loras.len(),
        );
        Ltx2GeneratePlan {
            pipeline: PipelineKind::Distilled,
            preset,
            checkpoint_is_distilled: req.model.contains("distilled"),
            execution_graph: graph,
            checkpoint_path: "/tmp/ltx2.safetensors".to_string(),
            distilled_checkpoint_path: None,
            distilled_lora_path: None,
            spatial_upsampler_path: None,
            temporal_upsampler_path: None,
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
            guidance: req.guidance,
            quantization: Some("fp8-cast".to_string()),
            streaming_prefetch_count: Some(2),
            conditioning,
            loras,
            retake_range: req.retake_range.clone(),
            spatial_upscale: req.spatial_upscale,
            temporal_upscale: req.temporal_upscale,
        }
    }

    #[test]
    fn emit_denoise_progress_reports_progress_event() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        let callback: ProgressCallback = Box::new(move |event| {
            sink.lock().unwrap().push(event);
        });

        emit_denoise_progress(Some(&callback), 3, 8, std::time::Duration::from_millis(12));

        let events = events.lock().unwrap();
        assert!(matches!(
            events.as_slice(),
            [ProgressEvent::DenoiseStep {
                step: 3,
                total: 8,
                ..
            }]
        ));
    }

    fn rebuild_execution_graph(plan: &mut Ltx2GeneratePlan, req: &GenerateRequest) {
        plan.execution_graph = crate::ltx2::execution::build_execution_graph(
            req,
            plan.pipeline,
            &plan.conditioning,
            &plan.preset,
            plan.loras.len(),
        );
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
        assert_eq!(prepared.video_pixel_shape.width, 608);
        assert_eq!(prepared.video_pixel_shape.height, 352);
        assert_eq!(
            prepared.video_latent_shape.channels,
            LTX2_VIDEO_LATENT_CHANNELS
        );
        assert_eq!(prepared.video_latent_shape.frames, 13);
        assert_eq!(
            prepared.video_positions.dims4().unwrap(),
            (1, 3, 13 * 11 * 19, 2)
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

        let rendered = session.render_native_video(&plan, &prepared, None).unwrap();
        assert_eq!(rendered.frames.len(), 97);
        assert_eq!(rendered.frames[0].dimensions(), (1216, 704));
        assert!(rendered.has_audio);
        assert_eq!(rendered.audio_sample_rate, Some(48_000));
        assert_eq!(rendered.audio_channels, Some(2));
    }

    #[test]
    fn runtime_prepare_keeps_av_audio_latents_for_silent_outputs() {
        let req = req("ltx-2-19b-distilled:fp8", OutputFormat::Gif, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert!(prepared.audio_latent_shape.is_some());
        assert!(prepared.audio_positions.is_some());
        assert!(prepared.cross_modal_temporal_positions.is_some());

        let rendered = session.render_native_video(&plan, &prepared, None).unwrap();
        assert_eq!(rendered.frames.len(), 97);
        assert!(!rendered.has_audio);
        assert_eq!(rendered.audio_sample_rate, None);
        assert_eq!(rendered.audio_channels, None);
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

    #[test]
    fn runtime_prepare_uses_stage_one_shape_for_temporal_upscale() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.frames = Some(17);
        req.fps = Some(12);
        req.temporal_upscale = Some(Ltx2TemporalUpscale::X2);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();
        let rendered = session.render_native_video(&plan, &prepared, None).unwrap();

        assert_eq!(prepared.video_pixel_shape.frames, 9);
        assert_eq!(prepared.video_pixel_shape.fps as u32, 6);
        assert_eq!(rendered.frames.len(), 17);
        assert_eq!(rendered.frames[0].dimensions(), (1216, 704));
    }

    #[test]
    fn runtime_prepare_uses_stage_one_shape_for_spatial_upscale() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X2);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();
        let rendered = session.render_native_video(&plan, &prepared, None).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 608);
        assert_eq!(prepared.video_pixel_shape.height, 352);
        assert_eq!(rendered.frames[0].dimensions(), (1216, 704));
    }

    #[test]
    fn runtime_prepare_uses_stage_one_shape_for_x1_5_spatial_upscale() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X1_5);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 800);
        assert_eq!(prepared.video_pixel_shape.height, 480);
        assert_eq!(prepared.video_latent_shape.width, 25);
        assert_eq!(prepared.video_latent_shape.height, 15);
    }

    #[test]
    fn runtime_prepare_uses_half_resolution_shape_for_distilled_pipeline() {
        let req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 608);
        assert_eq!(prepared.video_pixel_shape.height, 352);
    }

    #[test]
    fn runtime_prepare_aligns_implicit_two_stage_shape_to_latent_grid_for_odd_sizes() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.width = 608;
        req.height = 352;
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 320);
        assert_eq!(prepared.video_pixel_shape.height, 192);
        assert_eq!(prepared.video_latent_shape.width, 10);
        assert_eq!(prepared.video_latent_shape.height, 6);
    }

    #[test]
    fn runtime_prepare_aligns_explicit_x2_spatial_upscale_shape_to_latent_grid_for_odd_sizes() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.width = 608;
        req.height = 352;
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X2);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 320);
        assert_eq!(prepared.video_pixel_shape.height, 192);
        assert_eq!(prepared.video_latent_shape.width, 10);
        assert_eq!(prepared.video_latent_shape.height, 6);
    }

    #[test]
    fn runtime_render_native_video_accepts_bf16_prompt_encodings() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let mut prepared = session.prepare(&plan).unwrap();
        prepared.prompt.conditional.video_encoding = prepared
            .prompt
            .conditional
            .video_encoding
            .to_dtype(DType::BF16)
            .unwrap();
        prepared.prompt.unconditional.video_encoding = prepared
            .prompt
            .unconditional
            .video_encoding
            .to_dtype(DType::BF16)
            .unwrap();
        prepared.prompt.conditional.audio_encoding = prepared
            .prompt
            .conditional
            .audio_encoding
            .take()
            .map(|tensor| tensor.to_dtype(DType::BF16).unwrap());

        let rendered = session.render_native_video(&plan, &prepared, None).unwrap();

        assert_eq!(rendered.frames.len(), 97);
        assert_eq!(rendered.frames[0].dimensions(), (1216, 704));
    }

    #[test]
    fn decoded_video_to_frames_resizes_decoded_shape_to_requested_pixels() {
        let video = Tensor::zeros((1, 3, 2, 320, 544), DType::F32, &Device::Cpu).unwrap();
        let pixel_shape = VideoPixelShape {
            batch: 1,
            frames: 2,
            height: 352,
            width: 608,
            fps: 12.0,
        };

        let frames = decoded_video_to_frames(&video, pixel_shape).unwrap();

        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].dimensions(), (608, 352));
        assert_eq!(frames[1].dimensions(), (608, 352));
    }

    #[test]
    fn runtime_video_transformer_config_matches_upstream_av_gate_timestep_scale() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let config = ltx2_video_transformer_config(&plan);

        assert_eq!(config.av_ca_timestep_scale_multiplier, 1000.0);
        assert_eq!(config.streaming_prefetch_count, 2);
    }

    #[test]
    fn velocity_x0_roundtrip_preserves_sample_velocity_pair() {
        let sample = Tensor::new(&[[10.0f32, 4.0]], &Device::Cpu).unwrap();
        let velocity = Tensor::new(&[[2.0f32, -1.0]], &Device::Cpu).unwrap();
        let sigma = 0.5f32;

        let x0 = convert_velocity_to_x0(&sample, &velocity, sigma).unwrap();
        let roundtrip = convert_x0_to_velocity(&sample, &x0, sigma).unwrap();

        let values = roundtrip.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((values[0] - 2.0).abs() < 1e-5);
        assert!((values[1] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn cfg_guidance_is_applied_in_x0_space_before_velocity_conversion() {
        let sample = Tensor::new(&[[10.0f32]], &Device::Cpu).unwrap();
        let conditional_velocity = Tensor::new(&[[2.0f32]], &Device::Cpu).unwrap();
        let unconditional_velocity = Tensor::new(&[[4.0f32]], &Device::Cpu).unwrap();

        let guided = guided_velocity_from_cfg(
            &sample,
            &conditional_velocity,
            &unconditional_velocity,
            0.5,
            3.0,
        )
        .unwrap();
        let value = guided.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];

        assert!((value + 2.0).abs() < 1e-5);
    }

    #[test]
    fn denoiser_helpers_cast_velocity_and_denoised_to_sample_dtype() {
        let sample = Tensor::new(&[[[10.0f32, 4.0]]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let velocity = Tensor::new(&[[[2.0f32, -1.0]]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let sigma = Tensor::new(&[[0.5f32]], &Device::Cpu).unwrap();

        let denoised =
            super::denoised_from_velocity_with_sigma(&sample, &velocity, &sigma).unwrap();
        let restored = super::velocity_from_denoised(&sample, &denoised, 0.5).unwrap();

        assert_eq!(denoised.dtype(), DType::F32);
        assert_eq!(restored.dtype(), DType::F32);
        let values = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((values[0] - 2.0).abs() < 1e-3);
        assert!((values[1] + 1.0).abs() < 1e-3);
    }

    #[test]
    fn step_velocity_inspection_is_debug_only() {
        assert!(!should_inspect_step_velocity(None));
        assert!(should_inspect_step_velocity(Some("stage1")));
    }

    #[test]
    fn distilled_runtime_forces_simple_denoiser_guidance() {
        let req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        assert_eq!(plan.guidance, 3.0);
        assert_eq!(effective_native_guidance_scale(&plan), 1.0);
    }

    #[test]
    fn non_distilled_runtime_preserves_requested_guidance() {
        let req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        plan.guidance = 4.5;

        assert_eq!(effective_native_guidance_scale(&plan), 4.5);
    }

    #[test]
    fn distilled_runtime_skips_unconditional_prompt_encoding() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Distilled;
        rebuild_execution_graph(&mut plan, &req);

        assert!(!super::prompt_requires_unconditional_context_for_plan(&plan).unwrap());
    }

    #[test]
    fn ic_lora_runtime_skips_unconditional_prompt_encoding() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ic-lora.safetensors".to_string(),
            scale: 1.0,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::IcLora;
        rebuild_execution_graph(&mut plan, &req);

        assert!(!super::prompt_requires_unconditional_context_for_plan(&plan).unwrap());
    }

    #[test]
    fn two_stage_runtime_keeps_unconditional_prompt_encoding_for_multimodal_guidance() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::prompt_requires_unconditional_context_for_plan(&plan).unwrap());
    }

    #[test]
    fn a2vid_runtime_keeps_unconditional_prompt_encoding_for_multimodal_guidance() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.audio_file = Some(b"RIFFtestWAVEfmt ".to_vec());
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::A2Vid;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::prompt_requires_unconditional_context_for_plan(&plan).unwrap());
    }

    #[test]
    fn stage_unconditional_context_follows_multimodal_guidance_at_guidance_one() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        plan.guidance = 1.0;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::stage_requires_unconditional_context(&plan, 0).unwrap());
        assert!(!super::stage_requires_unconditional_context(&plan, 1).unwrap());
    }

    #[test]
    fn runtime_session_prepare_consumes_prompt_encoder() {
        // The encoder is still consumed on first prepare() — the encoder
        // slot moves out to free VRAM for the transformer. But same-prompt
        // follow-up calls now short-circuit through `cached_prompt_encoding`
        // so chain stages that replicate the prompt can reuse the session
        // instead of erroring on a consumed encoder.
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        session.prepare(&plan).unwrap();

        // Encoder slot is empty post-take.
        assert!(session.prompt_encoder.is_none());
        // But `can_reuse_for` reports true because the cached encoding
        // matches the incoming plan's prompt tokens.
        assert!(session.can_reuse_for(&plan));
        // Same-prompt re-prepare succeeds from the cache.
        session
            .prepare(&plan)
            .expect("same-prompt cache hit must succeed");
    }

    #[test]
    fn runtime_session_prepare_rejects_encoder_reuse_with_different_prompt() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        session.prepare(&plan).unwrap();

        // Mutate the plan's prompt tokens so the cache key misses.
        let mut plan_alt = plan.clone();
        plan_alt.prompt_tokens.conditional.input_ids[0] =
            plan_alt.prompt_tokens.conditional.input_ids[0].wrapping_add(1);

        // can_reuse_for must report false for a fresh prompt because the
        // encoder has already been consumed.
        assert!(!session.can_reuse_for(&plan_alt));
        // And prepare() with the new plan fails explicitly so the caller
        // knows to drop the session and rebuild.
        assert!(session.prepare(&plan_alt).is_err());
    }

    #[test]
    fn remap_ltx2_transformer_key_rewrites_only_exact_path_segments() {
        assert_eq!(
            super::remap_ltx2_transformer_key("proj_in.weight"),
            "model.diffusion_model.patchify_proj.weight"
        );
        assert_eq!(
            super::remap_ltx2_transformer_key("blocks.0.norm_q.weight"),
            "model.diffusion_model.blocks.0.q_norm.weight"
        );
        assert_eq!(
            super::remap_ltx2_transformer_key("blocks.0.patchify_proj_in.weight"),
            "model.diffusion_model.blocks.0.patchify_proj_in.weight"
        );
        assert_eq!(
            super::remap_ltx2_transformer_key("blocks.0.norm_q_extra.weight"),
            "model.diffusion_model.blocks.0.norm_q_extra.weight"
        );
    }

    #[test]
    fn one_stage_runtime_keeps_requested_full_resolution_shape() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::OneStage;

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 1216);
        assert_eq!(prepared.video_pixel_shape.height, 704);
        assert_eq!(prepared.video_latent_shape.width, 38);
        assert_eq!(prepared.video_latent_shape.height, 22);
    }

    #[test]
    fn retake_runtime_keeps_requested_full_resolution_shape() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.retake_range = Some(TimeRange {
            start_seconds: 1.0,
            end_seconds: 2.0,
        });
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Retake;
        rebuild_execution_graph(&mut plan, &req);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 1216);
        assert_eq!(prepared.video_pixel_shape.height, 704);
        assert_eq!(prepared.video_latent_shape.width, 38);
        assert_eq!(prepared.video_latent_shape.height, 22);
    }

    #[test]
    fn ic_lora_runtime_keeps_requested_stage1_shape() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ic-lora.safetensors".to_string(),
            scale: 1.0,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::IcLora;
        rebuild_execution_graph(&mut plan, &req);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 608);
        assert_eq!(prepared.video_pixel_shape.height, 352);
        assert_eq!(prepared.video_latent_shape.width, 19);
        assert_eq!(prepared.video_latent_shape.height, 11);
    }

    #[test]
    fn supports_real_video_path_accepts_plain_silent_one_stage_runs() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::OneStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_plain_audio_one_stage_runs() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::OneStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_source_image_distilled_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        req.source_image = Some(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        assert!(source_image_only_conditioning(&plan));
        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_keyframe_two_stage_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        req.keyframes = Some(vec![
            mold_core::KeyframeCondition {
                frame: 8,
                image: vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
            },
            mold_core::KeyframeCondition {
                frame: 48,
                image: vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
            },
        ]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Keyframe;
        rebuild_execution_graph(&mut plan, &req);

        assert!(keyframe_only_conditioning(&plan));
        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_retake_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.retake_range = Some(TimeRange {
            start_seconds: 0.5,
            end_seconds: 1.25,
        });
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Retake;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn temporal_token_denoise_mask_marks_only_overlapping_tokens() {
        let positions = Tensor::from_vec(
            vec![0.0f32, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0],
            (1, 1, 4, 2),
            &Device::Cpu,
        )
        .unwrap();
        let range = TimeRange {
            start_seconds: 0.75,
            end_seconds: 1.6,
        };

        let mask =
            super::build_temporal_token_denoise_mask(&range, &positions, &Device::Cpu).unwrap();

        assert_eq!(
            mask.to_vec2::<f32>().unwrap(),
            vec![vec![0.0, 1.0, 1.0, 1.0]]
        );
    }

    #[test]
    fn timestep_from_sigma_and_mask_defaults_to_full_sigma_without_mask() {
        let timestep = super::timestep_from_sigma_and_mask(0.75, 2, None, &Device::Cpu).unwrap();

        assert_eq!(timestep.to_vec1::<f32>().unwrap(), vec![0.75, 0.75]);
    }

    #[test]
    fn timestep_from_sigma_and_mask_scales_per_token_when_masked() {
        let mask = Tensor::from_vec(vec![0.0f32, 0.25, 1.0], (1, 3), &Device::Cpu).unwrap();

        let timestep =
            super::timestep_from_sigma_and_mask(0.8, 1, Some(&mask), &Device::Cpu).unwrap();

        assert_eq!(
            timestep.to_vec2::<f32>().unwrap(),
            vec![vec![0.0, 0.2, 0.8]]
        );
    }

    #[test]
    fn apply_video_token_replacements_blends_source_tokens_into_sequence() {
        let latents = Tensor::from_vec(
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
            (1, 3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let replacement_tokens =
            Tensor::from_vec(vec![10.0f32, 20.0], (1, 1, 2), &Device::Cpu).unwrap();
        let replacement = VideoTokenReplacement {
            start_token: 1,
            tokens: replacement_tokens,
            strength: 0.25,
        };

        let replaced = apply_video_token_replacements(&latents, &[replacement]).unwrap();
        let values = replaced.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(values, vec![0.0, 1.0, 4.0, 7.25, 4.0, 5.0]);
    }

    #[test]
    fn stage_video_conditioning_appends_keyframe_tokens_and_restores_them() {
        let latents =
            Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (1, 2, 2), &Device::Cpu).unwrap();
        let positions = Tensor::from_vec(
            vec![
                0.0f32, 1.0, 1.0, 2.0, 10.0, 11.0, 11.0, 12.0, 20.0, 21.0, 21.0, 22.0,
            ],
            (1, 3, 2, 2),
            &Device::Cpu,
        )
        .unwrap();
        let conditioning = StageVideoConditioning {
            replacements: vec![VideoTokenReplacement {
                start_token: 0,
                tokens: Tensor::from_vec(vec![7.0f32, 8.0], (1, 1, 2), &Device::Cpu).unwrap(),
                strength: 1.0,
            }],
            appended: vec![VideoTokenAppendCondition {
                tokens: Tensor::from_vec(vec![9.0f32, 10.0], (1, 1, 2), &Device::Cpu).unwrap(),
                positions: Tensor::from_vec(
                    vec![30.0f32, 31.0, 40.0, 41.0, 50.0, 51.0],
                    (1, 3, 1, 2),
                    &Device::Cpu,
                )
                .unwrap(),
                strength: 1.0,
            }],
        };

        let (conditioned_latents, conditioned_positions) =
            apply_stage_video_conditioning(&latents, &positions, &conditioning).unwrap();
        assert_eq!(conditioned_latents.dims3().unwrap(), (1, 3, 2));
        assert_eq!(conditioned_positions.dims4().unwrap(), (1, 3, 3, 2));
        assert_eq!(
            conditioned_latents
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![7.0, 8.0, 2.0, 3.0, 9.0, 10.0]
        );

        let mutated = Tensor::from_vec(
            vec![0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0],
            (1, 3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let reapplied = reapply_stage_video_conditioning(&mutated, 2, &conditioning).unwrap();
        assert_eq!(
            reapplied.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![7.0, 8.0, 1.0, 1.0, 9.0, 10.0]
        );

        let stripped = strip_appended_video_conditioning(&reapplied, 2).unwrap();
        assert_eq!(stripped.dims3().unwrap(), (1, 2, 2));
        assert_eq!(
            stripped.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![7.0, 8.0, 1.0, 1.0]
        );
    }

    #[test]
    fn reapply_stage_video_conditioning_keeps_soft_appended_tokens() {
        let latents =
            Tensor::from_vec(vec![0.0f32, 0.0, 1.0, 1.0], (1, 2, 2), &Device::Cpu).unwrap();
        let conditioning = StageVideoConditioning {
            replacements: vec![],
            appended: vec![VideoTokenAppendCondition {
                tokens: Tensor::from_vec(vec![9.0f32, 10.0], (1, 1, 2), &Device::Cpu).unwrap(),
                positions: Tensor::from_vec(vec![30.0f32, 40.0, 50.0], (1, 3, 1, 1), &Device::Cpu)
                    .unwrap(),
                strength: 0.4,
            }],
        };

        let reapplied = reapply_stage_video_conditioning(&latents, 2, &conditioning).unwrap();
        assert_eq!(reapplied.dims3().unwrap(), (1, 3, 2));
        assert_eq!(
            reapplied.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 0.0, 1.0, 1.0, 9.0, 10.0]
        );
    }

    #[test]
    fn clean_latents_replace_soft_blended_positions_with_pure_source() {
        // Simulate the state after `apply_stage_video_conditioning` with
        // strength 0.75: at the replacement positions, `video_latents` already
        // holds `noise*0.25 + source*0.75`. The denoise-mask blend uses
        // `clean_latents` as the target it pulls those positions toward at
        // every step — so the clean target must be pure source, not the
        // pre-blended mix.
        let noise = [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0];
        let source = [10.0f32, 10.0];
        let strength = 0.75f32;
        let blended_first = [
            noise[0] * (1.0 - strength) + source[0] * strength,
            noise[1] * (1.0 - strength) + source[1] * strength,
        ];
        let soft_blended = Tensor::from_vec(
            vec![
                blended_first[0],
                blended_first[1],
                noise[2],
                noise[3],
                noise[4],
                noise[5],
            ],
            (1, 3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let conditioning = StageVideoConditioning {
            replacements: vec![VideoTokenReplacement {
                start_token: 0,
                tokens: Tensor::from_vec(source.to_vec(), (1, 1, 2), &Device::Cpu).unwrap(),
                strength: strength as f64,
            }],
            appended: vec![],
        };

        let clean = clean_latents_for_conditioning(&soft_blended, &conditioning).unwrap();
        let values = clean.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(
            values,
            vec![source[0], source[1], noise[2], noise[3], noise[4], noise[5]],
            "soft-blended replacement positions must be overwritten with the pure \
             source tokens; other positions must be preserved unchanged"
        );
    }

    #[test]
    fn clean_latents_passthrough_when_no_replacements() {
        let latents =
            Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (1, 2, 2), &Device::Cpu).unwrap();
        let conditioning = StageVideoConditioning::default();

        let clean = clean_latents_for_conditioning(&latents, &conditioning).unwrap();
        assert_eq!(
            clean.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn staged_latent_patchifies_to_same_token_shape_as_image_at_single_latent_frame() {
        // A 4-pixel-frame motion tail at 1216×704 output lands on a latent
        // block of shape [1, 128, 1, 22, 38]. The render-chain orchestrator
        // produces this block from the prior stage's denoise result; the
        // image-conditioning path produces the same shape after VAE encode.
        // Both must patchify to [1, T*H*W, C] = [1, 1*22*38, 128] tokens so
        // the downstream replacement pass sees them identically regardless
        // of which path produced them.
        let latents = Tensor::zeros(
            (1, LTX2_VIDEO_LATENT_CHANNELS, 1, 22, 38),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let patchifier = super::VideoLatentPatchifier::new(1);
        let tokens = patchifier.patchify(&latents).expect("patchify");
        assert_eq!(tokens.dims(), &[1, 22 * 38, LTX2_VIDEO_LATENT_CHANNELS]);
    }

    #[test]
    fn video_conditioning_self_attention_mask_blocks_cross_keyframe_attention() {
        let conditioning = StageVideoConditioning {
            replacements: vec![],
            appended: vec![
                VideoTokenAppendCondition {
                    tokens: Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2), &Device::Cpu).unwrap(),
                    positions: Tensor::zeros((1, 3, 1, 2), DType::F32, &Device::Cpu).unwrap(),
                    strength: 1.0,
                },
                VideoTokenAppendCondition {
                    tokens: Tensor::from_vec(vec![3.0f32, 4.0], (1, 1, 2), &Device::Cpu).unwrap(),
                    positions: Tensor::zeros((1, 3, 1, 2), DType::F32, &Device::Cpu).unwrap(),
                    strength: 1.0,
                },
            ],
        };

        let mask =
            build_video_conditioning_self_attention_mask(2, &conditioning, &Device::Cpu).unwrap();
        let values = mask
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(
            values,
            vec![
                1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 0.0, //
                1.0, 1.0, 0.0, 1.0, //
            ]
        );
    }

    #[test]
    fn scale_video_spatial_positions_multiplies_only_height_and_width_axes() {
        let positions = Tensor::from_vec(
            vec![
                0.5f32, 1.5, //
                10.0, 11.0, //
                20.0, 21.0,
            ],
            (1, 3, 1, 2),
            &Device::Cpu,
        )
        .unwrap();

        let scaled = super::scale_video_spatial_positions(&positions, 2).unwrap();

        assert_eq!(
            scaled.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.5, 1.5, 20.0, 22.0, 40.0, 42.0]
        );
    }

    #[test]
    fn supports_real_video_path_accepts_plain_silent_two_stage_runs() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_a2vid_two_stage_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.audio_file = Some(b"RIFFtestWAVEfmt ".to_vec());
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::A2Vid;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_ic_lora_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ic-lora.safetensors".to_string(),
            scale: 1.0,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::IcLora;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn a2vid_stage1_uses_positive_only_audio_guidance() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.audio_file = Some(b"RIFFtestWAVEfmt ".to_vec());
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::A2Vid;
        rebuild_execution_graph(&mut plan, &req);

        let (_video_params, audio_params) =
            super::stage_multimodal_guider_params(&plan, 0).unwrap();

        assert_eq!(
            audio_params,
            crate::ltx2::guidance::MultiModalGuiderParams::default()
        );
    }

    #[test]
    fn stage_lora_stack_adds_internal_distilled_lora_for_two_stage_second_pass() {
        let req = req("ltx-2.3-22b-dev:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        plan.distilled_lora_path = Some("/tmp/distilled-lora.safetensors".to_string());
        rebuild_execution_graph(&mut plan, &req);

        let loras = super::stage_lora_stack(&plan, 1).unwrap();

        assert_eq!(loras.len(), 1);
        assert_eq!(loras[0].path, "/tmp/distilled-lora.safetensors");
        assert_eq!(loras[0].scale, 1.0);
    }

    #[test]
    fn stage_lora_stack_skips_internal_distilled_lora_for_distilled_checkpoint() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        rebuild_execution_graph(&mut plan, &req);

        let loras = super::stage_lora_stack(&plan, 1).unwrap();

        assert!(loras.is_empty());
    }

    #[test]
    fn stage_lora_stack_skips_user_loras_for_ic_lora_second_pass() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ic-lora.safetensors".to_string(),
            scale: 0.8,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::IcLora;
        rebuild_execution_graph(&mut plan, &req);

        let stage1_loras = super::stage_lora_stack(&plan, 0).unwrap();
        let stage2_loras = super::stage_lora_stack(&plan, 1).unwrap();

        assert_eq!(stage1_loras.len(), 1);
        assert!(stage2_loras.is_empty());
        assert_eq!(super::stage_guidance_scale(&plan, 0).unwrap(), 1.0);
        assert_eq!(super::stage_guidance_scale(&plan, 1).unwrap(), 1.0);
    }

    #[test]
    fn two_stage_stage2_sigmas_use_fixed_distilled_subset() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        plan.num_inference_steps = 30;
        plan.distilled_lora_path = Some("/tmp/distilled-lora.safetensors".to_string());
        rebuild_execution_graph(&mut plan, &req);

        let sigmas = super::stage_sigmas_no_terminal(&plan, 1, &Device::Cpu).unwrap();

        assert_eq!(sigmas, vec![0.909375, 0.725, 0.421875]);
    }

    #[test]
    fn two_stage_hq_stage_defaults_match_upstream_runtime() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStageHq;
        plan.num_inference_steps = 6;
        plan.distilled_lora_path = Some("/tmp/distilled-lora.safetensors".to_string());
        rebuild_execution_graph(&mut plan, &req);

        let stage1_sigmas = super::stage_sigmas_no_terminal(&plan, 0, &Device::Cpu).unwrap();
        let stage2_sigmas = super::stage_sigmas_no_terminal(&plan, 1, &Device::Cpu).unwrap();
        let stage1_loras = super::stage_lora_stack(&plan, 0).unwrap();
        let stage2_loras = super::stage_lora_stack(&plan, 1).unwrap();

        assert_eq!(
            super::stage_sampler_mode(&plan, 0).unwrap(),
            crate::ltx2::execution::SamplerMode::Res2S
        );
        assert_eq!(
            super::stage_sampler_mode(&plan, 1).unwrap(),
            crate::ltx2::execution::SamplerMode::Res2S
        );
        assert_eq!(stage1_sigmas.len(), 6);
        assert!(stage1_sigmas.windows(2).all(|pair| pair[0] >= pair[1]));
        assert!(stage1_sigmas.last().copied().unwrap() > 0.0);
        assert_eq!(stage2_sigmas, vec![0.909375, 0.725, 0.421875]);
        assert_eq!(stage1_loras.len(), 1);
        assert_eq!(stage1_loras[0].scale, 0.25);
        assert_eq!(stage2_loras.len(), 1);
        assert_eq!(stage2_loras[0].scale, 0.5);
        assert_eq!(super::stage_guidance_scale(&plan, 1).unwrap(), 1.0);
    }

    #[test]
    fn supports_real_video_path_rejects_one_stage_audio_and_upscale_requests() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X2);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::OneStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(!super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_distilled_spatial_upscale_runs() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X1_5);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Distilled;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_distilled_temporal_upscale_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        req.temporal_upscale = Some(Ltx2TemporalUpscale::X2);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Distilled;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_two_stage_spatial_upscale_runs() {
        let mut req = req("ltx-2-19b:fp8", OutputFormat::Mp4, Some(false));
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X2);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn waveform_to_audio_track_interleaves_stereo_samples() {
        let waveform = Tensor::from_vec(
            vec![0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3],
            (1, 2, 3),
            &Device::Cpu,
        )
        .unwrap();

        let track = super::waveform_to_audio_track(&waveform, 48_000)
            .unwrap()
            .unwrap();

        assert_eq!(track.channels, 2);
        assert_eq!(track.sample_rate, 48_000);
        assert_eq!(
            track.interleaved_samples,
            vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3]
        );
    }
}
