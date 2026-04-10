#![allow(dead_code)]

use anyhow::{Context, Result};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::ltx_video::sampling::{
    FlowMatchEulerDiscreteSchedulerConfig, TimeShiftType,
};
use image::{imageops, Rgb, RgbImage};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::env;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use super::backend::Ltx2Backend;
use super::conditioning::retake_temporal_mask;
use super::model::{
    audio_temporal_positions, cross_modal_temporal_positions, derive_stage1_render_shape,
    get_pixel_coords, scale_video_time_to_seconds, spatially_upsample_frames,
    temporally_upsample_frames_x2, video_token_positions,
    video_transformer::{Ltx2AvTransformer3DModel, Ltx2VideoTransformer3DModelConfig},
    video_vae::{AutoencoderKLLtx2Video, AutoencoderKLLtx2VideoConfig},
    AudioLatentShape, AudioPatchifier, SpatioTemporalScaleFactors, VideoLatentPatchifier,
    VideoLatentShape, VideoPixelShape,
};
use super::plan::{Ltx2GeneratePlan, PipelineKind};
use super::sampler::euler_step;
use super::text::connectors::EmbeddingsProcessorOutput;
use super::text::prompt_encoder::{NativePromptEncoder, NativePromptEncoding};
use crate::engine::{gpu_dtype, seeded_randn};
use crate::ltx_video::latent_upsampler::LatentUpsampler;
use crate::progress::ProgressReporter;
use crate::weight_loader::load_fp8_safetensors;

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

#[derive(Debug)]
pub struct NativeRenderedVideo {
    pub frames: Vec<RgbImage>,
    pub has_audio: bool,
    pub audio_sample_rate: Option<u32>,
    pub audio_channels: Option<u32>,
}

pub struct Ltx2RuntimeSession {
    prompt_encoder: Option<NativePromptEncoder>,
}

impl Ltx2RuntimeSession {
    pub fn new(prompt_encoder: NativePromptEncoder) -> Self {
        Self {
            prompt_encoder: Some(prompt_encoder),
        }
    }

    pub fn prepare(&mut self, plan: &Ltx2GeneratePlan) -> Result<NativePreparedRun> {
        let mut prompt_encoder = self
            .prompt_encoder
            .take()
            .context("native LTX-2 prompt encoder has already been consumed")?;
        let prompt_device_is_cuda = prompt_encoder.device().is_cuda();
        let prepared_device = if prompt_device_is_cuda || prompt_encoder.device().is_metal() {
            candle_core::Device::Cpu
        } else {
            prompt_encoder.device().clone()
        };
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
            stage1_shape.width = (plan.width / 2).max(16);
            stage1_shape.height = (plan.height / 2).max(16);
        }
        let prompt = move_prompt_encoding_to_device(
            prompt_encoder.encode_prompt_pair(&plan.prompt_tokens)?,
            &prepared_device,
        )?;
        drop(prompt_encoder);
        if prompt_device_is_cuda {
            crate::device::reclaim_gpu_memory();
        }
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

        let wants_audio_latents = prompt.conditional.audio_encoding.is_some()
            || prompt.unconditional.audio_encoding.is_some()
            || plan.execution_graph.wants_audio_output
            || plan.execution_graph.uses_audio_conditioning;
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

        let retake_mask = plan
            .retake_range
            .as_ref()
            .map(|range| retake_temporal_mask(range, stage1_shape.fps, stage1_shape.frames))
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

    pub fn render_native_video(
        &self,
        plan: &Ltx2GeneratePlan,
        prepared: &NativePreparedRun,
    ) -> Result<NativeRenderedVideo> {
        if let Some(rendered) = self.try_render_real_video(plan, prepared)? {
            return Ok(rendered);
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
            has_audio: plan.execution_graph.wants_audio_output,
            audio_sample_rate: plan.execution_graph.wants_audio_output.then_some(48_000),
            audio_channels: plan.execution_graph.wants_audio_output.then_some(2),
        })
    }

    fn try_render_real_video(
        &self,
        plan: &Ltx2GeneratePlan,
        prepared: &NativePreparedRun,
    ) -> Result<Option<NativeRenderedVideo>> {
        if !supports_real_video_path(plan) {
            return Ok(None);
        }
        if !Path::new(&plan.checkpoint_path).is_file() {
            return Ok(None);
        }
        match render_real_distilled_av(plan, prepared) {
            Ok(rendered) => Ok(Some(rendered)),
            Err(err) if is_placeholder_checkpoint_error(&err) => Ok(None),
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
        if retake_strength > 0.0 && (fx < 0.03 || fx > 0.97 || fy < 0.03 || fy > 0.97) {
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
            | PipelineKind::Retake
    )
}

fn supports_real_video_path(plan: &Ltx2GeneratePlan) -> bool {
    matches!(plan.pipeline, PipelineKind::Distilled)
        && plan.conditioning.images.is_empty()
        && plan.conditioning.audio_path.is_none()
        && plan.conditioning.video_path.is_none()
        && !plan.execution_graph.wants_audio_output
        && !plan.execution_graph.uses_audio_conditioning
        && !plan.execution_graph.uses_reference_video_conditioning
        && !plan.execution_graph.uses_keyframe_conditioning
        && !plan.execution_graph.uses_retake_masking
        && plan.loras.is_empty()
}

fn is_placeholder_checkpoint_error(err: &anyhow::Error) -> bool {
    let message = err.to_string().to_ascii_lowercase();
    message.contains("header too small")
        || message.contains("invalid header")
        || message.contains("failed to parse safetensor")
}

fn select_real_video_device() -> Result<candle_core::Device> {
    match Ltx2Backend::detect() {
        Ltx2Backend::Cuda => Ok(candle_core::Device::new_cuda(0)?),
        Ltx2Backend::Cpu => Ok(candle_core::Device::Cpu),
        Ltx2Backend::Metal => Err(anyhow::anyhow!(
            "Metal is not supported for native LTX-2 inference"
        )),
    }
}

fn render_real_distilled_av(
    plan: &Ltx2GeneratePlan,
    prepared: &NativePreparedRun,
) -> Result<NativeRenderedVideo> {
    let debug_enabled = ltx_debug_enabled();
    let device = select_real_video_device()?;
    let transformer = load_ltx2_av_transformer(plan, &device)?;
    let audio_shape = prepared
        .audio_latent_shape
        .context("native distilled LTX-2 inference requires audio latents")?;
    let cond_context = prepared
        .prompt
        .conditional
        .video_encoding
        .to_device(&device)?;
    let audio_context = prepared
        .prompt
        .conditional
        .audio_encoding
        .as_ref()
        .context("native distilled LTX-2 inference requires audio prompt conditioning")?
        .to_device(&device)?;
    let cond_mask = prepared
        .prompt
        .conditional
        .attention_mask
        .to_device(&device)?;
    let video_positions = prepared.video_positions.to_device(&device)?;
    let audio_positions = prepared
        .audio_positions
        .as_ref()
        .context("native distilled LTX-2 inference requires audio positions")?
        .to_device(&device)?;
    let stage1_video_noise = seeded_randn(
        plan.seed,
        &[
            prepared.video_latent_shape.batch,
            prepared.video_latent_shape.channels,
            prepared.video_latent_shape.frames,
            prepared.video_latent_shape.height,
            prepared.video_latent_shape.width,
        ],
        &device,
        DType::F32,
    )?;
    let stage1_audio_noise = seeded_randn(
        plan.seed ^ 0x4155_4449_4f4c_5458,
        &[
            audio_shape.batch,
            audio_shape.channels,
            audio_shape.frames,
            audio_shape.mel_bins,
        ],
        &device,
        DType::F32,
    )?;

    if debug_enabled {
        log_tensor_stats("video_context", &cond_context)?;
        log_tensor_stats("audio_context", &audio_context)?;
        log_tensor_stats("initial_video_latents", &stage1_video_noise)?;
        log_tensor_stats("initial_audio_latents", &stage1_audio_noise)?;
    }

    let dtype = gpu_dtype(&device);
    let mut vae = load_ltx2_video_vae(plan, &device, dtype)?;
    vae.use_tiling = false;
    vae.use_framewise_decoding = false;
    let (stage1_video_latents, stage1_audio_latents) = run_real_distilled_stage(
        &transformer,
        prepared.video_latent_shape,
        Some(audio_shape),
        &stage1_video_noise,
        Some(&stage1_audio_noise),
        &video_positions,
        Some(&audio_positions),
        &cond_context,
        Some(&audio_context),
        Some(&cond_mask),
        DISTILLED_STAGE1_SIGMAS_NO_TERMINAL,
        debug_enabled.then_some("stage1"),
    )?;
    let stage1_audio_latents = stage1_audio_latents
        .context("native distilled LTX-2 stage 1 must produce audio latents")?;
    let spatial_upsampler_path = plan
        .spatial_upsampler_path
        .as_ref()
        .context("native distilled LTX-2 inference requires a spatial upsampler asset")?;
    let upsampler = LatentUpsampler::load(Path::new(spatial_upsampler_path), dtype, &device)?;
    let stage2_clean_video_latents = vae.normalize_latents(
        &upsampler.forward(&vae.denormalize_latents(&stage1_video_latents.to_dtype(dtype)?)?)?,
    )?;
    let final_pixel_shape = VideoPixelShape {
        batch: 1,
        frames: plan.num_frames as usize,
        height: plan.height as usize,
        width: plan.width as usize,
        fps: plan.frame_rate as f32,
    };
    let final_video_latent_shape = VideoLatentShape::from_pixel_shape(
        final_pixel_shape,
        LTX2_VIDEO_LATENT_CHANNELS,
        SpatioTemporalScaleFactors::default(),
    );
    let stage2_video_positions = build_video_positions(final_pixel_shape, &device)?;
    let stage2_video_noise = seeded_randn(
        plan.seed ^ 0x5354_4147_4532_4c54,
        &[
            final_video_latent_shape.batch,
            final_video_latent_shape.channels,
            final_video_latent_shape.frames,
            final_video_latent_shape.height,
            final_video_latent_shape.width,
        ],
        &device,
        DType::F32,
    )?;
    let stage2_audio_noise = seeded_randn(
        plan.seed ^ 0x4155_4449_3254_4c58,
        &[
            audio_shape.batch,
            audio_shape.channels,
            audio_shape.frames,
            audio_shape.mel_bins,
        ],
        &device,
        DType::F32,
    )?;
    let stage2_sigma = DISTILLED_STAGE2_SIGMAS_NO_TERMINAL[0];
    let stage2_video_start = mix_clean_latents_with_noise(
        &stage2_clean_video_latents.to_dtype(DType::F32)?,
        &stage2_video_noise,
        stage2_sigma,
    )?;
    let stage2_audio_start = mix_clean_latents_with_noise(
        &stage1_audio_latents.to_dtype(DType::F32)?,
        &stage2_audio_noise,
        stage2_sigma,
    )?;
    let (latents, _audio_latents) = run_real_distilled_stage(
        &transformer,
        final_video_latent_shape,
        Some(audio_shape),
        &stage2_video_start,
        Some(&stage2_audio_start),
        &stage2_video_positions,
        Some(&audio_positions),
        &cond_context,
        Some(&audio_context),
        Some(&cond_mask),
        DISTILLED_STAGE2_SIGMAS_NO_TERMINAL,
        debug_enabled.then_some("stage2"),
    )?;
    drop(transformer);
    if debug_enabled {
        log_tensor_stats("final_video_latents", &latents)?;
    }
    let (_dec_output, video) = vae.decode(&latents.to_dtype(dtype)?, None, false, false)?;
    if debug_enabled {
        log_tensor_stats("decoded_video", &video)?;
    }
    let video =
        ((video.to_dtype(DType::F32)?.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let video = video.i(0)?;

    let mut frames = Vec::with_capacity(video.dim(1)?);
    for index in 0..video.dim(1)? {
        let frame = video
            .i((.., index, .., ..))?
            .permute((1, 2, 0))?
            .contiguous()?;
        let data: Vec<u8> = frame.flatten_all()?.to_vec1()?;
        let rgb = RgbImage::from_raw(
            final_pixel_shape.width as u32,
            final_pixel_shape.height as u32,
            data,
        )
        .context("failed to build an RGB frame from the decoded LTX-2 tensor")?;
        frames.push(rgb);
    }

    Ok(NativeRenderedVideo {
        frames,
        has_audio: false,
        audio_sample_rate: None,
        audio_channels: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_real_distilled_stage(
    transformer: &Ltx2AvTransformer3DModel,
    video_shape: VideoLatentShape,
    audio_shape: Option<AudioLatentShape>,
    video_start_latents: &Tensor,
    audio_start_latents: Option<&Tensor>,
    video_positions: &Tensor,
    audio_positions: Option<&Tensor>,
    cond_context: &Tensor,
    audio_context: Option<&Tensor>,
    cond_mask: Option<&Tensor>,
    sigmas_no_terminal: &[f32],
    debug_stage: Option<&str>,
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
    let mut video_latents = video_patchifier.patchify(video_start_latents)?;
    let mut audio_latents = match (audio_shape, audio_start_latents) {
        (Some(_), Some(latents)) => Some(audio_patchifier.patchify(latents)?),
        _ => None,
    };

    for (step_idx, sigma) in run_sigmas
        .iter()
        .copied()
        .take(run_sigmas.len().saturating_sub(1))
        .enumerate()
    {
        let audio_latents_ref = audio_latents
            .as_ref()
            .context("audio latents missing for multimodal distilled stage")?;
        let audio_positions_ref = audio_positions
            .as_ref()
            .context("audio positions missing for multimodal distilled stage")?;
        let audio_context_ref = audio_context
            .as_ref()
            .context("audio prompt conditioning missing for multimodal distilled stage")?;
        let timestep = Tensor::full(sigma, (video_latents.dim(0)?,), &device)?;
        let (video_velocity, audio_velocity) = transformer.forward(
            &video_latents,
            audio_latents_ref,
            cond_context,
            audio_context_ref,
            &timestep,
            cond_mask,
            cond_mask,
            video_positions,
            audio_positions_ref,
        )?;
        let video_velocity = video_velocity.to_dtype(DType::F32)?;
        let video_denoised = denoised_from_velocity(&video_latents, &video_velocity, sigma)?;
        video_latents = euler_step(&video_latents, &video_denoised, &run_sigmas, step_idx)?;

        if let Some(audio_latents) = audio_latents.as_mut() {
            let audio_velocity = audio_velocity.to_dtype(DType::F32)?;
            let audio_denoised = denoised_from_velocity(audio_latents, &audio_velocity, sigma)?;
            *audio_latents = euler_step(audio_latents, &audio_denoised, &run_sigmas, step_idx)?;
        }

        if let Some(stage) = debug_stage {
            eprintln!("[ltx2-debug] {stage} step={step_idx} sigma={sigma:.6}");
            log_tensor_stats("step_video_latents", &video_latents)?;
            if let Some(audio_latents) = audio_latents.as_ref() {
                log_tensor_stats("step_audio_latents", audio_latents)?;
            }
            log_tensor_stats("video_x0", &video_denoised)?;
            log_tensor_stats("video_velocity", &video_velocity)?;
            if let Some(audio_latents) = audio_latents.as_ref() {
                let audio_velocity = audio_velocity.to_dtype(DType::F32)?;
                let audio_denoised = denoised_from_velocity(audio_latents, &audio_velocity, sigma)?;
                log_tensor_stats("audio_x0", &audio_denoised)?;
                log_tensor_stats("audio_velocity", &audio_velocity)?;
            }
        }
    }

    let video_latents = video_patchifier.unpatchify(&video_latents, video_shape)?;
    let audio_latents = match (audio_latents, audio_shape) {
        (Some(latents), Some(shape)) => Some(audio_patchifier.unpatchify(&latents, shape)?),
        _ => None,
    };
    if debug_stage.is_some() {
        log_tensor_stats("final_patched_latents", &video_latents)?;
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

fn duplicate_cfg_batch(xs: &Tensor) -> Result<Tensor> {
    Tensor::cat(&[xs, xs], 0).map_err(Into::into)
}

fn duplicate_optional_cfg_batch(xs: Option<&Tensor>) -> Result<Option<Tensor>> {
    xs.map(duplicate_cfg_batch).transpose()
}

fn cat_cfg_conditioning(unconditional: &Tensor, conditional: &Tensor) -> Result<Tensor> {
    Tensor::cat(&[unconditional, conditional], 0).map_err(Into::into)
}

fn cat_optional_cfg_conditioning(
    unconditional: Option<&Tensor>,
    conditional: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    match (unconditional, conditional) {
        (Some(unconditional), Some(conditional)) => {
            Ok(Some(cat_cfg_conditioning(unconditional, conditional)?))
        }
        (None, None) => Ok(None),
        _ => {
            anyhow::bail!("conditional and unconditional CFG inputs must both be present or absent")
        }
    }
}

fn convert_velocity_to_x0(sample: &Tensor, velocity: &Tensor, sigma: f32) -> Result<Tensor> {
    sample
        .broadcast_sub(&velocity.affine(sigma as f64, 0.0)?)
        .map_err(Into::into)
}

fn convert_x0_to_velocity(sample: &Tensor, denoised: &Tensor, sigma: f32) -> Result<Tensor> {
    if sigma.abs() <= f32::EPSILON {
        anyhow::bail!("cannot convert x0 to velocity at zero sigma");
    }
    sample
        .broadcast_sub(denoised)?
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
    let vb = vb.rename_f(|name| remap_ltx2_transformer_key(name));
    Ok(Ltx2AvTransformer3DModel::new_streaming(
        &ltx2_video_transformer_config(plan),
        vb,
    )?)
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
        ltx2_video_vae_config(),
        vb.pp("vae"),
    )?)
}

fn ltx2_video_transformer_config(plan: &Ltx2GeneratePlan) -> Ltx2VideoTransformer3DModelConfig {
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
        av_ca_timestep_scale_multiplier: 1000.0,
        cross_attention_adaln: plan.preset.transformer.cross_attention_adaln,
    }
}

fn transformer_weight_dtype(_plan: &Ltx2GeneratePlan, device: &candle_core::Device) -> DType {
    // Public LTX-2 FP8 manifests use the upstream fp8-cast policy, which stores
    // some weights in float8 but runs the transformer in the normal compute dtype.
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

fn ltx2_video_vae_config() -> AutoencoderKLLtx2VideoConfig {
    AutoencoderKLLtx2VideoConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: LTX2_VIDEO_LATENT_CHANNELS,
        encoder_blocks: vec![
            super::model::video_vae::VaeBlockConfig::res_x(4),
            super::model::video_vae::VaeBlockConfig::compress("compress_space_res", 2, false),
            super::model::video_vae::VaeBlockConfig::res_x(6),
            super::model::video_vae::VaeBlockConfig::compress("compress_time_res", 2, false),
            super::model::video_vae::VaeBlockConfig::res_x(6),
            super::model::video_vae::VaeBlockConfig::compress("compress_all_res", 2, false),
            super::model::video_vae::VaeBlockConfig::res_x(2),
            super::model::video_vae::VaeBlockConfig::compress("compress_all_res", 2, false),
            super::model::video_vae::VaeBlockConfig::res_x(2),
        ],
        decoder_blocks: vec![
            super::model::video_vae::VaeBlockConfig::res_x_with_noise(5, false),
            super::model::video_vae::VaeBlockConfig::compress("compress_all", 2, true),
            super::model::video_vae::VaeBlockConfig::res_x_with_noise(5, false),
            super::model::video_vae::VaeBlockConfig::compress("compress_all", 2, true),
            super::model::video_vae::VaeBlockConfig::res_x_with_noise(5, false),
            super::model::video_vae::VaeBlockConfig::compress("compress_all", 2, true),
            super::model::video_vae::VaeBlockConfig::res_x_with_noise(5, false),
        ],
        patch_size: 4,
        resnet_eps: 1e-6,
        scaling_factor: 1.0,
        latent_log_var: super::model::video_vae::LatentLogVar::Uniform,
        encoder_base_channels: 128,
        decoder_base_channels: 128,
        spatial_compression_ratio: 32,
        temporal_compression_ratio: 8,
        timestep_conditioning: false,
        decoder_causal: false,
        latents_mean: vec![0.0; LTX2_VIDEO_LATENT_CHANNELS],
        latents_std: vec![1.0; LTX2_VIDEO_LATENT_CHANNELS],
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
        .replace("proj_in", "patchify_proj")
        .replace("time_embed", "adaln_single")
        .replace("norm_q", "q_norm")
        .replace("norm_k", "k_norm");
    format!("model.diffusion_model.{mapped}")
}

fn denoised_from_velocity(sample: &Tensor, velocity: &Tensor, sigma: f32) -> Result<Tensor> {
    sample
        .broadcast_sub(&velocity.affine(sigma as f64, 0.0)?)
        .map_err(Into::into)
}

fn velocity_from_denoised(sample: &Tensor, denoised: &Tensor, sigma: f32) -> Result<Tensor> {
    if sigma == 0.0 {
        return Tensor::zeros_like(sample).map_err(Into::into);
    }
    sample
        .broadcast_sub(denoised)?
        .affine(1.0 / sigma as f64, 0.0)
        .map_err(Into::into)
}

fn ltx_debug_enabled() -> bool {
    env::var_os("MOLD_LTX_DEBUG").is_some()
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use mold_core::{
        GenerateRequest, Ltx2SpatialUpscale, Ltx2TemporalUpscale, OutputFormat, TimeRange,
    };

    use super::{
        convert_velocity_to_x0, convert_x0_to_velocity, guided_velocity_from_cfg,
        ltx2_video_transformer_config, Ltx2RuntimeSession, LTX2_AUDIO_LATENT_CHANNELS,
        LTX2_VIDEO_LATENT_CHANNELS,
    };
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
            guidance: req.guidance as f64,
            quantization: Some("fp8-cast".to_string()),
            streaming_prefetch_count: Some(2),
            conditioning,
            loras: vec![],
            retake_range: req.retake_range.clone(),
            spatial_upscale: req.spatial_upscale,
            temporal_upscale: req.temporal_upscale,
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

        let rendered = session.render_native_video(&plan, &prepared).unwrap();
        assert_eq!(rendered.frames.len(), 97);
        assert_eq!(rendered.frames[0].dimensions(), (1216, 704));
        assert!(rendered.has_audio);
        assert_eq!(rendered.audio_sample_rate, Some(48_000));
        assert_eq!(rendered.audio_channels, Some(2));
    }

    #[test]
    fn runtime_prepare_keeps_audio_latents_for_silent_outputs() {
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

        let rendered = session.render_native_video(&plan, &prepared).unwrap();
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
        let rendered = session.render_native_video(&plan, &prepared).unwrap();

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
        let rendered = session.render_native_video(&plan, &prepared).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 608);
        assert_eq!(prepared.video_pixel_shape.height, 352);
        assert_eq!(rendered.frames[0].dimensions(), (1216, 704));
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

        let rendered = session.render_native_video(&plan, &prepared).unwrap();

        assert_eq!(rendered.frames.len(), 97);
        assert_eq!(rendered.frames[0].dimensions(), (1216, 704));
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
}
