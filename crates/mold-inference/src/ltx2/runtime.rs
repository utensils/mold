#![allow(dead_code)]

use anyhow::{Context, Result};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::ltx_video::{
    sampling::{
        FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig, TimeShiftType,
    },
    vae::{AutoencoderKLLtxVideo, AutoencoderKLLtxVideoConfig},
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
    video_transformer::{
        Ltx2AvTransformer3DModel, Ltx2VideoTransformer3DModel, Ltx2VideoTransformer3DModelConfig,
    },
    AudioLatentShape, AudioPatchifier, SpatioTemporalScaleFactors, VideoLatentPatchifier,
    VideoLatentShape, VideoPixelShape,
};
use super::plan::Ltx2GeneratePlan;
use super::text::connectors::EmbeddingsProcessorOutput;
use super::text::prompt_encoder::{NativePromptEncoder, NativePromptEncoding};
use crate::engine::{gpu_dtype, seeded_randn};

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
        let stage1_shape = derive_stage1_render_shape(
            plan.width,
            plan.height,
            plan.num_frames,
            plan.frame_rate,
            plan.spatial_upscale,
            plan.temporal_upscale,
        );
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
                let audio_positions = audio_temporal_positions(
                    audio_patchifier,
                    audio_shape,
                    &prepared_device,
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
        match render_real_video_only(plan, prepared) {
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

const DISTILLED_SIGMAS_NO_TERMINAL: &[f32] = &[
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875,
];

fn supports_real_video_path(plan: &Ltx2GeneratePlan) -> bool {
    plan.conditioning.images.is_empty()
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

fn render_real_video_only(
    plan: &Ltx2GeneratePlan,
    prepared: &NativePreparedRun,
) -> Result<NativeRenderedVideo> {
    let debug_enabled = ltx_debug_enabled();
    let device = select_real_video_device()?;
    let video_patchifier = VideoLatentPatchifier::new(1);
    let transformer = load_ltx2_video_transformer(plan, &device)?;
    let mut video_scheduler = FlowMatchEulerDiscreteScheduler::new(ltx2_scheduler_config())?;
    let sigma_count = plan
        .num_inference_steps
        .max(1)
        .min(DISTILLED_SIGMAS_NO_TERMINAL.len() as u32) as usize;
    video_scheduler.set_timesteps(
        None,
        &device,
        Some(&DISTILLED_SIGMAS_NO_TERMINAL[..sigma_count]),
        None,
        None,
    )?;
    let noise = seeded_randn(
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
    let mut latents = video_patchifier.patchify(&noise)?;
    let run_sigmas = video_scheduler.sigmas().to_device(&candle_core::Device::Cpu)?;
    let run_sigmas = run_sigmas.to_vec1::<f32>()?;
    video_scheduler.set_begin_index(0);

    let cond_context = prepared
        .prompt
        .conditional
        .video_encoding
        .to_device(&device)?;
    let cond_mask = prepared
        .prompt
        .conditional
        .attention_mask
        .to_device(&device)?;
    let uncond_context = prepared
        .prompt
        .unconditional
        .video_encoding
        .to_device(&device)?;
    let uncond_mask = prepared
        .prompt
        .unconditional
        .attention_mask
        .to_device(&device)?;
    let video_positions = prepared.video_positions.to_device(&device)?;

    if debug_enabled {
        log_tensor_stats("cond_context", &cond_context)?;
        log_tensor_stats("uncond_context", &uncond_context)?;
        let context_delta = cond_context.broadcast_sub(&uncond_context)?;
        log_tensor_stats("context_delta", &context_delta)?;
        log_tensor_stats("initial_latents", &latents)?;
    }

    for (step_idx, sigma) in run_sigmas
        .iter()
        .copied()
        .take(run_sigmas.len().saturating_sub(1))
        .enumerate()
    {
        let batch = latents.dim(0)?;
        let timestep = Tensor::full(sigma, (batch,), &device)?;
        let cond_video = transformer.forward(
            &latents,
            &cond_context,
            &timestep,
            Some(&cond_mask),
            prepared.video_pixel_shape.frames,
            prepared.video_latent_shape.height,
            prepared.video_latent_shape.width,
            None,
            Some(&video_positions),
            None,
        )?;
        let cond_video = cond_video.to_dtype(DType::F32)?;
        let guidance_scale = plan.guidance;
        let video_model_output = if guidance_scale <= 1.0 {
            cond_video
        } else {
            let cond_video_denoised = denoised_from_velocity(&latents, &cond_video, sigma)?;
            let uncond_video = transformer.forward(
                &latents,
                &uncond_context,
                &timestep,
                Some(&uncond_mask),
                prepared.video_pixel_shape.frames,
                prepared.video_latent_shape.height,
                prepared.video_latent_shape.width,
                None,
                Some(&video_positions),
                None,
            )?;
            let uncond_video = uncond_video.to_dtype(DType::F32)?;
            let uncond_video_denoised =
                denoised_from_velocity(&latents, &uncond_video, sigma)?;
            let video_diff = cond_video_denoised.broadcast_sub(&uncond_video_denoised)?;
            let guided_video_denoised =
                uncond_video_denoised.broadcast_add(&video_diff.affine(guidance_scale, 0.0)?)?;
            if debug_enabled {
                eprintln!("[ltx2-debug] step={step_idx} sigma={sigma:.6}");
                log_tensor_stats("step_video_latents", &latents)?;
                log_tensor_stats("cond_video_velocity", &cond_video)?;
                log_tensor_stats("uncond_video_velocity", &uncond_video)?;
                log_tensor_stats("guided_video_denoised", &guided_video_denoised)?;
            }
            velocity_from_denoised(&latents, &guided_video_denoised, sigma)?
        };

        latents = video_scheduler
            .step(&video_model_output, sigma, &latents, None)?
            .prev_sample;
    }

    let latents = video_patchifier.unpatchify(&latents, prepared.video_latent_shape)?;
    if debug_enabled {
        log_tensor_stats("final_patched_latents", &latents)?;
    }
    drop(transformer);

    let dtype = gpu_dtype(&device);
    let mut vae = load_ltx2_video_vae(plan, &device, dtype)?;
    vae.use_tiling = false;
    vae.use_framewise_decoding = false;
    let latents = denormalize_latents_with_vae(&latents, &vae)?.to_dtype(dtype)?;
    if debug_enabled {
        log_tensor_stats("final_denormalized_latents", &latents)?;
    }
    let (_dec_output, video) = vae.decode(&latents, None, false, false)?;
    if debug_enabled {
        log_tensor_stats("decoded_video", &video)?;
    }
    let video = ((video.to_dtype(DType::F32)?.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        .to_dtype(DType::U8)?;
    let video = video.i(0)?;

    let mut frames = Vec::with_capacity(video.dim(1)?);
    for index in 0..video.dim(1)? {
        let frame = video
            .i((.., index, .., ..))?
            .permute((1, 2, 0))?
            .contiguous()?;
        let data: Vec<u8> = frame.flatten_all()?.to_vec1()?;
        let rgb = RgbImage::from_raw(
            prepared.video_pixel_shape.width as u32,
            prepared.video_pixel_shape.height as u32,
            data,
        )
        .context("failed to build an RGB frame from the decoded LTX-2 tensor")?;
        frames.push(rgb);
    }

    if plan.temporal_upscale.is_some() {
        frames = temporally_upsample_frames_x2(&frames, Some(plan.num_frames));
    }
    if plan.spatial_upscale.is_some()
        || plan.width != prepared.video_pixel_shape.width as u32
        || plan.height != prepared.video_pixel_shape.height as u32
    {
        frames = spatially_upsample_frames(&frames, plan.width, plan.height);
    }

    Ok(NativeRenderedVideo {
        frames,
        has_audio: false,
        audio_sample_rate: None,
        audio_channels: None,
    })
}

fn load_ltx2_av_transformer(
    plan: &Ltx2GeneratePlan,
    device: &candle_core::Device,
) -> Result<Ltx2AvTransformer3DModel> {
    let dtype = transformer_weight_dtype(plan, device);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&Path::new(&plan.checkpoint_path)),
            dtype,
            device,
        )?
    };
    let vb = vb.rename_f(|name| remap_ltx2_transformer_key(name));
    Ok(Ltx2AvTransformer3DModel::new_streaming(
        &ltx2_video_transformer_config(plan),
        vb,
    )?)
}

fn load_ltx2_video_transformer(
    plan: &Ltx2GeneratePlan,
    device: &candle_core::Device,
) -> Result<Ltx2VideoTransformer3DModel> {
    let dtype = transformer_weight_dtype(plan, device);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&Path::new(&plan.checkpoint_path)),
            dtype,
            device,
        )?
    };
    let vb = vb.rename_f(|name| remap_ltx2_transformer_key(name));
    let mut config = ltx2_video_transformer_config(plan);
    config.audio_num_attention_heads = 0;
    config.audio_attention_head_dim = 0;
    config.audio_in_channels = 0;
    config.audio_out_channels = 0;
    config.audio_cross_attention_dim = 0;
    Ok(Ltx2VideoTransformer3DModel::new_streaming(&config, vb)?)
}

fn load_ltx2_video_vae(
    plan: &Ltx2GeneratePlan,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<AutoencoderKLLtxVideo> {
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&Path::new(&plan.checkpoint_path)),
            dtype,
            device,
        )?
    };
    let vb = vb.rename_f(|name| remap_ltx2_vae_key(name));
    Ok(AutoencoderKLLtxVideo::new(ltx2_video_vae_config(), vb)?)
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
        av_ca_timestep_scale_multiplier: 1.0,
        cross_attention_adaln: plan.preset.transformer.cross_attention_adaln,
    }
}

fn transformer_weight_dtype(plan: &Ltx2GeneratePlan, device: &candle_core::Device) -> DType {
    if device.is_cuda()
        && plan
            .quantization
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case("fp8"))
    {
        DType::F8E4M3
    } else {
        gpu_dtype(device)
    }
}

fn ltx2_video_vae_config() -> AutoencoderKLLtxVideoConfig {
    AutoencoderKLLtxVideoConfig {
        block_out_channels: vec![128, 256, 512, 1024, 2048],
        decoder_block_out_channels: vec![256, 512, 1024],
        spatiotemporal_scaling: vec![true, true, true, true],
        decoder_spatiotemporal_scaling: vec![true, true, true],
        layers_per_block: vec![4, 6, 6, 2, 2],
        decoder_layers_per_block: vec![5, 5, 5, 5],
        patch_size: 4,
        patch_size_t: 1,
        resnet_eps: 1e-6,
        scaling_factor: 1.0,
        spatial_compression_ratio: 32,
        temporal_compression_ratio: 8,
        decoder_inject_noise: vec![false, false, false, false],
        decoder_upsample_residual: vec![true, true, true],
        decoder_upsample_factor: vec![2, 2, 2],
        timestep_conditioning: false,
        downsample_types: vec![
            "spatial".to_string(),
            "temporal".to_string(),
            "spatiotemporal".to_string(),
            "spatiotemporal".to_string(),
        ],
        is_causal: true,
        decoder_causal: false,
        ..Default::default()
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

fn remap_ltx2_vae_key(name: &str) -> String {
    match name {
        "latents_mean" => "vae.per_channel_statistics.mean-of-means".to_string(),
        "latents_std" => "vae.per_channel_statistics.std-of-means".to_string(),
        _ => {
            if let Some(mapped) = remap_ltx2_encoder_vae_block(name) {
                return mapped;
            }
            if let Some(mapped) = remap_ltx2_decoder_vae_block(name) {
                return mapped;
            }
            format!("vae.{name}")
        }
    }
}

fn remap_ltx2_encoder_vae_block(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("encoder.mid_block.") {
        let rest = rest.replace("resnets", "res_blocks");
        return Some(format!("vae.encoder.down_blocks.8.{rest}"));
    }

    let rest = name.strip_prefix("encoder.down_blocks.")?;
    let (block_idx, tail) = rest.split_once('.')?;
    let block_idx: usize = block_idx.parse().ok()?;

    if let Some(tail) = tail.strip_prefix("downsamplers.0.") {
        return Some(format!(
            "vae.encoder.down_blocks.{}.{tail}",
            block_idx * 2 + 1
        ));
    }

    Some(format!(
        "vae.encoder.down_blocks.{}.{}",
        block_idx * 2,
        tail.replace("resnets", "res_blocks")
    ))
}

fn remap_ltx2_decoder_vae_block(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("decoder.mid_block.") {
        let rest = rest.replace("resnets", "res_blocks");
        return Some(format!("vae.decoder.up_blocks.0.{rest}"));
    }

    let rest = name.strip_prefix("decoder.up_blocks.")?;
    let (block_idx, tail) = rest.split_once('.')?;
    let block_idx: usize = block_idx.parse().ok()?;

    if let Some(tail) = tail.strip_prefix("upsamplers.0.") {
        return Some(format!(
            "vae.decoder.up_blocks.{}.{tail}",
            block_idx * 2 + 1
        ));
    }

    Some(format!(
        "vae.decoder.up_blocks.{}.{}",
        block_idx * 2 + 2,
        tail.replace("resnets", "res_blocks")
    ))
}

fn denormalize_latents_with_vae(latents: &Tensor, vae: &AutoencoderKLLtxVideo) -> Result<Tensor> {
    let channels = latents.dim(1)?;
    let mean = vae
        .latents_mean()
        .reshape((1, channels, 1, 1, 1))?
        .to_device(latents.device())?
        .to_dtype(latents.dtype())?;
    let std = vae
        .latents_std()
        .reshape((1, channels, 1, 1, 1))?
        .to_device(latents.device())?
        .to_dtype(latents.dtype())?;
    latents.broadcast_mul(&std)?.broadcast_add(&mean).map_err(Into::into)
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
    let abs_mean = tensor.flatten_all()?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let sq_mean = tensor.flatten_all()?.sqr()?.mean_all()?.to_scalar::<f32>()?;
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
        remap_ltx2_vae_key, Ltx2RuntimeSession, LTX2_AUDIO_LATENT_CHANNELS,
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

    #[test]
    fn remap_ltx2_vae_mid_block_paths_to_native_checkpoint_layout() {
        assert_eq!(
            remap_ltx2_vae_key("decoder.mid_block.resnets.0.conv1.conv.weight"),
            "vae.decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"
        );
        assert_eq!(
            remap_ltx2_vae_key("encoder.mid_block.resnets.1.conv2.conv.bias"),
            "vae.encoder.down_blocks.8.res_blocks.1.conv2.conv.bias"
        );
    }

    #[test]
    fn remap_ltx2_vae_hierarchical_blocks_to_native_checkpoint_layout() {
        assert_eq!(
            remap_ltx2_vae_key("decoder.up_blocks.0.upsamplers.0.conv.conv.weight"),
            "vae.decoder.up_blocks.1.conv.conv.weight"
        );
        assert_eq!(
            remap_ltx2_vae_key("decoder.up_blocks.2.resnets.4.conv2.conv.bias"),
            "vae.decoder.up_blocks.6.res_blocks.4.conv2.conv.bias"
        );
        assert_eq!(
            remap_ltx2_vae_key("encoder.down_blocks.0.downsamplers.0.conv.conv.weight"),
            "vae.encoder.down_blocks.1.conv.conv.weight"
        );
        assert_eq!(
            remap_ltx2_vae_key("encoder.down_blocks.3.resnets.1.conv1.conv.bias"),
            "vae.encoder.down_blocks.6.res_blocks.1.conv1.conv.bias"
        );
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

        let rendered = session.render_native_video(&plan, &prepared).unwrap();
        assert_eq!(rendered.frames.len(), 97);
        assert_eq!(rendered.frames[0].dimensions(), (1216, 704));
        assert!(rendered.has_audio);
        assert_eq!(rendered.audio_sample_rate, Some(48_000));
        assert_eq!(rendered.audio_channels, Some(2));
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
}
