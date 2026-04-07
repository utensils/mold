//! LTX Video inference engine — text-to-video generation.
//!
//! Architecture: T5-XXL text encoder → LTXVideoTransformer3DModel → 3D Causal VAE → GIF
//! Follows the same patterns as Flux2Engine (drop-and-reload, VRAM management, progress).

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::ltx_video::{
    sampling::{
        FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig, TimeShiftType,
    },
    transformer::{LtxVideoTransformer3DModel, LtxVideoTransformer3DModelConfig},
    vae::{AutoencoderKLLtxVideo, AutoencoderKLLtxVideoConfig},
};

use mold_core::{GenerateRequest, GenerateResponse, ModelPaths, OutputFormat, VideoData};

use crate::engine::{gpu_dtype, rand_seed, seeded_randn, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::progress::{ProgressCallback, ProgressEvent};
use crate::shared_pool::SharedPool;

use super::video_enc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Spatial compression ratio of the LTX Video VAE.
const VAE_SPATIAL_COMPRESSION: usize = 32;
/// Temporal compression ratio of the LTX Video VAE.
const VAE_TEMPORAL_COMPRESSION: usize = 8;
/// Latent channels in the VAE.
const LATENT_CHANNELS: usize = 128;
/// Patch sizes (both 1 for current LTX Video checkpoints).
const PATCH_SIZE: usize = 1;
const PATCH_SIZE_T: usize = 1;

const LTX_2B_098_DISTILLED_SIGMAS: &[f32] =
    &[1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250];
const LTX_13B_098_DISTILLED_SIGMAS: &[f32] =
    &[1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250];
const LTX_096_DEV_SKIP_BLOCKS: &[usize] = &[19];
const LTX_098_2B_DISTILLED_SKIP_BLOCKS: &[usize] = &[42];
const LTX_098_13B_DISTILLED_SKIP_BLOCKS: &[usize] = &[42];
const LTX_098_13B_DEV_SKIP_BLOCKS: &[usize] = &[28];

fn is_official_ltx_transformer_checkpoint(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.ends_with(".safetensors")
                && name.starts_with("ltx")
                && !name.starts_with("diffusion_pytorch_model")
        })
}

fn remap_official_ltx_transformer_key(key: &str) -> String {
    let key = key
        .replace("proj_in", "patchify_proj")
        .replace("time_embed", "adaln_single")
        .replace("norm_q", "q_norm")
        .replace("norm_k", "k_norm");
    format!("model.diffusion_model.{key}")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LtxPipelineMode {
    Base,
    MultiscaleFirstPassFallback,
}

#[derive(Clone, Debug)]
struct LtxModelPreset {
    transformer_config: LtxVideoTransformer3DModelConfig,
    vae_config: AutoencoderKLLtxVideoConfig,
    scheduler_config: FlowMatchEulerDiscreteSchedulerConfig,
    default_steps: u32,
    decode_timestep: f32,
    decode_noise_scale: f32,
    custom_sigmas: Option<&'static [f32]>,
    skip_block_list: &'static [usize],
    mode: LtxPipelineMode,
}

impl LtxModelPreset {
    fn for_model(model_name: &str) -> Result<Self> {
        if model_name.contains("ltx-video-0.9.6-distilled") {
            Ok(Self {
                transformer_config: transformer_2b_config(),
                vae_config: improved_vae_config(),
                scheduler_config: scheduler_config(true),
                default_steps: 8,
                decode_timestep: 0.05,
                decode_noise_scale: 0.025,
                custom_sigmas: None,
                skip_block_list: &[],
                mode: LtxPipelineMode::Base,
            })
        } else if model_name.contains("ltx-video-0.9.6") {
            Ok(Self {
                transformer_config: transformer_2b_config(),
                vae_config: improved_vae_config(),
                scheduler_config: scheduler_config(false),
                default_steps: 40,
                decode_timestep: 0.05,
                decode_noise_scale: 0.025,
                custom_sigmas: None,
                skip_block_list: LTX_096_DEV_SKIP_BLOCKS,
                mode: LtxPipelineMode::Base,
            })
        } else if model_name.contains("ltx-video-0.9.8-2b-distilled") {
            Ok(Self {
                transformer_config: transformer_2b_config(),
                vae_config: improved_vae_config(),
                scheduler_config: scheduler_config(false),
                default_steps: 7,
                decode_timestep: 0.05,
                decode_noise_scale: 0.025,
                custom_sigmas: Some(LTX_2B_098_DISTILLED_SIGMAS),
                skip_block_list: LTX_098_2B_DISTILLED_SKIP_BLOCKS,
                mode: LtxPipelineMode::MultiscaleFirstPassFallback,
            })
        } else if model_name.contains("ltx-video-0.9.8-13b-distilled") {
            Ok(Self {
                transformer_config: transformer_13b_config(),
                vae_config: improved_vae_config(),
                scheduler_config: scheduler_config(false),
                default_steps: 7,
                decode_timestep: 0.05,
                decode_noise_scale: 0.025,
                custom_sigmas: Some(LTX_13B_098_DISTILLED_SIGMAS),
                skip_block_list: LTX_098_13B_DISTILLED_SKIP_BLOCKS,
                mode: LtxPipelineMode::MultiscaleFirstPassFallback,
            })
        } else if model_name.contains("ltx-video-0.9.8-13b-dev") {
            Ok(Self {
                transformer_config: transformer_13b_config(),
                vae_config: improved_vae_config(),
                scheduler_config: scheduler_config(false),
                default_steps: 30,
                decode_timestep: 0.05,
                decode_noise_scale: 0.025,
                custom_sigmas: None,
                skip_block_list: LTX_098_13B_DEV_SKIP_BLOCKS,
                mode: LtxPipelineMode::MultiscaleFirstPassFallback,
            })
        } else {
            bail!("unsupported LTX model preset for {}", model_name);
        }
    }
}

fn transformer_2b_config() -> LtxVideoTransformer3DModelConfig {
    LtxVideoTransformer3DModelConfig {
        num_layers: 28,
        num_attention_heads: 32,
        attention_head_dim: 64,
        cross_attention_dim: 2048,
        caption_channels: 4096,
        ..Default::default()
    }
}

fn transformer_13b_config() -> LtxVideoTransformer3DModelConfig {
    LtxVideoTransformer3DModelConfig {
        num_layers: 48,
        num_attention_heads: 32,
        attention_head_dim: 128,
        cross_attention_dim: 4096,
        caption_channels: 4096,
        ..Default::default()
    }
}

fn improved_vae_config() -> AutoencoderKLLtxVideoConfig {
    AutoencoderKLLtxVideoConfig {
        block_out_channels: vec![128, 256, 512, 1024, 2048],
        decoder_block_out_channels: vec![256, 512, 1024],
        spatiotemporal_scaling: vec![true, true, true, true],
        decoder_spatiotemporal_scaling: vec![true, true, true],
        layers_per_block: vec![4, 6, 6, 2, 2],
        decoder_layers_per_block: vec![5, 5, 5, 5],
        decoder_inject_noise: vec![false, false, false, false],
        decoder_upsample_residual: vec![true, true, true],
        decoder_upsample_factor: vec![2, 2, 2],
        timestep_conditioning: true,
        ..Default::default()
    }
}

fn scheduler_config(stochastic_sampling: bool) -> FlowMatchEulerDiscreteSchedulerConfig {
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
        stochastic_sampling,
    }
}

// ---------------------------------------------------------------------------
// Loaded state
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct LoadedLtxVideo {
    transformer: Option<LtxVideoTransformer3DModel>,
    vae: Option<AutoencoderKLLtxVideo>,
    device: Device,
    dtype: DType,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct LtxVideoEngine {
    base: EngineBase<LoadedLtxVideo>,
    t5_variant: Option<String>,
    shared_pool: Option<Arc<Mutex<SharedPool>>>,
}

impl LtxVideoEngine {
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        t5_variant: Option<String>,
        load_strategy: LoadStrategy,
        shared_pool: Option<Arc<Mutex<SharedPool>>>,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy),
            t5_variant,
            shared_pool,
        }
    }
}

// ---------------------------------------------------------------------------
// InferenceEngine trait
// ---------------------------------------------------------------------------

impl crate::engine::InferenceEngine for LtxVideoEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let start = Instant::now();
        let preset = LtxModelPreset::for_model(&self.base.model_name)?;

        // Video parameters with defaults
        let num_frames = req.frames.unwrap_or(25);
        let fps = req.fps.unwrap_or(24);
        let steps = req.steps;
        let guidance = req.guidance;

        // Validate frame count: must be 8n+1
        if !(num_frames.wrapping_sub(1)).is_multiple_of(8) {
            bail!(
                "frame count must be 8n+1 (9, 17, 25, 33, ...), got {}",
                num_frames
            );
        }

        let seed = req.seed.unwrap_or_else(rand_seed);
        let width = req.width;
        let height = req.height;

        // Validate dimensions are multiples of 32 (VAE spatial compression)
        if !width.is_multiple_of(VAE_SPATIAL_COMPRESSION as u32)
            || !height.is_multiple_of(VAE_SPATIAL_COMPRESSION as u32)
        {
            bail!(
                "LTX Video requires width and height to be multiples of {}, got {}x{}",
                VAE_SPATIAL_COMPRESSION,
                width,
                height
            );
        }

        // Latent dimensions
        let latent_h = height as usize / VAE_SPATIAL_COMPRESSION;
        let latent_w = width as usize / VAE_SPATIAL_COMPRESSION;
        let latent_f = (num_frames as usize - 1) / VAE_TEMPORAL_COMPRESSION + 1;
        let video_seq_len = latent_f * latent_h * latent_w;

        // Always use sequential mode for video (high VRAM usage)
        self.generate_sequential(
            req,
            &preset,
            seed,
            num_frames,
            fps,
            steps,
            guidance,
            width,
            height,
            latent_h,
            latent_w,
            latent_f,
            video_seq_len,
            start,
        )
    }

    fn model_name(&self) -> &str {
        &self.base.model_name
    }

    fn is_loaded(&self) -> bool {
        self.base.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        // Video engine always uses sequential mode — components loaded per-generate
        Ok(())
    }

    fn unload(&mut self) {
        self.base.unload();
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.base.set_on_progress(callback);
    }

    fn clear_on_progress(&mut self) {
        self.base.clear_on_progress();
    }

    fn model_paths(&self) -> Option<&ModelPaths> {
        Some(&self.base.paths)
    }
}

// ---------------------------------------------------------------------------
// Sequential generation (load-use-drop each component)
// ---------------------------------------------------------------------------

impl LtxVideoEngine {
    #[allow(clippy::too_many_arguments)]
    fn generate_sequential(
        &mut self,
        req: &GenerateRequest,
        preset: &LtxModelPreset,
        seed: u64,
        num_frames: u32,
        fps: u32,
        steps: u32,
        guidance: f64,
        width: u32,
        height: u32,
        latent_h: usize,
        latent_w: usize,
        latent_f: usize,
        _video_seq_len: usize,
        start: Instant,
    ) -> Result<GenerateResponse> {
        let progress = &self.base.progress;
        let paths = &self.base.paths;
        let ltx_debug = std::env::var("MOLD_LTX_DEBUG").is_ok_and(|v| v == "1");

        if preset.mode == LtxPipelineMode::MultiscaleFirstPassFallback
            && paths.spatial_upscaler.is_none()
        {
            bail!("LTX 0.9.8 requires a spatial upscaler asset in the pulled model files");
        }

        // Select device
        let device = crate::device::create_device(progress)?;
        let dtype = gpu_dtype(&device);

        progress.info(&format!(
            "LTX Video: {}×{} × {} frames, {} steps, seed {}",
            width, height, num_frames, steps, seed
        ));
        if preset.mode == LtxPipelineMode::MultiscaleFirstPassFallback {
            progress.info(
                "Using the 0.9.8 first-pass schedule. The spatial upscaler asset is present, but mold does not yet run the second refinement pass.",
            );
        }

        // ---------------------------------------------------------------
        // Step 1: Encode prompt with T5-XXL
        // ---------------------------------------------------------------
        progress.stage_start("Loading T5-XXL encoder");
        let t5_start = Instant::now();

        let t5_encoder_path = paths
            .t5_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("T5 encoder path not configured"))?;
        let t5_tokenizer_path = paths
            .t5_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("T5 tokenizer path not configured"))?;

        let mut t5 = crate::encoders::t5::T5Encoder::load(
            t5_encoder_path,
            t5_tokenizer_path,
            &device,
            dtype,
            progress,
        )?;
        progress.stage_done("Loading T5-XXL encoder", t5_start.elapsed());

        progress.stage_start("Encoding prompt");
        let encode_start = Instant::now();
        let prompt_embeds = t5.encode(&req.prompt, &device, dtype)?;
        // prompt_embeds: [1, seq_len, 4096] (T5 encoder already adds batch dim)
        progress.stage_done("Encoding prompt", encode_start.elapsed());

        // Build attention mask (all ones — no padding for single prompt)
        let prompt_seq_len = prompt_embeds.dim(1)?;
        let attention_mask =
            Tensor::ones((1, prompt_seq_len), DType::F32, &device)?.to_dtype(dtype)?;

        // Encode empty prompt for CFG (classifier-free guidance)
        let do_cfg = guidance > 1.0;
        let (uncond_embeds, uncond_mask) = if do_cfg {
            progress.stage_start("Encoding negative prompt (CFG)");
            let ue = t5.encode("", &device, dtype)?;
            let ue_seq = ue.dim(1)?;
            let um = Tensor::ones((1, ue_seq), DType::F32, &device)?.to_dtype(dtype)?;
            progress.stage_done("Encoding negative prompt (CFG)", encode_start.elapsed());
            (Some(ue), Some(um))
        } else {
            (None, None)
        };

        // Drop T5 to free VRAM
        drop(t5);
        device.synchronize()?;
        progress.info("T5 encoder dropped, VRAM freed");

        // ---------------------------------------------------------------
        // Step 2: Load transformer and denoise
        // ---------------------------------------------------------------
        progress.stage_start("Loading LTX Video transformer");
        let xformer_start = Instant::now();

        // Load transformer — supports sharded safetensors (diffusers format) or single file
        let transformer_files: Vec<std::path::PathBuf> = if !paths.transformer_shards.is_empty() {
            paths.transformer_shards.clone()
        } else {
            vec![paths.transformer.clone()]
        };

        let is_gguf = transformer_files
            .first()
            .and_then(|p| p.extension())
            .is_some_and(|e| e == "gguf");

        if is_gguf {
            bail!("GGUF quantized LTX Video transformer is not yet supported — use :bf16 variant");
        }

        // SAFETY: mmap'd safetensors files are not modified while mapped.
        // The files are read-only model weights from the HuggingFace cache.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&transformer_files, dtype, &device)? };
        let vb = if transformer_files.len() == 1
            && is_official_ltx_transformer_checkpoint(&transformer_files[0])
        {
            vb.rename_f(remap_official_ltx_transformer_key)
        } else {
            vb
        };
        let mut transformer = LtxVideoTransformer3DModel::new(&preset.transformer_config, vb)?;
        if !preset.skip_block_list.is_empty() {
            transformer.set_skip_block_list(preset.skip_block_list.to_vec());
        }
        progress.stage_done("Loading LTX Video transformer", xformer_start.elapsed());

        // Generate initial noise (raw, std≈1 — no normalization needed for v0.9 transformer)
        let noise = seeded_randn(
            seed,
            &[1, LATENT_CHANNELS, latent_f, latent_h, latent_w],
            &device,
            DType::F32,
        )?;

        // Pack latents: [B,C,F,H,W] → [B,S,D]
        let latents = pack_latents(&noise, PATCH_SIZE, PATCH_SIZE_T)?;

        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(preset.scheduler_config.clone())?;
        scheduler.set_timesteps(
            if preset.custom_sigmas.is_some() && steps == preset.default_steps {
                None
            } else {
                Some(steps as usize)
            },
            &device,
            if preset.custom_sigmas.is_some() && steps == preset.default_steps {
                preset.custom_sigmas
            } else {
                None
            },
            None,
            None,
        )?;
        scheduler.set_begin_index(0);
        let sched_sigmas = scheduler
            .sigmas()
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        let total_steps = sched_sigmas.len() - 1;

        // Build video coordinates for 3D RoPE (critical for proper positional encoding)
        let video_coords = build_video_coords(1, latent_f, latent_h, latent_w, fps, &device)?;

        // ---------------------------------------------------------------
        // Step 3: Denoising loop
        // ---------------------------------------------------------------
        progress.stage_start("Denoising");
        let denoise_start = Instant::now();

        let mut latents = latents;

        for (step, sigma) in sched_sigmas.iter().copied().enumerate().take(total_steps) {
            let step_start = Instant::now();

            let b = latents.dim(0)?;
            // Pass raw sigma to model — it internally multiplies by 1000
            let timestep_t = Tensor::full(sigma, (b,), &device)?.to_dtype(dtype)?;
            let latents_input = latents.to_dtype(dtype)?;

            // Transformer forward pass (with optional CFG)
            let noise_pred = if do_cfg {
                // Unconditional pass
                let uncond_pred = transformer.forward(
                    &latents_input,
                    uncond_embeds.as_ref().unwrap(),
                    &timestep_t,
                    uncond_mask.as_ref().map(|m| m as &Tensor),
                    latent_f,
                    latent_h,
                    latent_w,
                    None,
                    Some(&video_coords),
                )?;
                // Conditional pass
                let cond_pred = transformer.forward(
                    &latents_input,
                    &prompt_embeds,
                    &timestep_t,
                    Some(&attention_mask),
                    latent_f,
                    latent_h,
                    latent_w,
                    None,
                    Some(&video_coords),
                )?;
                // CFG: uncond + guidance * (cond - uncond)
                let uncond_f32 = uncond_pred.to_dtype(DType::F32)?;
                let cond_f32 = cond_pred.to_dtype(DType::F32)?;
                let diff = (&cond_f32 - &uncond_f32)?;
                (&uncond_f32 + diff * guidance)?
            } else {
                transformer.forward(
                    &latents_input,
                    &prompt_embeds,
                    &timestep_t,
                    Some(&attention_mask),
                    latent_f,
                    latent_h,
                    latent_w,
                    None,
                    Some(&video_coords),
                )?
            };

            // Debug: track per-step stats (MOLD_LTX_DEBUG=1)
            if ltx_debug {
                let v_f32 = noise_pred.to_dtype(DType::F32)?;
                let v_rms = v_f32.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
                let l_rms = latents.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
                let li_rms = latents_input
                    .to_dtype(DType::F32)?
                    .sqr()?
                    .mean_all()?
                    .to_scalar::<f32>()?
                    .sqrt();
                if step < 3 || step == total_steps - 1 {
                    progress.info(&format!(
                        "Step {}: sigma={:.4}, input_rms={:.4}, output_rms={:.4}, lat_rms={:.4}",
                        step, sigma, li_rms, v_rms, l_rms
                    ));
                }
            }

            let noise_pred_f32 = noise_pred.to_dtype(DType::F32)?;
            latents = scheduler
                .step(&noise_pred_f32, sigma, &latents, None)?
                .prev_sample;

            progress.emit(ProgressEvent::DenoiseStep {
                step: step + 1,
                total: total_steps,
                elapsed: step_start.elapsed(),
            });
        }

        // Debug: log sigma schedule and latent stats (MOLD_LTX_DEBUG=1)
        if ltx_debug {
            let first_5: Vec<f32> = sched_sigmas.iter().take(5).copied().collect();
            let last_3: Vec<f32> = sched_sigmas.iter().rev().take(3).rev().copied().collect();
            progress.info(&format!("Sigmas: {:?}...{:?}", first_5, last_3));
            let l_f32 = latents.to_dtype(DType::F32)?;
            let mean = l_f32.mean_all()?.to_scalar::<f32>()?;
            let var = l_f32
                .var_keepdim(candle_core::D::Minus1)?
                .mean_all()?
                .to_scalar::<f32>()?;
            progress.info(&format!(
                "Denoised latents: mean={:.4}, var={:.4}",
                mean, var
            ));
        }

        progress.stage_done("Denoising", denoise_start.elapsed());

        // Drop transformer to free VRAM for VAE
        drop(transformer);
        device.synchronize()?;
        progress.info("Transformer dropped, VRAM freed for VAE decode");

        // ---------------------------------------------------------------
        // Step 4: Unpack and denormalize latents
        // ---------------------------------------------------------------
        let mut latents = unpack_latents(
            &latents,
            latent_f,
            latent_h,
            latent_w,
            PATCH_SIZE,
            PATCH_SIZE_T,
        )?;
        // latents is now [B, C, F, H, W] in F32

        // ---------------------------------------------------------------
        // Step 5: Load VAE and decode
        // ---------------------------------------------------------------
        progress.stage_start("Loading VAE decoder");
        let vae_start = Instant::now();

        // SAFETY: mmap'd safetensors file is not modified while mapped.
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&paths.vae), dtype, &device)?
        };
        let vae = AutoencoderKLLtxVideo::new(preset.vae_config.clone(), vae_vb)?;
        progress.stage_done("Loading VAE decoder", vae_start.elapsed());

        progress.stage_start("Decoding video frames");
        let decode_start = Instant::now();

        let decode_timestep = if vae.config().timestep_conditioning {
            if preset.decode_noise_scale > 0.0 {
                let noise =
                    seeded_randn(seed ^ 0xdec0de, latents.shape().dims(), &device, DType::F32)?;
                latents = (&latents * (1.0 - preset.decode_noise_scale as f64))?
                    .broadcast_add(&(noise * preset.decode_noise_scale as f64)?)?;
            }
            Some(Tensor::full(preset.decode_timestep, (1,), &device)?.to_dtype(dtype)?)
        } else {
            None
        };

        // Un-normalize latents immediately before VAE decode.
        let latents_mean = vae.latents_mean();
        let latents_std = vae.latents_std();
        {
            let c = latents.dim(1)?;
            let mean = latents_mean
                .reshape((1, c, 1, 1, 1))?
                .to_device(latents.device())?
                .to_dtype(latents.dtype())?;
            let std = latents_std
                .reshape((1, c, 1, 1, 1))?
                .to_device(latents.device())?
                .to_dtype(latents.dtype())?;
            latents = latents.broadcast_mul(&std)?.broadcast_add(&mean)?;
        }

        if ltx_debug {
            let l_f32 = latents.to_dtype(DType::F32)?;
            progress.info(&format!(
                "Latents pre-VAE (un-normalized): mean={:.4}, std={:.4}",
                l_f32.mean_all()?.to_scalar::<f32>()?,
                l_f32.flatten_all()?.var(0)?.to_scalar::<f32>()?.sqrt()
            ));
        }

        latents = latents.to_dtype(dtype)?;
        let (_dec_output, video) = vae.decode(&latents, decode_timestep.as_ref(), false, false)?;
        // video: [B, 3, F, H, W] in model dtype
        if ltx_debug {
            let v_f32 = video.to_dtype(DType::F32)?;
            progress.info(&format!(
                "VAE output: shape={:?}, mean={:.4}, min={:.4}, max={:.4}",
                v_f32.shape(),
                v_f32.mean_all()?.to_scalar::<f32>()?,
                v_f32.flatten_all()?.min(0)?.to_scalar::<f32>()?,
                v_f32.flatten_all()?.max(0)?.to_scalar::<f32>()?
            ));
        }

        progress.stage_done("Decoding video frames", decode_start.elapsed());

        // Drop VAE
        drop(vae);
        device.synchronize()?;

        // ---------------------------------------------------------------
        // Step 6: Post-process and encode video
        // ---------------------------------------------------------------
        // Default to APNG for video output (lossless, metadata-rich)
        let output_format = if req.output_format.is_video() {
            req.output_format
        } else {
            OutputFormat::Apng
        };
        let format_name = output_format.extension().to_uppercase();
        progress.stage_start(&format!("Encoding {format_name}"));
        let encode_start = Instant::now();

        // Convert to [0, 255] u8
        let video = video.to_dtype(DType::F32)?;
        let video = ((video.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let video = video.i(0)?; // Remove batch dim: [3, F, H, W]

        // Extract individual frames as RgbImage
        let num_output_frames = video.dim(1)?;
        let mut frames = Vec::with_capacity(num_output_frames);
        for f in 0..num_output_frames {
            let frame = video.i((.., f, .., ..))?.contiguous()?; // [3, H, W]
            let frame = frame.permute((1, 2, 0))?; // [H, W, 3]
            let frame_data: Vec<u8> = frame.flatten_all()?.to_vec1()?;
            let rgb = image::RgbImage::from_raw(width, height, frame_data)
                .ok_or_else(|| anyhow::anyhow!("failed to create frame image"))?;
            frames.push(rgb);
        }

        let video_bytes = match output_format {
            OutputFormat::Apng => {
                let metadata = video_enc::VideoMetadata {
                    prompt: req.prompt.clone(),
                    model: self.base.model_name.clone(),
                    seed,
                    steps,
                    guidance: req.guidance,
                    width,
                    height,
                    frames: num_output_frames as u32,
                    fps,
                };
                video_enc::encode_apng(&frames, fps, Some(&metadata))?
            }
            OutputFormat::Gif => video_enc::encode_gif(&frames, fps)?,
            #[cfg(feature = "webp")]
            OutputFormat::Webp => video_enc::encode_webp(&frames, fps)?,
            #[cfg(feature = "mp4")]
            OutputFormat::Mp4 => video_enc::encode_mp4(&frames, fps)?,
            #[cfg(not(feature = "webp"))]
            OutputFormat::Webp => {
                bail!("WebP output requires the 'webp' feature — rebuild with --features webp")
            }
            #[cfg(not(feature = "mp4"))]
            OutputFormat::Mp4 => {
                bail!("MP4 output requires the 'mp4' feature — rebuild with --features mp4")
            }
            _ => bail!("{format_name} is not a supported video output format"),
        };
        let thumbnail_bytes = video_enc::first_frame_png(&frames)?;
        // Generate a GIF preview only when the caller will use it (TUI gallery or --preview).
        // If the primary format is already GIF, reuse the data; otherwise encode on demand.
        let gif_preview = if req.gif_preview {
            if output_format == OutputFormat::Gif {
                video_bytes.clone()
            } else {
                video_enc::encode_gif(&frames, fps)?
            }
        } else {
            Vec::new()
        };

        progress.stage_done(&format!("Encoding {format_name}"), encode_start.elapsed());

        let generation_time_ms = start.elapsed().as_millis() as u64;
        progress.info(&format!(
            "Done: {} frames, {:.1}s total",
            num_output_frames,
            generation_time_ms as f64 / 1000.0
        ));

        Ok(GenerateResponse {
            images: vec![],
            video: Some(VideoData {
                data: video_bytes,
                format: output_format,
                width,
                height,
                frames: num_output_frames as u32,
                fps,
                thumbnail: thumbnail_bytes,
                gif_preview,
            }),
            generation_time_ms,
            model: self.base.model_name.clone(),
            seed_used: seed,
        })
    }
}

// ---------------------------------------------------------------------------
// Latent packing/unpacking (matches LTX Video pipeline)
// ---------------------------------------------------------------------------

/// Pack latents from [B,C,F,H,W] → [B,S,D] where S = F*H*W, D = C*pt*p*p.
fn pack_latents(latents: &Tensor, patch_size: usize, patch_size_t: usize) -> Result<Tensor> {
    let (b, c, f, h, w) = latents.dims5()?;
    if f % patch_size_t != 0 || h % patch_size != 0 || w % patch_size != 0 {
        bail!("latent dims not divisible by patch sizes");
    }
    let f2 = f / patch_size_t;
    let h2 = h / patch_size;
    let w2 = w / patch_size;

    // [B, C, F2, pt, H2, p, W2, p]
    let x = latents.reshape(&[b, c, f2, patch_size_t, h2, patch_size, w2, patch_size])?;
    // permute → [B, F2, H2, W2, C, pt, p, p]
    let x = x.permute([0, 2, 4, 6, 1, 3, 5, 7])?;
    // flatten last 4 → [B, F2, H2, W2, D]
    let x = x.flatten_from(4)?;
    let d = x.dim(4)?;
    let s = f2 * h2 * w2;
    Ok(x.reshape((b, s, d))?)
}

/// Unpack latents from [B,S,D] → [B,C,F,H,W].
fn unpack_latents(
    latents: &Tensor,
    num_frames: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    patch_size_t: usize,
) -> Result<Tensor> {
    let (b, _s, d) = latents.dims3()?;
    let denom = patch_size_t * patch_size * patch_size;
    if d % denom != 0 {
        bail!("D={d} not divisible by patch product {denom}");
    }
    let c = d / denom;

    let x = latents.reshape(&[
        b,
        num_frames,
        height,
        width,
        c,
        patch_size_t,
        patch_size,
        patch_size,
    ])?;
    // [B, C, F2, pt, H2, p, W2, p]
    let x = x.permute([0, 4, 1, 5, 2, 6, 3, 7])?.contiguous()?;
    Ok(x.reshape((
        b,
        c,
        num_frames * patch_size_t,
        height * patch_size,
        width * patch_size,
    ))?)
}

/// Build video coordinates for 3D RoPE: [B, seq, 3] with (frame, height, width).
fn build_video_coords(
    batch_size: usize,
    latent_f: usize,
    latent_h: usize,
    latent_w: usize,
    fps: u32,
    device: &Device,
) -> Result<Tensor> {
    let grid_f = Tensor::arange(0u32, latent_f as u32, device)?.to_dtype(DType::F32)?;
    let grid_h = Tensor::arange(0u32, latent_h as u32, device)?.to_dtype(DType::F32)?;
    let grid_w = Tensor::arange(0u32, latent_w as u32, device)?.to_dtype(DType::F32)?;

    let f = grid_f
        .reshape((latent_f, 1, 1))?
        .broadcast_as((latent_f, latent_h, latent_w))?;
    let h = grid_h
        .reshape((1, latent_h, 1))?
        .broadcast_as((latent_f, latent_h, latent_w))?;
    let w = grid_w
        .reshape((1, 1, latent_w))?
        .broadcast_as((latent_f, latent_h, latent_w))?;

    let grid = Tensor::stack(&[f, h, w], 0)?; // [3, F, H, W]
    let seq = latent_f * latent_h * latent_w;
    let grid = grid.flatten_from(1)?.transpose(0, 1)?.unsqueeze(0)?; // [1, seq, 3]

    // Apply compression ratios to coordinates
    let vf = grid.i((.., .., 0))?;
    let vh = grid.i((.., .., 1))?;
    let vw = grid.i((.., .., 2))?;

    // Temporal: (L * 8 + 1 - 8).clamp(0) / fps
    let ts_ratio = VAE_TEMPORAL_COMPRESSION as f64;
    let vf = vf
        .affine(ts_ratio, 1.0 - ts_ratio)?
        .clamp(0.0f32, 10000.0f32)?
        .affine(1.0 / fps as f64, 0.0)?;
    // Spatial: L * 32
    let sp_ratio = VAE_SPATIAL_COMPRESSION as f64;
    let vh = vh.affine(sp_ratio, 0.0)?;
    let vw = vw.affine(sp_ratio, 0.0)?;

    let coords = Tensor::stack(&[vf, vh, vw], candle_core::D::Minus1)?;
    if batch_size > 1 {
        Ok(coords.broadcast_as((batch_size, seq, 3))?)
    } else {
        Ok(coords)
    }
}

#[cfg(test)]
mod tests {
    use super::{is_official_ltx_transformer_checkpoint, remap_official_ltx_transformer_key};
    use std::path::Path;

    #[test]
    fn detects_official_ltx_single_file_checkpoints() {
        assert!(is_official_ltx_transformer_checkpoint(Path::new(
            "ltxv-2b-0.9.6-distilled-04-25.safetensors"
        )));
        assert!(is_official_ltx_transformer_checkpoint(Path::new(
            "ltxv-13b-0.9.8-dev.safetensors"
        )));
        assert!(!is_official_ltx_transformer_checkpoint(Path::new(
            "diffusion_pytorch_model-00001-of-00002.safetensors"
        )));
        assert!(!is_official_ltx_transformer_checkpoint(Path::new(
            "transformer.gguf"
        )));
    }

    #[test]
    fn remaps_official_transformer_keys_to_upstream_checkpoint_names() {
        assert_eq!(
            remap_official_ltx_transformer_key("proj_in.weight"),
            "model.diffusion_model.patchify_proj.weight"
        );
        assert_eq!(
            remap_official_ltx_transformer_key("time_embed.emb.timestep_embedder.linear_1.weight"),
            "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight"
        );
        assert_eq!(
            remap_official_ltx_transformer_key("transformer_blocks.0.attn1.norm_q.weight"),
            "model.diffusion_model.transformer_blocks.0.attn1.q_norm.weight"
        );
        assert_eq!(
            remap_official_ltx_transformer_key("caption_projection.linear_2.bias"),
            "model.diffusion_model.caption_projection.linear_2.bias"
        );
    }
}
