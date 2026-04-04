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
    sampling::{FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig, TimeShiftType},
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
/// Patch sizes (both 1 for LTX Video v0.9.5).
const PATCH_SIZE: usize = 1;
const PATCH_SIZE_T: usize = 1;

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

        // Video parameters with defaults
        let num_frames = req.frames.unwrap_or(25);
        let fps = req.fps.unwrap_or(24);
        let steps = req.steps;
        let guidance = req.guidance;

        // Validate frame count: must be 8n+1
        if (num_frames.wrapping_sub(1)) % 8 != 0 {
            bail!(
                "frame count must be 8n+1 (9, 17, 25, 33, ...), got {}",
                num_frames
            );
        }

        let seed = req.seed.unwrap_or_else(rand_seed);
        let width = req.width;
        let height = req.height;

        // Latent dimensions
        let latent_h = height as usize / VAE_SPATIAL_COMPRESSION;
        let latent_w = width as usize / VAE_SPATIAL_COMPRESSION;
        let latent_f = (num_frames as usize - 1) / VAE_TEMPORAL_COMPRESSION + 1;
        let video_seq_len = latent_f * latent_h * latent_w;

        // Always use sequential mode for video (high VRAM usage)
        self.generate_sequential(
            req, seed, num_frames, fps, steps, guidance, width, height, latent_h, latent_w,
            latent_f, video_seq_len, start,
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
        seed: u64,
        num_frames: u32,
        fps: u32,
        steps: u32,
        _guidance: f64,
        width: u32,
        height: u32,
        latent_h: usize,
        latent_w: usize,
        latent_f: usize,
        video_seq_len: usize,
        start: Instant,
    ) -> Result<GenerateResponse> {
        let progress = &self.base.progress;
        let paths = &self.base.paths;

        // Select device
        let device = crate::device::create_device(progress)?;
        let dtype = gpu_dtype(&device);

        progress.info(&format!(
            "LTX Video: {}×{} × {} frames, {} steps, seed {}",
            width, height, num_frames, steps, seed
        ));

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

        // Drop T5 to free VRAM
        drop(t5);
        device.synchronize()?;
        progress.info("T5 encoder dropped, VRAM freed");

        // ---------------------------------------------------------------
        // Step 2: Load transformer and denoise
        // ---------------------------------------------------------------
        progress.stage_start("Loading LTX Video transformer");
        let xformer_start = Instant::now();

        let transformer_path = &paths.transformer;
        let is_gguf = transformer_path
            .extension()
            .is_some_and(|e| e == "gguf");

        if is_gguf {
            bail!("GGUF quantized LTX Video transformer is not yet supported — use :bf16 variant");
        }

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[transformer_path.clone()],
                dtype,
                &device,
            )?
        };
        // Official LTX Video weights use "model.diffusion_model." prefix and
        // different key names than our diffusers-style candle model.
        // rename_f maps queried names → stored names in the safetensors file.
        let vb = vb.rename_f(|queried: &str| {
            // Our code queries: "proj_in.weight"
            // File stores:     "model.diffusion_model.patchify_proj.weight"
            let remapped = queried
                .replace("proj_in", "patchify_proj")
                .replace("time_embed", "adaln_single")
                .replace("norm_q", "q_norm")
                .replace("norm_k", "k_norm");
            format!("model.diffusion_model.{remapped}")
        });
        // Debug: verify a weight tensor loaded correctly
        {
            let test = vb.get_unchecked_dtype(
                "transformer_blocks.0.scale_shift_table",
                dtype,
            )?;
            let test_f32 = test.to_dtype(DType::F32)?;
            let mean = test_f32.mean_all()?.to_scalar::<f32>()?;
            let std_val = test_f32.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
            progress.info(&format!(
                "Debug: block0.scale_shift_table shape={:?}, mean={:.6}, rms={:.6}",
                test.dims(), mean, std_val
            ));
        }

        let config = LtxVideoTransformer3DModelConfig::default();
        let transformer = LtxVideoTransformer3DModel::new(&config, vb)?;
        progress.stage_done("Loading LTX Video transformer", xformer_start.elapsed());

        // Generate initial noise
        let noise = seeded_randn(
            seed,
            &[1, LATENT_CHANNELS, latent_f, latent_h, latent_w],
            &device,
            DType::F32,
        )?;

        // Pack latents: [B,C,F,H,W] → [B,S,D]
        let latents = pack_latents(&noise, PATCH_SIZE, PATCH_SIZE_T)?;

        // Build scheduler matching the candle-video pipeline
        let scheduler_config = FlowMatchEulerDiscreteSchedulerConfig {
            num_train_timesteps: 1000,
            shift: 1.0,
            use_dynamic_shifting: false,
            base_shift: Some(0.95),
            max_shift: Some(2.05),
            base_image_seq_len: Some(1024),
            max_image_seq_len: Some(4096),
            shift_terminal: Some(0.1),
            time_shift_type: TimeShiftType::Exponential,
            ..Default::default()
        };
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_config)?;

        // Calculate mu and set timesteps with the sigma schedule
        let mu = calculate_shift(video_seq_len, 1024, 4096, 0.95, 2.05);
        let sigmas: Vec<f32> = linspace(1.0, 1.0 / steps as f32, steps as usize);
        scheduler.set_timesteps(
            Some(steps as usize),
            &device,
            Some(&sigmas),
            Some(mu),
            None,
        )?;

        let timesteps: Vec<f32> = scheduler.timesteps().to_vec1::<f32>()?;
        let total_steps = timesteps.len();

        // ---------------------------------------------------------------
        // Step 3: Denoising loop
        // ---------------------------------------------------------------
        progress.stage_start("Denoising");
        let denoise_start = Instant::now();

        let mut latents = latents;

        for (step, &t) in timesteps.iter().enumerate() {
            let step_start = Instant::now();

            let b = latents.dim(0)?;
            let timestep_t = Tensor::full(t, (b,), &device)?.to_dtype(dtype)?;
            let latents_input = latents.to_dtype(dtype)?;

            // Transformer forward pass
            let noise_pred = transformer.forward(
                &latents_input,
                &prompt_embeds,
                &timestep_t,
                Some(&attention_mask),
                latent_f,
                latent_h,
                latent_w,
                None,
            )?;

            // Debug first step
            if step == 0 {
                let v_f32 = noise_pred.to_dtype(DType::F32)?;
                let v_mean = v_f32.mean_all()?.to_scalar::<f32>()?;
                let v_rms = v_f32.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
                progress.info(&format!(
                    "Step 0: t={:.1}, v_mean={:.4}, v_rms={:.4}", t, v_mean, v_rms
                ));
            }

            // Euler step via scheduler
            let step_out = scheduler.step(
                &noise_pred.to_dtype(DType::F32)?,
                t,
                &latents,
                None,
            )?;
            latents = step_out.prev_sample;

            progress.emit(ProgressEvent::DenoiseStep {
                step: step + 1,
                total: total_steps,
                elapsed: step_start.elapsed(),
            });
        }

        // Debug: log sigma schedule and latent stats
        {
            let first_5: Vec<f32> = sigmas.iter().take(5).copied().collect();
            let last_3: Vec<f32> = sigmas.iter().rev().take(3).rev().copied().collect();
            progress.info(&format!("Sigmas: {:?}...{:?} (mu={:.3})", first_5, last_3, mu));
            let l_f32 = latents.to_dtype(DType::F32)?;
            let mean = l_f32.mean_all()?.to_scalar::<f32>()?;
            let var = l_f32.var_keepdim(candle_core::D::Minus1)?.mean_all()?.to_scalar::<f32>()?;
            progress.info(&format!("Denoised latents: mean={:.4}, var={:.4}", mean, var));
        }

        progress.stage_done("Denoising", denoise_start.elapsed());

        // Drop transformer to free VRAM for VAE
        drop(transformer);
        device.synchronize()?;
        progress.info("Transformer dropped, VRAM freed for VAE decode");

        // ---------------------------------------------------------------
        // Step 4: Unpack and denormalize latents
        // ---------------------------------------------------------------
        let mut latents = unpack_latents(&latents, latent_f, latent_h, latent_w, PATCH_SIZE, PATCH_SIZE_T)?;
        // latents is now [B, C, F, H, W] in F32

        // ---------------------------------------------------------------
        // Step 5: Load VAE and decode
        // ---------------------------------------------------------------
        progress.stage_start("Loading VAE decoder");
        let vae_start = Instant::now();

        let vae_path = &paths.vae;
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[vae_path.clone()], dtype, &device)?
        };
        // VAE config matching the diffusers-format config.json from Lightricks/LTX-Video.
        // This is the v0.9 VAE architecture (simpler, shared encoder/decoder params).
        let vae_config = AutoencoderKLLtxVideoConfig {
            block_out_channels: vec![128, 256, 512, 512],
            decoder_block_out_channels: vec![128, 256, 512, 512],
            spatiotemporal_scaling: vec![true, true, true, false],
            decoder_spatiotemporal_scaling: vec![true, true, true, false],
            layers_per_block: vec![4, 3, 3, 3, 4],
            decoder_layers_per_block: vec![4, 3, 3, 3, 4],
            decoder_inject_noise: vec![false, false, false, false, false],
            decoder_upsample_residual: vec![false, false, false, false, false],
            decoder_upsample_factor: vec![1, 1, 1, 1, 1],
            timestep_conditioning: false,
            ..Default::default()
        };
        let scaling_factor = vae_config.scaling_factor as f32;
        let vae = AutoencoderKLLtxVideo::new(vae_config, vae_vb)?;
        progress.stage_done("Loading VAE decoder", vae_start.elapsed());

        progress.stage_start("Decoding video frames");
        let decode_start = Instant::now();

        // Denormalize latents using VAE statistics
        let latents_mean = vae.latents_mean();
        let latents_std = vae.latents_std();
        latents = denormalize_latents(&latents, latents_mean, latents_std, scaling_factor)?;

        // Debug: latent stats after denormalization
        {
            let l_f32 = latents.to_dtype(DType::F32)?;
            let mean = l_f32.mean_all()?.to_scalar::<f32>()?;
            let var = l_f32.var_keepdim(candle_core::D::Minus1)?.mean_all()?.to_scalar::<f32>()?;
            progress.info(&format!("Denormalized latents: mean={:.4}, var={:.4}", mean, var));
        }

        latents = latents.to_dtype(dtype)?;
        let (_dec_output, video) = vae.decode(&latents, None, false, false)?;
        // video: [B, 3, F, H, W] in model dtype

        progress.stage_done("Decoding video frames", decode_start.elapsed());

        // Drop VAE
        drop(vae);
        device.synchronize()?;

        // ---------------------------------------------------------------
        // Step 6: Post-process and encode to GIF
        // ---------------------------------------------------------------
        progress.stage_start("Encoding GIF");
        let gif_start = Instant::now();

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

        let gif_bytes = video_enc::encode_gif(&frames, fps)?;
        let thumbnail_bytes = video_enc::first_frame_png(&frames)?;

        progress.stage_done("Encoding GIF", gif_start.elapsed());

        let generation_time_ms = start.elapsed().as_millis() as u64;
        progress.info(&format!(
            "Done: {} frames, {:.1}s total",
            num_output_frames,
            generation_time_ms as f64 / 1000.0
        ));

        Ok(GenerateResponse {
            images: vec![],
            video: Some(VideoData {
                data: gif_bytes,
                format: OutputFormat::Gif,
                width,
                height,
                frames: num_output_frames as u32,
                fps,
                thumbnail: thumbnail_bytes,
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

/// Denormalize latents using VAE statistics: x = x * std / scaling_factor + mean.
fn denormalize_latents(
    latents: &Tensor,
    mean: &Tensor,
    std: &Tensor,
    scaling_factor: f32,
) -> Result<Tensor> {
    let c = latents.dim(1)?;
    let mean = mean
        .reshape((1, c, 1, 1, 1))?
        .to_device(latents.device())?
        .to_dtype(latents.dtype())?;
    let std = std
        .reshape((1, c, 1, 1, 1))?
        .to_device(latents.device())?
        .to_dtype(latents.dtype())?;

    let x = latents.broadcast_mul(&std)?;
    let x = x
        .affine((1.0 / scaling_factor) as f64, 0.0)?
        .broadcast_add(&mean)?;
    Ok(x)
}

/// Calculate the dynamic shift mu for the scheduler (SD3-style).
fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f32,
    max_shift: f32,
) -> f32 {
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len) as f32;
    let b = base_shift - m * base_seq_len as f32;
    image_seq_len as f32 * m + b
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

fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps <= 1 {
        return vec![start];
    }
    let d = (steps - 1) as f32;
    (0..steps)
        .map(|i| start + (end - start) * i as f32 / d)
        .collect()
}
