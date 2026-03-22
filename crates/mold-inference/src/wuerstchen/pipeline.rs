use anyhow::{bail, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::wuerstchen::ddpm::{DDPMWScheduler, DDPMWSchedulerConfig};
use candle_transformers::models::wuerstchen::diffnext::WDiffNeXt;
use candle_transformers::models::wuerstchen::paella_vq::PaellaVQ;
use candle_transformers::models::wuerstchen::prior::WPrior;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::time::Instant;

use crate::device::{check_memory_budget, memory_status_string, preflight_memory_check};
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::image::encode_image;
use crate::progress::{ProgressCallback, ProgressEvent, ProgressReporter};

/// Wuerstchen v2 prior dimensions.
const PRIOR_C_IN: usize = 16;
const PRIOR_C: usize = 1536;
const PRIOR_C_COND: usize = 1280;
const PRIOR_C_R: usize = 64;
const PRIOR_DEPTH: usize = 32;
const PRIOR_NHEAD: usize = 24;

/// Wuerstchen v2 decoder (Stage B) dimensions.
const DECODER_C_IN: usize = 4;
const DECODER_C_OUT: usize = 4;
const DECODER_C_R: usize = 64;
const DECODER_C_COND: usize = 1280;
const DECODER_CLIP_EMBD: usize = 1280;
const DECODER_PATCH_SIZE: usize = 2;

/// Latent compression ratio for Stage C (prior).
/// Wuerstchen operates in a 42x compressed latent space.
const LATENT_DIM_SCALE: f64 = 42.67;

/// Latent compression ratio for Stage B (decoder → VQ-GAN input).
const STAGE_B_SCALE: usize = 4;

/// Loaded Wuerstchen model components, ready for inference.
struct LoadedWuerstchen {
    prior: WPrior,
    decoder: WDiffNeXt,
    vqgan: PaellaVQ,
    clip_g: stable_diffusion::clip::ClipTextTransformer,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    dtype: DType,
}

/// Wuerstchen v2 inference engine.
///
/// Three-stage cascade: CLIP-G encode -> Prior (Stage C) -> Decoder (Stage B) -> VQ-GAN (Stage A).
pub struct WuerstchenEngine {
    loaded: Option<LoadedWuerstchen>,
    model_name: String,
    paths: ModelPaths,
    progress: ProgressReporter,
    load_strategy: LoadStrategy,
}

impl WuerstchenEngine {
    pub fn new(model_name: String, paths: ModelPaths, load_strategy: LoadStrategy) -> Self {
        Self {
            loaded: None,
            model_name,
            paths,
            progress: ProgressReporter::default(),
            load_strategy,
        }
    }

    /// Validate and return required Wuerstchen paths.
    fn validate_paths(
        &self,
    ) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
        let decoder = self
            .paths
            .decoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Decoder (Stage B) path required for Wuerstchen"))?
            .clone();
        let clip_encoder = self
            .paths
            .clip_encoder_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-G encoder path required for Wuerstchen"))?
            .clone();
        let clip_tokenizer = self
            .paths
            .clip_tokenizer_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-G tokenizer path required for Wuerstchen"))?
            .clone();

        for (label, path) in [
            ("prior (Stage C)", &self.paths.transformer),
            ("decoder (Stage B)", &decoder),
            ("vqgan (Stage A)", &self.paths.vae),
            ("clip_encoder (CLIP-G)", &clip_encoder),
            ("clip_tokenizer (CLIP-G)", &clip_tokenizer),
        ] {
            if !path.exists() {
                bail!("{label} file not found: {}", path.display());
            }
        }

        Ok((decoder, clip_encoder, clip_tokenizer))
    }

    /// Load all Wuerstchen model components (Eager mode).
    pub fn load(&mut self) -> Result<()> {
        if self.loaded.is_some() {
            return Ok(());
        }

        if self.load_strategy == LoadStrategy::Sequential {
            return Ok(());
        }

        let (decoder_path, clip_encoder_path, clip_tokenizer_path) = self.validate_paths()?;

        tracing::info!(model = %self.model_name, "loading Wuerstchen model components...");

        let device = crate::device::create_device(&self.progress)?;
        let dtype = if device.is_cuda() || device.is_metal() {
            DType::F16
        } else {
            DType::F32
        };

        // Load Prior (Stage C)
        self.progress.stage_start("Loading Prior (Stage C)");
        let prior_start = Instant::now();
        let prior_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[&self.paths.transformer],
                dtype,
                &device,
            )?
        };
        let prior = WPrior::new(
            PRIOR_C_IN,
            PRIOR_C,
            PRIOR_C_COND,
            PRIOR_C_R,
            PRIOR_DEPTH,
            PRIOR_NHEAD,
            false,
            prior_vb,
        )?;
        self.progress
            .stage_done("Loading Prior (Stage C)", prior_start.elapsed());

        // Load Decoder (Stage B)
        self.progress.stage_start("Loading Decoder (Stage B)");
        let decoder_start = Instant::now();
        let decoder_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&decoder_path], dtype, &device)?
        };
        let decoder = WDiffNeXt::new(
            DECODER_C_IN,
            DECODER_C_OUT,
            DECODER_C_R,
            DECODER_C_COND,
            DECODER_CLIP_EMBD,
            DECODER_PATCH_SIZE,
            false,
            decoder_vb,
        )?;
        self.progress
            .stage_done("Loading Decoder (Stage B)", decoder_start.elapsed());

        // Load VQ-GAN (Stage A)
        self.progress.stage_start("Loading VQ-GAN (Stage A)");
        let vqgan_start = Instant::now();
        let vqgan_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&self.paths.vae], dtype, &device)?
        };
        let vqgan = PaellaVQ::new(vqgan_vb)?;
        self.progress
            .stage_done("Loading VQ-GAN (Stage A)", vqgan_start.elapsed());

        // Load CLIP-G encoder
        self.progress.stage_start("Loading CLIP-G encoder");
        let clip_start = Instant::now();
        let clip_config = stable_diffusion::clip::Config::wuerstchen_prior();
        let clip_g = stable_diffusion::build_clip_transformer(
            &clip_config,
            &clip_encoder_path,
            &device,
            DType::F32,
        )?;
        self.progress
            .stage_done("Loading CLIP-G encoder", clip_start.elapsed());

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(&clip_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP-G tokenizer: {e}"))?;

        self.loaded = Some(LoadedWuerstchen {
            prior,
            decoder,
            vqgan,
            clip_g,
            tokenizer,
            device,
            dtype,
        });

        tracing::info!(model = %self.model_name, "all Wuerstchen components loaded successfully");
        Ok(())
    }

    /// Tokenize a prompt for the CLIP-G encoder.
    fn tokenize(
        tokenizer: &tokenizers::Tokenizer,
        prompt: &str,
        max_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(max_len);
        while ids.len() < max_len {
            ids.push(49407); // CLIP EOS/PAD token
        }
        let ids = ids.into_iter().map(|i| i as i64).collect::<Vec<_>>();
        Ok(Tensor::new(ids, device)?.unsqueeze(0)?)
    }

    /// Run the Stage C (Prior) denoising loop.
    fn denoise_prior(
        &self,
        prior: &WPrior,
        text_embeddings: &Tensor,
        latents: &mut Tensor,
        steps: usize,
        guidance: f64,
        device: &Device,
    ) -> Result<()> {
        let use_cfg = guidance > 1.0;
        let scheduler = DDPMWScheduler::new(steps, DDPMWSchedulerConfig::default())?;
        let timesteps = scheduler.timesteps().to_vec();

        let label = format!("Stage C Prior ({} steps)", timesteps.len() - 1);
        self.progress.stage_start(&label);
        let start = Instant::now();

        for (step_idx, &t) in timesteps.iter().enumerate() {
            if step_idx + 1 >= timesteps.len() {
                break; // last timestep is 0.0, not used for denoising
            }
            let step_start = Instant::now();

            let r = Tensor::new(&[t], device)?.to_dtype(latents.dtype())?;

            let noise_pred = if use_cfg {
                let latent_input = Tensor::cat(&[&*latents, &*latents], 0)?;
                let r_input = Tensor::cat(&[&r, &r], 0)?;
                let uncond = Tensor::zeros_like(text_embeddings)?;
                let c_input = Tensor::cat(&[&uncond, text_embeddings], 0)?;
                let pred = prior.forward(&latent_input, &r_input, &c_input)?;
                let chunks = pred.chunk(2, 0)?;
                let pred_uncond = &chunks[0];
                let pred_cond = &chunks[1];
                (pred_uncond + ((pred_cond - pred_uncond)? * guidance)?)?
            } else {
                prior.forward(&*latents, &r, text_embeddings)?
            };

            *latents = scheduler.step(&noise_pred, t, &*latents)?;

            self.progress.emit(ProgressEvent::DenoiseStep {
                step: step_idx + 1,
                total: timesteps.len() - 1,
                elapsed: step_start.elapsed(),
            });
        }

        self.progress.stage_done(&label, start.elapsed());
        Ok(())
    }

    /// Run the Stage B (Decoder) denoising loop.
    fn denoise_decoder(
        &self,
        decoder: &WDiffNeXt,
        text_embeddings: &Tensor,
        latents: &mut Tensor,
        steps: usize,
        guidance: f64,
        device: &Device,
    ) -> Result<()> {
        let use_cfg = guidance > 1.0;
        let scheduler = DDPMWScheduler::new(steps, DDPMWSchedulerConfig::default())?;
        let timesteps = scheduler.timesteps().to_vec();

        // EfficientNet features: pass zeros (matching candle example approach)
        let effnet = Tensor::zeros(
            (1, 16, latents.dim(2)?, latents.dim(3)?),
            latents.dtype(),
            device,
        )?;

        let label = format!("Stage B Decoder ({} steps)", timesteps.len() - 1);
        self.progress.stage_start(&label);
        let start = Instant::now();

        for (step_idx, &t) in timesteps.iter().enumerate() {
            if step_idx + 1 >= timesteps.len() {
                break;
            }
            let step_start = Instant::now();

            let r = Tensor::new(&[t], device)?.to_dtype(latents.dtype())?;

            let noise_pred = if use_cfg {
                let latent_input = Tensor::cat(&[&*latents, &*latents], 0)?;
                let r_input = Tensor::cat(&[&r, &r], 0)?;
                let effnet_input = Tensor::cat(&[&effnet, &effnet], 0)?;
                let uncond = Tensor::zeros_like(text_embeddings)?;
                let c_input = Tensor::cat(&[&uncond, text_embeddings], 0)?;
                let pred =
                    decoder.forward(&latent_input, &r_input, &effnet_input, Some(&c_input))?;
                let chunks = pred.chunk(2, 0)?;
                let pred_uncond = &chunks[0];
                let pred_cond = &chunks[1];
                (pred_uncond + ((pred_cond - pred_uncond)? * guidance)?)?
            } else {
                decoder.forward(&*latents, &r, &effnet, Some(text_embeddings))?
            };

            *latents = scheduler.step(&noise_pred, t, &*latents)?;

            self.progress.emit(ProgressEvent::DenoiseStep {
                step: step_idx + 1,
                total: timesteps.len() - 1,
                elapsed: step_start.elapsed(),
            });
        }

        self.progress.stage_done(&label, start.elapsed());
        Ok(())
    }

    /// Generate an image using sequential loading strategy.
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let (decoder_path, clip_encoder_path, clip_tokenizer_path) = self.validate_paths()?;

        if let Some(warning) = check_memory_budget(&self.paths, LoadStrategy::Sequential) {
            self.progress.info(&warning);
        }

        let device = crate::device::create_device(&self.progress)?;
        let dtype = if device.is_cuda() || device.is_metal() {
            DType::F16
        } else {
            DType::F32
        };

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);
        let width = req.width as usize;
        let height = req.height as usize;
        let guidance = req.guidance;
        let prior_steps = req.steps as usize;
        let decoder_steps = 12;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            prior_steps,
            decoder_steps,
            guidance,
            "starting sequential Wuerstchen generation"
        );

        self.progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: CLIP-G encode ---
        if let Some(status) = memory_status_string() {
            self.progress.info(&status);
        }

        let tokenizer = tokenizers::Tokenizer::from_file(&clip_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP-G tokenizer: {e}"))?;

        self.progress.stage_start("Loading CLIP-G encoder");
        let clip_start = Instant::now();
        let clip_config = stable_diffusion::clip::Config::wuerstchen_prior();
        let clip_g = stable_diffusion::build_clip_transformer(
            &clip_config,
            &clip_encoder_path,
            &device,
            DType::F32,
        )?;
        self.progress
            .stage_done("Loading CLIP-G encoder", clip_start.elapsed());

        self.progress.stage_start("Encoding prompt (CLIP-G)");
        let encode_start = Instant::now();
        let tokens = Self::tokenize(
            &tokenizer,
            &req.prompt,
            clip_config.max_position_embeddings,
            &device,
        )?;
        let text_embeddings = clip_g.forward(&tokens)?.to_dtype(dtype)?;
        self.progress
            .stage_done("Encoding prompt (CLIP-G)", encode_start.elapsed());

        drop(clip_g);
        self.progress.info("Freed CLIP-G encoder");
        tracing::info!("CLIP-G encoder dropped (sequential mode)");

        // --- Phase 2: Prior (Stage C) ---
        let prior_size = std::fs::metadata(&self.paths.transformer)
            .map(|m| m.len())
            .unwrap_or(0);
        preflight_memory_check("Prior (Stage C)", prior_size)?;
        if let Some(status) = memory_status_string() {
            self.progress.info(&status);
        }

        self.progress.stage_start("Loading Prior (Stage C)");
        let prior_start = Instant::now();
        let prior_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[&self.paths.transformer],
                dtype,
                &device,
            )?
        };
        let prior = WPrior::new(
            PRIOR_C_IN,
            PRIOR_C,
            PRIOR_C_COND,
            PRIOR_C_R,
            PRIOR_DEPTH,
            PRIOR_NHEAD,
            false,
            prior_vb,
        )?;
        self.progress
            .stage_done("Loading Prior (Stage C)", prior_start.elapsed());

        // Stage C latent dimensions: 42x compression
        let latent_h = (height as f64 / LATENT_DIM_SCALE).ceil() as usize;
        let latent_w = (width as f64 / LATENT_DIM_SCALE).ceil() as usize;
        let mut prior_latents = crate::engine::seeded_randn(
            seed,
            &[1, PRIOR_C_IN, latent_h, latent_w],
            &device,
            dtype,
        )?;

        self.denoise_prior(
            &prior,
            &text_embeddings,
            &mut prior_latents,
            prior_steps,
            guidance,
            &device,
        )?;

        drop(prior);
        device.synchronize()?;
        self.progress.info("Freed Prior (Stage C)");

        // --- Phase 3: Decoder (Stage B) ---
        let decoder_size = std::fs::metadata(&decoder_path)
            .map(|m| m.len())
            .unwrap_or(0);
        preflight_memory_check("Decoder (Stage B)", decoder_size)?;
        if let Some(status) = memory_status_string() {
            self.progress.info(&status);
        }

        self.progress.stage_start("Loading Decoder (Stage B)");
        let dec_start = Instant::now();
        let decoder_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&decoder_path], dtype, &device)?
        };
        let decoder = WDiffNeXt::new(
            DECODER_C_IN,
            DECODER_C_OUT,
            DECODER_C_R,
            DECODER_C_COND,
            DECODER_CLIP_EMBD,
            DECODER_PATCH_SIZE,
            false,
            decoder_vb,
        )?;
        self.progress
            .stage_done("Loading Decoder (Stage B)", dec_start.elapsed());

        let stage_b_h = latent_h * STAGE_B_SCALE;
        let stage_b_w = latent_w * STAGE_B_SCALE;
        let mut decoder_latents =
            crate::engine::seeded_randn(seed + 1, &[1, 4, stage_b_h, stage_b_w], &device, dtype)?;

        self.denoise_decoder(
            &decoder,
            &text_embeddings,
            &mut decoder_latents,
            decoder_steps,
            guidance,
            &device,
        )?;

        drop(decoder);
        drop(text_embeddings);
        device.synchronize()?;
        self.progress.info("Freed Decoder (Stage B)");

        // --- Phase 4: VQ-GAN decode (Stage A) ---
        self.progress.stage_start("Loading VQ-GAN (Stage A)");
        let vqgan_start = Instant::now();
        let vqgan_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&self.paths.vae], dtype, &device)?
        };
        let vqgan = PaellaVQ::new(vqgan_vb)?;
        self.progress
            .stage_done("Loading VQ-GAN (Stage A)", vqgan_start.elapsed());

        self.progress.stage_start("VQ-GAN decode");
        let decode_start = Instant::now();
        let img = vqgan.decode(&decoder_latents)?;
        let img = img.clamp(0f32, 1f32)?;
        let img = (img * 255.)?.to_dtype(DType::U8)?;
        let img = img.squeeze(0)?;
        self.progress
            .stage_done("VQ-GAN decode", decode_start.elapsed());

        let image_bytes = encode_image(&img, req.output_format, req.width, req.height)?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            generation_time_ms,
            seed,
            "sequential Wuerstchen generation complete"
        );

        Ok(GenerateResponse {
            images: vec![ImageData {
                data: image_bytes,
                format: req.output_format,
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms,
            model: req.model.clone(),
            seed_used: seed,
        })
    }
}

impl InferenceEngine for WuerstchenEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if req.scheduler.is_some() {
            tracing::warn!("scheduler selection not supported for Wuerstchen, ignoring");
        }
        if req.source_image.is_some() {
            tracing::warn!("img2img not yet supported for Wuerstchen — generating from text only");
        }
        if req.mask_image.is_some() {
            tracing::warn!("inpainting not yet supported for Wuerstchen — ignoring mask");
        }

        if self.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        let loaded = self
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);
        let width = req.width as usize;
        let height = req.height as usize;
        let guidance = req.guidance;
        let prior_steps = req.steps as usize;
        let decoder_steps = 12;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            prior_steps,
            decoder_steps,
            guidance,
            "starting Wuerstchen generation"
        );

        // 1. Encode prompt with CLIP-G
        let clip_config = stable_diffusion::clip::Config::wuerstchen_prior();
        let max_len = clip_config.max_position_embeddings;

        self.progress.stage_start("Encoding prompt (CLIP-G)");
        let encode_start = Instant::now();
        let tokens = Self::tokenize(&loaded.tokenizer, &req.prompt, max_len, &loaded.device)?;
        let text_embeddings = loaded.clip_g.forward(&tokens)?.to_dtype(loaded.dtype)?;
        self.progress
            .stage_done("Encoding prompt (CLIP-G)", encode_start.elapsed());

        // 2. Stage C (Prior): denoise in highly compressed latent space
        let latent_h = (height as f64 / LATENT_DIM_SCALE).ceil() as usize;
        let latent_w = (width as f64 / LATENT_DIM_SCALE).ceil() as usize;
        let mut prior_latents = crate::engine::seeded_randn(
            seed,
            &[1, PRIOR_C_IN, latent_h, latent_w],
            &loaded.device,
            loaded.dtype,
        )?;

        self.denoise_prior(
            &loaded.prior,
            &text_embeddings,
            &mut prior_latents,
            prior_steps,
            guidance,
            &loaded.device,
        )?;

        // 3. Stage B (Decoder): decode prior latents to VQ-GAN latent space
        let stage_b_h = latent_h * STAGE_B_SCALE;
        let stage_b_w = latent_w * STAGE_B_SCALE;
        let mut decoder_latents = crate::engine::seeded_randn(
            seed + 1,
            &[1, 4, stage_b_h, stage_b_w],
            &loaded.device,
            loaded.dtype,
        )?;

        self.denoise_decoder(
            &loaded.decoder,
            &text_embeddings,
            &mut decoder_latents,
            decoder_steps,
            guidance,
            &loaded.device,
        )?;

        // 4. Stage A (VQ-GAN): decode to pixel space
        self.progress.stage_start("VQ-GAN decode");
        let decode_start = Instant::now();
        let img = loaded.vqgan.decode(&decoder_latents)?;
        let img = img.clamp(0f32, 1f32)?;
        let img = (img * 255.)?.to_dtype(DType::U8)?;
        let img = img.squeeze(0)?;
        self.progress
            .stage_done("VQ-GAN decode", decode_start.elapsed());

        // 5. Encode to image format
        let image_bytes = encode_image(&img, req.output_format, req.width, req.height)?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "Wuerstchen generation complete");

        Ok(GenerateResponse {
            images: vec![ImageData {
                data: image_bytes,
                format: req.output_format,
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms,
            model: req.model.clone(),
            seed_used: seed,
        })
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_loaded(&self) -> bool {
        self.load_strategy == LoadStrategy::Sequential || self.loaded.is_some()
    }

    fn load(&mut self) -> Result<()> {
        WuerstchenEngine::load(self)
    }

    fn unload(&mut self) {
        self.loaded = None;
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }
}
