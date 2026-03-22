use anyhow::{bail, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_transformers::models::stable_diffusion;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::time::Instant;

use crate::device::{check_memory_budget, memory_status_string, preflight_memory_check};
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::image::encode_image;
use crate::progress::{ProgressCallback, ProgressEvent, ProgressReporter};

/// VAE scaling factor for SD1.5 models.
const VAE_SCALE: f64 = 0.18215;

/// Loaded SD1.5 model components, ready for inference.
struct LoadedSD15 {
    unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    vae: stable_diffusion::vae::AutoEncoderKL,
    clip: stable_diffusion::clip::ClipTextTransformer,
    tokenizer: tokenizers::Tokenizer,
    sd_config: stable_diffusion::StableDiffusionConfig,
    device: Device,
    dtype: DType,
}

/// SD1.5 inference engine backed by candle's stable_diffusion module.
///
/// Simplified variant of SDXL: single CLIP-L encoder, smaller UNet, 512x512 default.
pub struct SD15Engine {
    loaded: Option<LoadedSD15>,
    model_name: String,
    paths: ModelPaths,
    scheduler_name: String,
    progress: ProgressReporter,
    load_strategy: LoadStrategy,
}

impl SD15Engine {
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        scheduler_name: String,
        load_strategy: LoadStrategy,
    ) -> Self {
        Self {
            loaded: None,
            model_name,
            paths,
            scheduler_name,
            progress: ProgressReporter::default(),
            load_strategy,
        }
    }

    /// Validate and return required SD1.5 paths (CLIP-L encoder + tokenizer).
    fn validate_paths(&self) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
        let clip_encoder = self
            .paths
            .clip_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-L encoder path required for SD1.5 models"))?
            .clone();
        let clip_tokenizer = self
            .paths
            .clip_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-L tokenizer path required for SD1.5 models"))?
            .clone();

        for (label, path) in [
            ("transformer (UNet)", &self.paths.transformer),
            ("vae", &self.paths.vae),
            ("clip_encoder (CLIP-L)", &clip_encoder),
            ("clip_tokenizer (CLIP-L)", &clip_tokenizer),
        ] {
            if !path.exists() {
                bail!("{label} file not found: {}", path.display());
            }
        }

        Ok((clip_encoder, clip_tokenizer))
    }

    /// Create the SD1.5 config.
    fn sd_config(&self) -> stable_diffusion::StableDiffusionConfig {
        stable_diffusion::StableDiffusionConfig::v1_5(None, None, None)
    }

    /// Load all SD1.5 model components (Eager mode).
    ///
    /// On error, `self.loaded` remains `None` — all components are assembled into
    /// local variables and only stored in `self.loaded` on success, so partial loads
    /// cannot leave the engine in an inconsistent state.
    pub fn load(&mut self) -> Result<()> {
        if self.loaded.is_some() {
            return Ok(());
        }

        // Sequential mode defers loading to generate_sequential()
        if self.load_strategy == LoadStrategy::Sequential {
            return Ok(());
        }

        let (clip_encoder, clip_tokenizer) = self.validate_paths()?;

        tracing::info!(model = %self.model_name, "loading SD1.5 model components...");

        let device = crate::device::create_device(&self.progress)?;
        let dtype = if device.is_cuda() || device.is_metal() {
            DType::F16
        } else {
            DType::F32
        };

        let sd_config = self.sd_config();

        // Load UNet
        self.progress.stage_start("Loading UNet (GPU)");
        let unet_start = Instant::now();
        let unet = sd_config.build_unet(
            &self.paths.transformer,
            &device,
            4,     // in_channels
            false, // use_flash_attn
            dtype,
        )?;
        self.progress
            .stage_done("Loading UNet (GPU)", unet_start.elapsed());

        // Load VAE
        self.progress.stage_start("Loading VAE (GPU)");
        let vae_start = Instant::now();
        let vae = sd_config.build_vae(&self.paths.vae, &device, dtype)?;
        self.progress
            .stage_done("Loading VAE (GPU)", vae_start.elapsed());

        // Load CLIP-L encoder
        self.progress.stage_start("Loading CLIP-L encoder");
        let clip_start = Instant::now();
        let clip = stable_diffusion::build_clip_transformer(
            &sd_config.clip,
            &clip_encoder,
            &device,
            DType::F32,
        )?;
        self.progress
            .stage_done("Loading CLIP-L encoder", clip_start.elapsed());

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(&clip_tokenizer)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP-L tokenizer: {e}"))?;

        self.loaded = Some(LoadedSD15 {
            unet,
            vae,
            clip,
            tokenizer,
            sd_config,
            device,
            dtype,
        });

        tracing::info!(model = %self.model_name, "all SD1.5 components loaded successfully");
        Ok(())
    }

    /// Tokenize a prompt for the CLIP encoder, padding/truncating to max_len tokens.
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
        // Pad with 0s (EOS/PAD token for CLIP)
        while ids.len() < max_len {
            ids.push(0);
        }
        let ids = ids.into_iter().map(|i| i as i64).collect::<Vec<_>>();
        Ok(Tensor::new(ids, device)?.unsqueeze(0)?)
    }

    /// Run the denoising loop (shared between eager and sequential).
    fn denoise_loop(
        &self,
        unet: &stable_diffusion::unet_2d::UNet2DConditionModel,
        text_embeddings: &Tensor,
        sd_config: &stable_diffusion::StableDiffusionConfig,
        latents: &mut Tensor,
        guidance: f64,
        steps: u32,
    ) -> Result<()> {
        let use_cfg = guidance > 1.0;
        let mut scheduler = sd_config.build_scheduler(steps as usize)?;
        let timesteps = scheduler.timesteps().to_vec();

        let denoise_label = format!("Denoising ({} steps)", timesteps.len());
        self.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        for (step_idx, &t) in timesteps.iter().enumerate() {
            let step_start = std::time::Instant::now();
            let latent_input = if use_cfg {
                Tensor::cat(&[&*latents, &*latents], 0)?
            } else {
                latents.clone()
            };

            let latent_input = scheduler.scale_model_input(latent_input, t)?;
            let noise_pred = unet.forward(&latent_input, t as f64, text_embeddings)?;

            let noise_pred = if use_cfg {
                let chunks = noise_pred.chunk(2, 0)?;
                let noise_pred_uncond = &chunks[0];
                let noise_pred_cond = &chunks[1];
                (noise_pred_uncond + ((noise_pred_cond - noise_pred_uncond)? * guidance)?)?
            } else {
                noise_pred
            };

            *latents = scheduler.step(&noise_pred, t, &*latents)?;
            self.progress.emit(ProgressEvent::DenoiseStep {
                step: step_idx + 1,
                total: timesteps.len(),
                elapsed: step_start.elapsed(),
            });
        }

        self.progress
            .stage_done(&denoise_label, denoise_start.elapsed());
        Ok(())
    }

    /// Encode prompt with the single CLIP-L encoder.
    #[allow(clippy::too_many_arguments)]
    fn encode_prompt(
        &self,
        clip: &stable_diffusion::clip::ClipTextTransformer,
        tokenizer: &tokenizers::Tokenizer,
        prompt: &str,
        max_len: usize,
        device: &Device,
        dtype: DType,
        guidance: f64,
    ) -> Result<Tensor> {
        let use_cfg = guidance > 1.0;

        self.progress.stage_start("Encoding prompt (CLIP-L)");
        let encode_start = Instant::now();
        let tokens = Self::tokenize(tokenizer, prompt, max_len, device)?;
        let text_embeddings = clip.forward(&tokens)?;
        self.progress
            .stage_done("Encoding prompt (CLIP-L)", encode_start.elapsed());

        let text_embeddings = if use_cfg {
            let uncond_tokens = Self::tokenize(tokenizer, "", max_len, device)?;
            let uncond_embeddings = clip.forward(&uncond_tokens)?;
            Tensor::cat(&[&uncond_embeddings, &text_embeddings], 0)?
        } else {
            text_embeddings
        };

        Ok(text_embeddings.to_dtype(dtype)?)
    }

    /// Generate an image using sequential loading strategy.
    ///
    /// Loads components one at a time and drops them when done:
    /// 1. Load CLIP-L → encode → drop CLIP-L
    /// 2. Load UNet → denoise → drop UNet
    /// 3. Load VAE → decode → drop VAE
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let (clip_encoder, clip_tokenizer) = self.validate_paths()?;

        // Check memory budget
        if let Some(warning) = check_memory_budget(&self.paths, LoadStrategy::Sequential) {
            self.progress.info(&warning);
        }

        let device = crate::device::create_device(&self.progress)?;
        let dtype = if device.is_cuda() || device.is_metal() {
            DType::F16
        } else {
            DType::F32
        };

        let sd_config = self.sd_config();
        let max_len = sd_config.clip.max_position_embeddings;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;
        let guidance = req.guidance;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            guidance,
            "starting sequential SD1.5 generation"
        );

        self.progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: Load CLIP-L encoder, encode, then drop ---
        if let Some(status) = memory_status_string() {
            self.progress.info(&status);
        }

        // Load tokenizer (kept in memory — tiny)
        let tokenizer = tokenizers::Tokenizer::from_file(&clip_tokenizer)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP-L tokenizer: {e}"))?;

        // Load CLIP-L
        self.progress.stage_start("Loading CLIP-L encoder");
        let clip_start = Instant::now();
        let clip = stable_diffusion::build_clip_transformer(
            &sd_config.clip,
            &clip_encoder,
            &device,
            DType::F32,
        )?;
        self.progress
            .stage_done("Loading CLIP-L encoder", clip_start.elapsed());

        // Encode prompt
        let text_embeddings = self.encode_prompt(
            &clip,
            &tokenizer,
            &req.prompt,
            max_len,
            &device,
            dtype,
            guidance,
        )?;

        // Drop CLIP encoder to free memory
        drop(clip);
        self.progress.info("Freed CLIP-L encoder");
        tracing::info!("CLIP encoder dropped (sequential mode)");

        // --- Phase 2: Load UNet and denoise ---
        let unet_size = std::fs::metadata(&self.paths.transformer)
            .map(|m| m.len())
            .unwrap_or(0);
        preflight_memory_check("UNet", unet_size)?;
        if let Some(status) = memory_status_string() {
            self.progress.info(&status);
        }

        self.progress.stage_start("Loading UNet (GPU)");
        let unet_start = Instant::now();
        let unet = sd_config.build_unet(&self.paths.transformer, &device, 4, false, dtype)?;
        self.progress
            .stage_done("Loading UNet (GPU)", unet_start.elapsed());

        let latent_h = height / 8;
        let latent_w = width / 8;
        let scheduler = sd_config.build_scheduler(req.steps as usize)?;
        let init_noise_sigma = scheduler.init_noise_sigma();
        let mut latents =
            (crate::engine::seeded_randn(seed, &[1, 4, latent_h, latent_w], &device, DType::F32)?
                * init_noise_sigma)?;
        latents = latents.to_dtype(dtype)?;

        self.denoise_loop(
            &unet,
            &text_embeddings,
            &sd_config,
            &mut latents,
            guidance,
            req.steps,
        )?;

        // Drop UNet to free memory for VAE decode
        drop(unet);
        drop(text_embeddings);
        device.synchronize()?;
        self.progress.info("Freed UNet");
        tracing::info!("UNet dropped (sequential mode)");

        // --- Phase 3: Load VAE and decode ---
        self.progress.stage_start("Loading VAE (GPU)");
        let vae_start = Instant::now();
        let vae = sd_config.build_vae(&self.paths.vae, &device, dtype)?;
        self.progress
            .stage_done("Loading VAE (GPU)", vae_start.elapsed());

        self.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();

        let latents = (latents / VAE_SCALE)?;
        let img = vae.decode(&latents.to_dtype(dtype)?)?;

        let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
        let img = (img * 255.)?.to_dtype(DType::U8)?;
        let img = img.squeeze(0)?;

        self.progress
            .stage_done("VAE decode", vae_decode_start.elapsed());

        let image_bytes = encode_image(&img, req.output_format, req.width, req.height)?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            generation_time_ms,
            seed,
            "sequential SD1.5 generation complete"
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

impl InferenceEngine for SD15Engine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        // Sequential mode: load-use-drop each component
        if self.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components
        let loaded = self
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;
        let guidance = req.guidance;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            guidance,
            scheduler = %self.scheduler_name,
            "starting SD1.5 generation"
        );

        // 1. Encode prompt with CLIP-L
        let max_len = loaded.sd_config.clip.max_position_embeddings;
        let text_embeddings = self.encode_prompt(
            &loaded.clip,
            &loaded.tokenizer,
            &req.prompt,
            max_len,
            &loaded.device,
            loaded.dtype,
            guidance,
        )?;

        // 2. Build scheduler and create initial latents
        let latent_h = height / 8;
        let latent_w = width / 8;
        let scheduler = loaded.sd_config.build_scheduler(req.steps as usize)?;
        let init_noise_sigma = scheduler.init_noise_sigma();
        let mut latents = (crate::engine::seeded_randn(
            seed,
            &[1, 4, latent_h, latent_w],
            &loaded.device,
            DType::F32,
        )? * init_noise_sigma)?;
        latents = latents.to_dtype(loaded.dtype)?;

        // 3. Denoising loop
        self.denoise_loop(
            &loaded.unet,
            &text_embeddings,
            &loaded.sd_config,
            &mut latents,
            guidance,
            req.steps,
        )?;

        // 4. VAE decode
        self.progress.stage_start("VAE decode");
        let vae_start = Instant::now();

        let latents = (latents / VAE_SCALE)?;
        let img = loaded.vae.decode(&latents.to_dtype(loaded.dtype)?)?;

        let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
        let img = (img * 255.)?.to_dtype(DType::U8)?;
        let img = img.squeeze(0)?;

        self.progress.stage_done("VAE decode", vae_start.elapsed());

        // 5. Encode to image format
        let image_bytes = encode_image(&img, req.output_format, req.width, req.height)?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "SD1.5 generation complete");

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
        // Sequential mode is always "ready" — it loads on demand
        self.load_strategy == LoadStrategy::Sequential || self.loaded.is_some()
    }

    fn load(&mut self) -> Result<()> {
        SD15Engine::load(self)
    }

    fn unload(&mut self) {
        self.loaded = None;
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }
}
