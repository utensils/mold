use anyhow::{Result, bail};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_transformers::models::stable_diffusion;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::time::Instant;

use crate::device::{check_memory_budget, memory_status_string, preflight_memory_check};
use crate::engine::{InferenceEngine, LoadStrategy, rand_seed};
use crate::image::encode_image;
use crate::progress::{ProgressCallback, ProgressEvent, ProgressReporter};

/// Loaded SDXL model components, ready for inference.
struct LoadedSDXL {
    unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    vae: stable_diffusion::vae::AutoEncoderKL,
    clip_l: stable_diffusion::clip::ClipTextTransformer,
    clip_g: stable_diffusion::clip::ClipTextTransformer,
    tokenizer_l: tokenizers::Tokenizer,
    tokenizer_g: tokenizers::Tokenizer,
    sd_config: stable_diffusion::StableDiffusionConfig,
    device: Device,
    dtype: DType,
}

/// SDXL inference engine backed by candle's stable_diffusion module.
pub struct SDXLEngine {
    loaded: Option<LoadedSDXL>,
    model_name: String,
    paths: ModelPaths,
    scheduler_name: String,
    is_turbo: bool,
    progress: ProgressReporter,
    /// How to load model components (Eager = all at once, Sequential = load-use-drop).
    load_strategy: LoadStrategy,
}

/// VAE scaling factor for standard SDXL models.
const VAE_SCALE_STANDARD: f64 = 0.18215;
/// VAE scaling factor for SDXL Turbo models.
const VAE_SCALE_TURBO: f64 = 0.13025;

impl SDXLEngine {
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        scheduler_name: String,
        is_turbo: bool,
        load_strategy: LoadStrategy,
    ) -> Self {
        Self {
            loaded: None,
            model_name,
            paths,
            scheduler_name,
            is_turbo,
            progress: ProgressReporter::default(),
            load_strategy,
        }
    }

    /// Validate and return required SDXL paths.
    fn validate_paths(
        &self,
    ) -> Result<(
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
    )> {
        let clip_encoder = self
            .paths
            .clip_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-L encoder path required for SDXL models"))?
            .clone();
        let clip_tokenizer = self
            .paths
            .clip_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-L tokenizer path required for SDXL models"))?
            .clone();
        let clip_encoder_2 = self
            .paths
            .clip_encoder_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-G encoder path required for SDXL models"))?
            .clone();
        let clip_tokenizer_2 = self
            .paths
            .clip_tokenizer_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-G tokenizer path required for SDXL models"))?
            .clone();

        for (label, path) in [
            ("transformer (UNet)", &self.paths.transformer),
            ("vae", &self.paths.vae),
            ("clip_encoder (CLIP-L)", &clip_encoder),
            ("clip_tokenizer (CLIP-L)", &clip_tokenizer),
            ("clip_encoder_2 (CLIP-G)", &clip_encoder_2),
            ("clip_tokenizer_2 (CLIP-G)", &clip_tokenizer_2),
        ] {
            if !path.exists() {
                bail!("{label} file not found: {}", path.display());
            }
        }

        Ok((
            clip_encoder,
            clip_tokenizer,
            clip_encoder_2,
            clip_tokenizer_2,
        ))
    }

    /// Create the SDXL config.
    fn sd_config(&self) -> stable_diffusion::StableDiffusionConfig {
        if self.is_turbo {
            stable_diffusion::StableDiffusionConfig::sdxl_turbo(None, None, None)
        } else {
            stable_diffusion::StableDiffusionConfig::sdxl(None, None, None)
        }
    }

    /// Load all SDXL model components (Eager mode).
    pub fn load(&mut self) -> Result<()> {
        if self.loaded.is_some() {
            return Ok(());
        }

        // Sequential mode defers loading to generate_sequential()
        if self.load_strategy == LoadStrategy::Sequential {
            return Ok(());
        }

        let (clip_encoder, clip_tokenizer, clip_encoder_2, clip_tokenizer_2) =
            self.validate_paths()?;

        tracing::info!(model = %self.model_name, "loading SDXL model components...");

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
        let clip_l_start = Instant::now();
        let clip_l = stable_diffusion::build_clip_transformer(
            &sd_config.clip,
            &clip_encoder,
            &device,
            DType::F32,
        )?;
        self.progress
            .stage_done("Loading CLIP-L encoder", clip_l_start.elapsed());

        // Load CLIP-G encoder
        self.progress.stage_start("Loading CLIP-G encoder");
        let clip_g_start = Instant::now();
        let clip2_config = sd_config
            .clip2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("SDXL config missing clip2 configuration"))?;
        let clip_g = stable_diffusion::build_clip_transformer(
            clip2_config,
            &clip_encoder_2,
            &device,
            DType::F32,
        )?;
        self.progress
            .stage_done("Loading CLIP-G encoder", clip_g_start.elapsed());

        // Load tokenizers
        let tokenizer_l = tokenizers::Tokenizer::from_file(&clip_tokenizer)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP-L tokenizer: {e}"))?;
        let tokenizer_g = tokenizers::Tokenizer::from_file(&clip_tokenizer_2)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP-G tokenizer: {e}"))?;

        self.loaded = Some(LoadedSDXL {
            unet,
            vae,
            clip_l,
            clip_g,
            tokenizer_l,
            tokenizer_g,
            sd_config,
            device,
            dtype,
        });

        tracing::info!(model = %self.model_name, "all SDXL components loaded successfully");
        Ok(())
    }

    /// Tokenize a prompt for a CLIP encoder, padding/truncating to max_len tokens.
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

    /// Encode prompt with both CLIP encoders.
    #[allow(clippy::too_many_arguments)]
    fn encode_prompt(
        &self,
        clip_l: &stable_diffusion::clip::ClipTextTransformer,
        clip_g: &stable_diffusion::clip::ClipTextTransformer,
        tokenizer_l: &tokenizers::Tokenizer,
        tokenizer_g: &tokenizers::Tokenizer,
        prompt: &str,
        max_len: usize,
        device: &Device,
        dtype: DType,
        guidance: f64,
    ) -> Result<Tensor> {
        let use_cfg = guidance > 1.0;

        self.progress.stage_start("Encoding prompt (CLIP-L)");
        let encode_l_start = Instant::now();
        let tokens_l = Self::tokenize(tokenizer_l, prompt, max_len, device)?;
        let text_emb_l = clip_l.forward(&tokens_l)?;
        self.progress
            .stage_done("Encoding prompt (CLIP-L)", encode_l_start.elapsed());

        self.progress.stage_start("Encoding prompt (CLIP-G)");
        let encode_g_start = Instant::now();
        let tokens_g = Self::tokenize(tokenizer_g, prompt, max_len, device)?;
        let text_emb_g = clip_g.forward(&tokens_g)?;
        self.progress
            .stage_done("Encoding prompt (CLIP-G)", encode_g_start.elapsed());

        let text_embeddings = Tensor::cat(&[&text_emb_l, &text_emb_g], D::Minus1)?;

        let text_embeddings = if use_cfg {
            let uncond_tokens_l = Self::tokenize(tokenizer_l, "", max_len, device)?;
            let uncond_emb_l = clip_l.forward(&uncond_tokens_l)?;
            let uncond_tokens_g = Self::tokenize(tokenizer_g, "", max_len, device)?;
            let uncond_emb_g = clip_g.forward(&uncond_tokens_g)?;
            let uncond_embeddings = Tensor::cat(&[&uncond_emb_l, &uncond_emb_g], D::Minus1)?;
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
    /// 2. Load CLIP-G → encode → drop CLIP-G
    /// 3. Load UNet → denoise → drop UNet
    /// 4. Load VAE → decode → drop VAE
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let (clip_encoder, clip_tokenizer, clip_encoder_2, clip_tokenizer_2) =
            self.validate_paths()?;

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
        device.set_seed(seed)?;

        let width = req.width as usize;
        let height = req.height as usize;
        let guidance = req.guidance;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            guidance,
            "starting sequential SDXL generation"
        );

        self.progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: Load both CLIP encoders, encode, then drop ---
        // SDXL CLIP encoders are small (~1.7GB total) so we load both for encoding,
        // then drop them together before loading the UNet.
        if let Some(status) = memory_status_string() {
            self.progress.info(&status);
        }

        // Load tokenizers (kept in memory — tiny)
        let tokenizer_l = tokenizers::Tokenizer::from_file(&clip_tokenizer)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP-L tokenizer: {e}"))?;
        let tokenizer_g = tokenizers::Tokenizer::from_file(&clip_tokenizer_2)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP-G tokenizer: {e}"))?;

        // Load CLIP-L
        self.progress.stage_start("Loading CLIP-L encoder");
        let clip_l_start = Instant::now();
        let clip_l = stable_diffusion::build_clip_transformer(
            &sd_config.clip,
            &clip_encoder,
            &device,
            DType::F32,
        )?;
        self.progress
            .stage_done("Loading CLIP-L encoder", clip_l_start.elapsed());

        // Load CLIP-G
        self.progress.stage_start("Loading CLIP-G encoder");
        let clip_g_start = Instant::now();
        let clip2_config = sd_config
            .clip2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("SDXL config missing clip2 configuration"))?;
        let clip_g = stable_diffusion::build_clip_transformer(
            clip2_config,
            &clip_encoder_2,
            &device,
            DType::F32,
        )?;
        self.progress
            .stage_done("Loading CLIP-G encoder", clip_g_start.elapsed());

        // Encode prompt
        let text_embeddings = self.encode_prompt(
            &clip_l,
            &clip_g,
            &tokenizer_l,
            &tokenizer_g,
            &req.prompt,
            max_len,
            &device,
            dtype,
            guidance,
        )?;

        // Drop CLIP encoders to free memory
        drop(clip_l);
        drop(clip_g);
        self.progress.info("Freed CLIP-L and CLIP-G encoders");
        tracing::info!("CLIP encoders dropped (sequential mode)");

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
            (Tensor::randn(0f32, 1f32, &[1, 4, latent_h, latent_w], &device)? * init_noise_sigma)?;
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

        let vae_scale = if self.is_turbo {
            VAE_SCALE_TURBO
        } else {
            VAE_SCALE_STANDARD
        };
        let latents = (latents / vae_scale)?;
        let img = vae.decode(&latents.to_dtype(dtype)?)?;

        let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
        let img = (img * 255.)?.to_dtype(DType::U8)?;
        let img = img.squeeze(0)?;

        self.progress
            .stage_done("VAE decode", vae_decode_start.elapsed());

        // VAE dropped here
        let image_bytes = encode_image(&img, req.output_format, req.width, req.height)?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            generation_time_ms,
            seed,
            "sequential SDXL generation complete"
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

impl InferenceEngine for SDXLEngine {
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
        loaded.device.set_seed(seed)?;

        let width = req.width as usize;
        let height = req.height as usize;
        let guidance = req.guidance;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            guidance,
            scheduler = %self.scheduler_name,
            "starting SDXL generation"
        );

        // 1. Encode prompt with both CLIP encoders
        let max_len = loaded.sd_config.clip.max_position_embeddings;
        let text_embeddings = self.encode_prompt(
            &loaded.clip_l,
            &loaded.clip_g,
            &loaded.tokenizer_l,
            &loaded.tokenizer_g,
            &req.prompt,
            max_len,
            &loaded.device,
            loaded.dtype,
            guidance,
        )?;

        // 3. Build scheduler
        let latent_h = height / 8;
        let latent_w = width / 8;
        let scheduler = loaded.sd_config.build_scheduler(req.steps as usize)?;
        let init_noise_sigma = scheduler.init_noise_sigma();
        let mut latents =
            (Tensor::randn(0f32, 1f32, &[1, 4, latent_h, latent_w], &loaded.device)?
                * init_noise_sigma)?;
        latents = latents.to_dtype(loaded.dtype)?;

        // 5. Denoising loop
        self.denoise_loop(
            &loaded.unet,
            &text_embeddings,
            &loaded.sd_config,
            &mut latents,
            guidance,
            req.steps,
        )?;

        // 6. VAE decode
        self.progress.stage_start("VAE decode");
        let vae_start = Instant::now();

        let vae_scale = if self.is_turbo {
            VAE_SCALE_TURBO
        } else {
            VAE_SCALE_STANDARD
        };
        let latents = (latents / vae_scale)?;
        let img = loaded.vae.decode(&latents.to_dtype(loaded.dtype)?)?;

        // 7. Post-process: [1, 3, H, W] → clamp → u8
        let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
        let img = (img * 255.)?.to_dtype(DType::U8)?;
        let img = img.squeeze(0)?; // [3, H, W]

        self.progress.stage_done("VAE decode", vae_start.elapsed());

        // 8. Encode to image format
        let image_bytes = encode_image(&img, req.output_format, req.width, req.height)?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "SDXL generation complete");

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
        SDXLEngine::load(self)
    }

    fn unload(&mut self) {
        self.loaded = None;
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }
}
