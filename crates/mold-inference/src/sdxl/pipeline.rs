use anyhow::{bail, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::stable_diffusion::schedulers::PredictionType;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths, Scheduler};
use std::sync::Mutex;
use std::time::Instant;

use crate::cache::{
    clear_cache, get_or_insert_cached_tensor, image_size_cache_key, latent_size_cache_key,
    prompt_cache_key, restore_cached_tensor, CachedTensor, ImageSizeCacheKey, LatentSizeCacheKey,
    LruCache, PromptCacheKey, DEFAULT_IMAGE_CACHE_CAPACITY, DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{check_memory_budget, memory_status_string, preflight_memory_check};
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image};
use crate::progress::{ProgressCallback, ProgressEvent};

/// Loaded SDXL model components, ready for inference.
struct LoadedSDXL {
    /// None after being dropped for VAE decode VRAM; reloaded on next generate.
    unet: Option<stable_diffusion::unet_2d::UNet2DConditionModel>,
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
    base: EngineBase<LoadedSDXL>,
    scheduler: Scheduler,
    is_turbo: bool,
    prompt_cache: Mutex<LruCache<PromptCacheKey, CachedTensor>>,
    source_latent_cache: Mutex<LruCache<ImageSizeCacheKey, CachedTensor>>,
    mask_cache: Mutex<LruCache<LatentSizeCacheKey, CachedTensor>>,
}

/// VAE scaling factor for standard SDXL models.
const VAE_SCALE_STANDARD: f64 = 0.18215;
/// VAE scaling factor for SDXL Turbo models.
const VAE_SCALE_TURBO: f64 = 0.13025;

impl SDXLEngine {
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        scheduler: Scheduler,
        is_turbo: bool,
        load_strategy: LoadStrategy,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy),
            scheduler,
            is_turbo,
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            source_latent_cache: Mutex::new(LruCache::new(DEFAULT_IMAGE_CACHE_CAPACITY)),
            mask_cache: Mutex::new(LruCache::new(DEFAULT_IMAGE_CACHE_CAPACITY)),
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
            .base
            .paths
            .clip_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-L encoder path required for SDXL models"))?
            .clone();
        let clip_tokenizer = self
            .base
            .paths
            .clip_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-L tokenizer path required for SDXL models"))?
            .clone();
        let clip_encoder_2 = self
            .base
            .paths
            .clip_encoder_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-G encoder path required for SDXL models"))?
            .clone();
        let clip_tokenizer_2 = self
            .base
            .paths
            .clip_tokenizer_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-G tokenizer path required for SDXL models"))?
            .clone();

        for (label, path) in [
            ("transformer (UNet)", &self.base.paths.transformer),
            ("vae", &self.base.paths.vae),
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

    /// Reload UNet if it was dropped after VAE decode.
    fn reload_unet_if_needed(&mut self) -> Result<()> {
        let needs_reload = self
            .base
            .loaded
            .as_ref()
            .map(|l| l.unet.is_none())
            .unwrap_or(false);

        if needs_reload {
            let sd_config = self.sd_config();
            let loaded = self.base.loaded.as_ref().unwrap();
            let device = loaded.device.clone();
            let dtype = loaded.dtype;
            let _ = loaded;

            self.base.progress.stage_start("Reloading UNet (GPU)");
            let reload_start = Instant::now();
            let unet =
                sd_config.build_unet(&self.base.paths.transformer, &device, 4, false, dtype)?;
            self.base.loaded.as_mut().unwrap().unet = Some(unet);
            self.base
                .progress
                .stage_done("Reloading UNet (GPU)", reload_start.elapsed());
        }
        Ok(())
    }

    /// Load all SDXL model components (Eager mode).
    ///
    /// On error, `self.base.loaded` remains `None` — all components are assembled into
    /// local variables and only stored in `self.base.loaded` on success, so partial loads
    /// cannot leave the engine in an inconsistent state.
    pub fn load(&mut self) -> Result<()> {
        if self.base.loaded.is_some() {
            return Ok(());
        }

        // Sequential mode defers loading to generate_sequential()
        if self.base.load_strategy == LoadStrategy::Sequential {
            return Ok(());
        }

        let (clip_encoder, clip_tokenizer, clip_encoder_2, clip_tokenizer_2) =
            self.validate_paths()?;

        tracing::info!(model = %self.base.model_name, "loading SDXL model components...");

        let device = crate::device::create_device(&self.base.progress)?;
        let dtype = if crate::device::is_gpu(&device) {
            DType::F16
        } else {
            DType::F32
        };

        let sd_config = self.sd_config();

        // Load UNet
        self.base.progress.stage_start("Loading UNet (GPU)");
        let unet_start = Instant::now();
        let unet = sd_config.build_unet(
            &self.base.paths.transformer,
            &device,
            4,     // in_channels
            false, // use_flash_attn
            dtype,
        )?;
        self.base
            .progress
            .stage_done("Loading UNet (GPU)", unet_start.elapsed());

        // Load VAE
        self.base.progress.stage_start("Loading VAE (GPU)");
        let vae_start = Instant::now();
        let vae = sd_config.build_vae(&self.base.paths.vae, &device, dtype)?;
        self.base
            .progress
            .stage_done("Loading VAE (GPU)", vae_start.elapsed());

        // Load CLIP-L encoder
        self.base.progress.stage_start("Loading CLIP-L encoder");
        let clip_l_start = Instant::now();
        let clip_l = stable_diffusion::build_clip_transformer(
            &sd_config.clip,
            &clip_encoder,
            &device,
            DType::F32,
        )?;
        self.base
            .progress
            .stage_done("Loading CLIP-L encoder", clip_l_start.elapsed());

        // Load CLIP-G encoder
        self.base.progress.stage_start("Loading CLIP-G encoder");
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
        self.base
            .progress
            .stage_done("Loading CLIP-G encoder", clip_g_start.elapsed());

        // Load tokenizers
        let tokenizer_l = tokenizers::Tokenizer::from_file(&clip_tokenizer)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP-L tokenizer: {e}"))?;
        let tokenizer_g = tokenizers::Tokenizer::from_file(&clip_tokenizer_2)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP-G tokenizer: {e}"))?;

        self.base.loaded = Some(LoadedSDXL {
            unet: Some(unet),
            vae,
            clip_l,
            clip_g,
            tokenizer_l,
            tokenizer_g,
            sd_config,
            device,
            dtype,
        });

        tracing::info!(model = %self.base.model_name, "all SDXL components loaded successfully");
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
    ///
    /// `start_step` allows starting from a later timestep for img2img (0 = full txt2img).
    #[allow(clippy::too_many_arguments)]
    fn denoise_loop(
        &self,
        unet: &stable_diffusion::unet_2d::UNet2DConditionModel,
        text_embeddings: &Tensor,
        sched: Scheduler,
        latents: &mut Tensor,
        guidance: f64,
        steps: u32,
        start_step: usize,
        inpaint_ctx: Option<&crate::img_utils::InpaintContext>,
    ) -> Result<()> {
        let use_cfg = guidance > 1.0;
        let mut scheduler = crate::scheduler::build_scheduler(
            sched,
            steps as usize,
            PredictionType::Epsilon,
            self.is_turbo,
        )?;
        let timesteps = scheduler.timesteps().to_vec();
        let active_timesteps = &timesteps[start_step..];

        let denoise_label = format!("Denoising ({} steps)", active_timesteps.len());
        self.base.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        for (step_idx, &t) in active_timesteps.iter().enumerate() {
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

            if let Some(ctx) = inpaint_ctx {
                let noised_original =
                    scheduler.add_noise(&ctx.original_latents, ctx.noise.clone(), t)?;
                *latents = ((&ctx.mask * &*latents)? + (&(1.0 - &ctx.mask)? * &noised_original)?)?;
            }

            self.base.progress.emit(ProgressEvent::DenoiseStep {
                step: step_idx + 1,
                total: active_timesteps.len(),
                elapsed: step_start.elapsed(),
            });
        }

        self.base
            .progress
            .stage_done(&denoise_label, denoise_start.elapsed());
        Ok(())
    }

    /// Prepare img2img latents: VAE encode source image, add noise at the appropriate timestep.
    /// Returns (noised_latents, start_step, encoded, noise).
    #[allow(clippy::too_many_arguments)]
    fn prepare_img2img_latents(
        &self,
        vae: &stable_diffusion::vae::AutoEncoderKL,
        source_bytes: &[u8],
        width: u32,
        height: u32,
        strength: f64,
        steps: u32,
        sched: Scheduler,
        seed: u64,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, usize, Tensor, Tensor)> {
        use crate::img_utils::{decode_source_image, NormalizeRange};
        let vae_scale = if self.is_turbo {
            VAE_SCALE_TURBO
        } else {
            VAE_SCALE_STANDARD
        };
        let cache_key = image_size_cache_key(source_bytes, width, height);
        let (encoded, cache_hit) = get_or_insert_cached_tensor(
            &self.source_latent_cache,
            cache_key,
            device,
            dtype,
            || {
                self.base
                    .progress
                    .stage_start("Encoding source image (VAE)");
                let encode_start = Instant::now();

                let source_tensor = decode_source_image(
                    source_bytes,
                    width,
                    height,
                    NormalizeRange::MinusOneToOne,
                    device,
                    dtype,
                )?;
                let encoded = vae.encode(&source_tensor)?;
                let encoded = (encoded.sample()? * vae_scale)?;

                self.base
                    .progress
                    .stage_done("Encoding source image (VAE)", encode_start.elapsed());
                Ok(encoded)
            },
        )?;
        if cache_hit {
            self.base.progress.cache_hit("source image latents");
        }

        let start_step = ((steps as f64) * (1.0 - strength)).round() as usize;
        let start_step = start_step.min(steps as usize);

        let scheduler = crate::scheduler::build_scheduler(
            sched,
            steps as usize,
            PredictionType::Epsilon,
            self.is_turbo,
        )?;
        let timesteps = scheduler.timesteps().to_vec();

        let latent_h = height as usize / 8;
        let latent_w = width as usize / 8;
        let noise =
            crate::engine::seeded_randn(seed, &[1, 4, latent_h, latent_w], device, DType::F32)?;
        let noise = noise.to_dtype(dtype)?;

        let noised = if start_step < timesteps.len() {
            scheduler.add_noise(&encoded, noise.clone(), timesteps[start_step])?
        } else {
            encoded.clone()
        };

        tracing::info!(
            start_step,
            total_steps = steps,
            strength,
            "img2img: starting from step {start_step}"
        );

        Ok((noised, start_step, encoded, noise))
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
        negative_prompt: &str,
        max_len: usize,
        device: &Device,
        dtype: DType,
        guidance: f64,
    ) -> Result<Tensor> {
        let cache_key = prompt_cache_key(prompt, guidance);
        let (text_embeddings, cache_hit) =
            get_or_insert_cached_tensor(&self.prompt_cache, cache_key, device, dtype, || {
                let use_cfg = guidance > 1.0;

                self.base.progress.stage_start("Encoding prompt (CLIP-L)");
                let encode_l_start = Instant::now();
                let tokens_l = Self::tokenize(tokenizer_l, prompt, max_len, device)?;
                let text_emb_l = clip_l.forward(&tokens_l)?;
                self.base
                    .progress
                    .stage_done("Encoding prompt (CLIP-L)", encode_l_start.elapsed());

                self.base.progress.stage_start("Encoding prompt (CLIP-G)");
                let encode_g_start = Instant::now();
                let tokens_g = Self::tokenize(tokenizer_g, prompt, max_len, device)?;
                let text_emb_g = clip_g.forward(&tokens_g)?;
                self.base
                    .progress
                    .stage_done("Encoding prompt (CLIP-G)", encode_g_start.elapsed());

                let text_embeddings = Tensor::cat(&[&text_emb_l, &text_emb_g], D::Minus1)?;

                let text_embeddings = if use_cfg {
                    let uncond_tokens_l =
                        Self::tokenize(tokenizer_l, negative_prompt, max_len, device)?;
                    let uncond_emb_l = clip_l.forward(&uncond_tokens_l)?;
                    let uncond_tokens_g =
                        Self::tokenize(tokenizer_g, negative_prompt, max_len, device)?;
                    let uncond_emb_g = clip_g.forward(&uncond_tokens_g)?;
                    let uncond_embeddings =
                        Tensor::cat(&[&uncond_emb_l, &uncond_emb_g], D::Minus1)?;
                    Tensor::cat(&[&uncond_embeddings, &text_embeddings], 0)?
                } else {
                    text_embeddings
                };

                Ok(text_embeddings.to_dtype(dtype)?)
            })?;
        if cache_hit {
            self.base.progress.cache_hit("prompt conditioning");
            return Ok(text_embeddings);
        }
        Ok(text_embeddings)
    }

    fn cached_mask(
        &self,
        mask_bytes: &[u8],
        latent_h: usize,
        latent_w: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let key = latent_size_cache_key(mask_bytes, latent_h, latent_w);
        let (mask, cache_hit) =
            get_or_insert_cached_tensor(&self.mask_cache, key, device, dtype, || {
                crate::img_utils::decode_mask_image(mask_bytes, latent_h, latent_w, device, dtype)
            })?;
        if cache_hit {
            self.base.progress.cache_hit("inpaint mask");
            return Ok(mask);
        }
        Ok(mask)
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
        if let Some(warning) = check_memory_budget(&self.base.paths, LoadStrategy::Sequential) {
            self.base.progress.info(&warning);
        }

        let device = crate::device::create_device(&self.base.progress)?;
        let dtype = if crate::device::is_gpu(&device) {
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
            "starting sequential SDXL generation"
        );

        self.base
            .progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: Encode prompt (check cache first to skip encoder load) ---
        let neg = req.negative_prompt.as_deref().unwrap_or("");
        let cache_key = prompt_cache_key(&req.prompt, guidance);
        let text_embeddings = if let Some(tensor) =
            restore_cached_tensor(&self.prompt_cache, &cache_key, &device, dtype)?
        {
            self.base.progress.cache_hit("prompt conditioning");
            tensor
        } else {
            if let Some(status) = memory_status_string() {
                self.base.progress.info(&status);
            }

            let tokenizer_l = tokenizers::Tokenizer::from_file(&clip_tokenizer)
                .map_err(|e| anyhow::anyhow!("failed to load CLIP-L tokenizer: {e}"))?;
            let tokenizer_g = tokenizers::Tokenizer::from_file(&clip_tokenizer_2)
                .map_err(|e| anyhow::anyhow!("failed to load CLIP-G tokenizer: {e}"))?;

            self.base.progress.stage_start("Loading CLIP-L encoder");
            let clip_l_start = Instant::now();
            let clip_l = stable_diffusion::build_clip_transformer(
                &sd_config.clip,
                &clip_encoder,
                &device,
                DType::F32,
            )?;
            self.base
                .progress
                .stage_done("Loading CLIP-L encoder", clip_l_start.elapsed());

            self.base.progress.stage_start("Loading CLIP-G encoder");
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
            self.base
                .progress
                .stage_done("Loading CLIP-G encoder", clip_g_start.elapsed());

            let text_embeddings = self.encode_prompt(
                &clip_l,
                &clip_g,
                &tokenizer_l,
                &tokenizer_g,
                &req.prompt,
                neg,
                max_len,
                &device,
                dtype,
                guidance,
            )?;

            drop(clip_l);
            drop(clip_g);
            self.base.progress.info("Freed CLIP-L and CLIP-G encoders");
            tracing::info!("CLIP encoders dropped (sequential mode)");

            text_embeddings
        };

        // --- Phase 2: Load UNet and denoise ---
        let unet_size = std::fs::metadata(&self.base.paths.transformer)
            .map(|m| m.len())
            .unwrap_or(0);
        preflight_memory_check("UNet", unet_size)?;
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        self.base.progress.stage_start("Loading UNet (GPU)");
        let unet_start = Instant::now();
        let unet = sd_config.build_unet(&self.base.paths.transformer, &device, 4, false, dtype)?;
        self.base
            .progress
            .stage_done("Loading UNet (GPU)", unet_start.elapsed());

        let sched = req.scheduler.unwrap_or(self.scheduler);
        let is_img2img = req.source_image.is_some();

        let (mut latents, start_step, inpaint_ctx) = if let Some(ref source_bytes) =
            req.source_image
        {
            self.base
                .progress
                .info("img2img mode: encoding source image before denoising");

            self.base.progress.stage_start("Loading VAE (GPU)");
            let vae_start_t = Instant::now();
            let vae = sd_config.build_vae(&self.base.paths.vae, &device, dtype)?;
            self.base
                .progress
                .stage_done("Loading VAE (GPU)", vae_start_t.elapsed());

            let (latents, start_step, encoded, noise) = self.prepare_img2img_latents(
                &vae,
                source_bytes,
                req.width,
                req.height,
                req.strength,
                req.steps,
                sched,
                seed,
                &device,
                dtype,
            )?;

            let inpaint_ctx = if let Some(ref mask_bytes) = req.mask_image {
                let mask = self.cached_mask(mask_bytes, height / 8, width / 8, &device, dtype)?;
                Some(crate::img_utils::InpaintContext {
                    original_latents: encoded,
                    mask,
                    noise,
                })
            } else {
                None
            };

            drop(vae);
            self.base
                .progress
                .info("Freed VAE (will reload for decode)");
            device.synchronize()?;

            (latents, start_step, inpaint_ctx)
        } else {
            let latent_h = height / 8;
            let latent_w = width / 8;
            let init_scheduler = crate::scheduler::build_scheduler(
                sched,
                req.steps as usize,
                PredictionType::Epsilon,
                self.is_turbo,
            )?;
            let init_noise_sigma = init_scheduler.init_noise_sigma();
            drop(init_scheduler);
            let latents = (crate::engine::seeded_randn(
                seed,
                &[1, 4, latent_h, latent_w],
                &device,
                DType::F32,
            )? * init_noise_sigma)?;
            (latents.to_dtype(dtype)?, 0, None)
        };

        self.denoise_loop(
            &unet,
            &text_embeddings,
            sched,
            &mut latents,
            guidance,
            req.steps,
            start_step,
            inpaint_ctx.as_ref(),
        )?;

        drop(inpaint_ctx);
        drop(unet);
        drop(text_embeddings);
        device.synchronize()?;
        self.base.progress.info("Freed UNet");
        tracing::info!("UNet dropped (sequential mode)");

        // --- Phase 3: Load VAE and decode ---
        let vae_load_label = if is_img2img {
            "Reloading VAE (GPU)"
        } else {
            "Loading VAE (GPU)"
        };
        self.base.progress.stage_start(vae_load_label);
        let vae_start = Instant::now();
        let vae = sd_config.build_vae(&self.base.paths.vae, &device, dtype)?;
        self.base
            .progress
            .stage_done(vae_load_label, vae_start.elapsed());

        self.base.progress.stage_start("VAE decode");
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

        self.base
            .progress
            .stage_done("VAE decode", vae_decode_start.elapsed());

        // VAE dropped here
        let output_metadata = build_output_metadata(req, seed, Some(sched));
        let image_bytes = encode_image(
            &img,
            req.output_format,
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

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
            video: None,
        })
    }
}

impl InferenceEngine for SDXLEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        // Sequential mode: load-use-drop each component
        if self.base.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Eager mode: reload UNet if dropped after previous VAE decode
        self.reload_unet_if_needed()?;

        let loaded = self
            .base
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
            scheduler = %self.scheduler,
            "starting SDXL generation"
        );

        // 1. Encode prompt with both CLIP encoders
        let max_len = loaded.sd_config.clip.max_position_embeddings;
        let neg = req.negative_prompt.as_deref().unwrap_or("");
        let text_embeddings = self.encode_prompt(
            &loaded.clip_l,
            &loaded.clip_g,
            &loaded.tokenizer_l,
            &loaded.tokenizer_g,
            &req.prompt,
            neg,
            max_len,
            &loaded.device,
            loaded.dtype,
            guidance,
        )?;

        // 3. Build scheduler and create initial latents
        let sched = req.scheduler.unwrap_or(self.scheduler);

        let (mut latents, start_step, inpaint_ctx) =
            if let Some(ref source_bytes) = req.source_image {
                let (latents, start_step, encoded, noise) = self.prepare_img2img_latents(
                    &loaded.vae,
                    source_bytes,
                    req.width,
                    req.height,
                    req.strength,
                    req.steps,
                    sched,
                    seed,
                    &loaded.device,
                    loaded.dtype,
                )?;
                let inpaint_ctx = if let Some(ref mask_bytes) = req.mask_image {
                    let mask = self.cached_mask(
                        mask_bytes,
                        height / 8,
                        width / 8,
                        &loaded.device,
                        loaded.dtype,
                    )?;
                    Some(crate::img_utils::InpaintContext {
                        original_latents: encoded,
                        mask,
                        noise,
                    })
                } else {
                    None
                };
                (latents, start_step, inpaint_ctx)
            } else {
                let latent_h = height / 8;
                let latent_w = width / 8;
                let init_scheduler = crate::scheduler::build_scheduler(
                    sched,
                    req.steps as usize,
                    PredictionType::Epsilon,
                    self.is_turbo,
                )?;
                let init_noise_sigma = init_scheduler.init_noise_sigma();
                drop(init_scheduler);
                let latents = (crate::engine::seeded_randn(
                    seed,
                    &[1, 4, latent_h, latent_w],
                    &loaded.device,
                    DType::F32,
                )? * init_noise_sigma)?;
                (latents.to_dtype(loaded.dtype)?, 0, None)
            };

        // 5. Denoising loop
        let unet = loaded
            .unet
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("UNet not loaded"))?;
        self.denoise_loop(
            unet,
            &text_embeddings,
            sched,
            &mut latents,
            guidance,
            req.steps,
            start_step,
            inpaint_ctx.as_ref(),
        )?;

        // Drop UNet before VAE decode to free VRAM for conv2d intermediates.
        drop(inpaint_ctx);
        let _ = loaded;
        let loaded = self.base.loaded.as_mut().unwrap();
        loaded.unet = None;
        loaded.device.synchronize()?;
        tracing::info!("UNet dropped to free VRAM for VAE decode");
        let _ = loaded;
        let loaded = self.base.loaded.as_ref().unwrap();

        // 6. VAE decode
        self.base.progress.stage_start("VAE decode");
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

        self.base
            .progress
            .stage_done("VAE decode", vae_start.elapsed());

        // 8. Encode to image format
        let output_metadata = build_output_metadata(req, seed, Some(sched));
        let image_bytes = encode_image(
            &img,
            req.output_format,
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

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
            video: None,
        })
    }

    fn model_name(&self) -> &str {
        self.base.model_name()
    }

    fn is_loaded(&self) -> bool {
        // Sequential mode is always "ready" — it loads on demand
        self.base.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        SDXLEngine::load(self)
    }

    fn unload(&mut self) {
        self.base.unload();
        clear_cache(&self.prompt_cache);
        clear_cache(&self.source_latent_cache);
        clear_cache(&self.mask_cache);
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.base.set_on_progress(callback);
    }

    fn clear_on_progress(&mut self) {
        self.base.clear_on_progress();
    }

    fn model_paths(&self) -> Option<&mold_core::ModelPaths> {
        Some(&self.base.paths)
    }
}
