use anyhow::{bail, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_transformers::models::stable_diffusion;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::time::Instant;

use crate::engine::{rand_seed, InferenceEngine};
use crate::image::encode_image;
use crate::progress::{ProgressCallback, ProgressEvent};

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
    on_progress: Option<ProgressCallback>,
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
    ) -> Self {
        Self {
            loaded: None,
            model_name,
            paths,
            scheduler_name,
            is_turbo,
            on_progress: None,
        }
    }

    fn emit(&self, event: ProgressEvent) {
        if let Some(cb) = &self.on_progress {
            cb(event);
        }
    }

    fn stage_start(&self, name: &str) {
        self.emit(ProgressEvent::StageStart {
            name: name.to_string(),
        });
    }

    fn stage_done(&self, name: &str, elapsed: std::time::Duration) {
        self.emit(ProgressEvent::StageDone {
            name: name.to_string(),
            elapsed,
        });
    }

    fn info(&self, message: &str) {
        self.emit(ProgressEvent::Info {
            message: message.to_string(),
        });
    }

    /// Load all SDXL model components.
    pub fn load(&mut self) -> Result<()> {
        if self.loaded.is_some() {
            return Ok(());
        }

        // Validate SDXL-specific paths
        let clip_encoder_2 = self
            .paths
            .clip_encoder_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-G encoder path required for SDXL models"))?;
        let clip_tokenizer_2 = self
            .paths
            .clip_tokenizer_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-G tokenizer path required for SDXL models"))?;

        // Validate all paths exist
        for (label, path) in [
            ("transformer (UNet)", &self.paths.transformer),
            ("vae", &self.paths.vae),
            ("clip_encoder (CLIP-L)", &self.paths.clip_encoder),
            ("clip_tokenizer (CLIP-L)", &self.paths.clip_tokenizer),
            ("clip_encoder_2 (CLIP-G)", clip_encoder_2),
            ("clip_tokenizer_2 (CLIP-G)", clip_tokenizer_2),
        ] {
            if !path.exists() {
                bail!("{label} file not found: {}", path.display());
            }
        }

        tracing::info!(model = %self.model_name, "loading SDXL model components...");

        let device = if candle_core::utils::cuda_is_available() {
            self.info("CUDA detected, using GPU");
            Device::new_cuda(0)?
        } else if candle_core::utils::metal_is_available() {
            self.info("Metal detected, using GPU");
            Device::new_metal(0)?
        } else {
            self.info("No GPU detected, using CPU");
            Device::Cpu
        };

        let dtype = if device.is_cuda() || device.is_metal() {
            DType::F16
        } else {
            DType::F32
        };

        // Create SDXL config (sets up UNet, VAE, and CLIP configs)
        let sd_config = if self.is_turbo {
            stable_diffusion::StableDiffusionConfig::sdxl_turbo(None, None, None)
        } else {
            stable_diffusion::StableDiffusionConfig::sdxl(None, None, None)
        };

        // Load UNet
        self.stage_start("Loading UNet (GPU)");
        let unet_start = Instant::now();
        let unet = sd_config.build_unet(
            &self.paths.transformer,
            &device,
            4,     // in_channels
            false, // use_flash_attn
            dtype,
        )?;
        self.stage_done("Loading UNet (GPU)", unet_start.elapsed());

        // Load VAE
        self.stage_start("Loading VAE (GPU)");
        let vae_start = Instant::now();
        let vae = sd_config.build_vae(&self.paths.vae, &device, dtype)?;
        self.stage_done("Loading VAE (GPU)", vae_start.elapsed());

        // Load CLIP-L encoder
        self.stage_start("Loading CLIP-L encoder");
        let clip_l_start = Instant::now();
        let clip_l = stable_diffusion::build_clip_transformer(
            &sd_config.clip,
            &self.paths.clip_encoder,
            &device,
            DType::F32,
        )?;
        self.stage_done("Loading CLIP-L encoder", clip_l_start.elapsed());

        // Load CLIP-G encoder
        self.stage_start("Loading CLIP-G encoder");
        let clip_g_start = Instant::now();
        let clip2_config = sd_config
            .clip2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("SDXL config missing clip2 configuration"))?;
        let clip_g = stable_diffusion::build_clip_transformer(
            clip2_config,
            clip_encoder_2,
            &device,
            DType::F32,
        )?;
        self.stage_done("Loading CLIP-G encoder", clip_g_start.elapsed());

        // Load tokenizers
        let tokenizer_l = tokenizers::Tokenizer::from_file(&self.paths.clip_tokenizer)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP-L tokenizer: {e}"))?;
        let tokenizer_g = tokenizers::Tokenizer::from_file(clip_tokenizer_2)
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
}

impl InferenceEngine for SDXLEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
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
        let use_cfg = guidance > 1.0;

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

        self.stage_start("Encoding prompt (CLIP-L)");
        let encode_l_start = Instant::now();
        let tokens_l = Self::tokenize(&loaded.tokenizer_l, &req.prompt, max_len, &loaded.device)?;
        let text_emb_l = loaded.clip_l.forward(&tokens_l)?;
        self.stage_done("Encoding prompt (CLIP-L)", encode_l_start.elapsed());

        self.stage_start("Encoding prompt (CLIP-G)");
        let encode_g_start = Instant::now();
        let tokens_g = Self::tokenize(&loaded.tokenizer_g, &req.prompt, max_len, &loaded.device)?;
        let text_emb_g = loaded.clip_g.forward(&tokens_g)?;
        self.stage_done("Encoding prompt (CLIP-G)", encode_g_start.elapsed());

        // Concatenate CLIP-L (768) + CLIP-G (1280) → 2048-dim
        let text_embeddings = Tensor::cat(&[&text_emb_l, &text_emb_g], D::Minus1)?;

        // 2. If using classifier-free guidance, also encode empty prompt
        let text_embeddings = if use_cfg {
            let uncond_tokens_l = Self::tokenize(&loaded.tokenizer_l, "", max_len, &loaded.device)?;
            let uncond_emb_l = loaded.clip_l.forward(&uncond_tokens_l)?;
            let uncond_tokens_g = Self::tokenize(&loaded.tokenizer_g, "", max_len, &loaded.device)?;
            let uncond_emb_g = loaded.clip_g.forward(&uncond_tokens_g)?;
            let uncond_embeddings = Tensor::cat(&[&uncond_emb_l, &uncond_emb_g], D::Minus1)?;
            // [uncond, cond] → batch=2
            Tensor::cat(&[&uncond_embeddings, &text_embeddings], 0)?
        } else {
            text_embeddings
        };

        let text_embeddings = text_embeddings.to_dtype(loaded.dtype)?;

        // 3. Build scheduler
        let mut scheduler = loaded.sd_config.build_scheduler(req.steps as usize)?;
        let timesteps = scheduler.timesteps().to_vec();

        // 4. Generate initial noise
        let latent_h = height / 8;
        let latent_w = width / 8;
        let init_noise_sigma = scheduler.init_noise_sigma();
        let mut latents =
            (Tensor::randn(0f32, 1f32, &[1, 4, latent_h, latent_w], &loaded.device)?
                * init_noise_sigma)?;
        latents = latents.to_dtype(loaded.dtype)?;

        // 5. Denoising loop
        let denoise_label = format!("Denoising ({} steps)", timesteps.len());
        self.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        for &t in &timesteps {
            let latent_input = if use_cfg {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            let latent_input = scheduler.scale_model_input(latent_input, t)?;
            let noise_pred = loaded
                .unet
                .forward(&latent_input, t as f64, &text_embeddings)?;

            let noise_pred = if use_cfg {
                let chunks = noise_pred.chunk(2, 0)?;
                let noise_pred_uncond = &chunks[0];
                let noise_pred_cond = &chunks[1];
                // CFG: uncond + guidance * (cond - uncond)
                (noise_pred_uncond + ((noise_pred_cond - noise_pred_uncond)? * guidance)?)?
            } else {
                noise_pred
            };

            latents = scheduler.step(&noise_pred, t, &latents)?;
        }

        self.stage_done(&denoise_label, denoise_start.elapsed());

        // 6. VAE decode
        self.stage_start("VAE decode");
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

        self.stage_done("VAE decode", vae_start.elapsed());

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
        self.loaded.is_some()
    }

    fn load(&mut self) -> Result<()> {
        SDXLEngine::load(self)
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.on_progress = Some(callback);
    }
}
