use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::wuerstchen::ddpm::{DDPMWScheduler, DDPMWSchedulerConfig};
use candle_transformers::models::wuerstchen::diffnext::WDiffNeXt;
use candle_transformers::models::wuerstchen::paella_vq::PaellaVQ;
use candle_transformers::models::wuerstchen::prior::WPrior;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::sync::Mutex;
use std::time::Instant;

use crate::cache::{
    clear_cache, get_or_insert_cached_tensor_pair, prompt_text_key, CachedTensorPair, LruCache,
    DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{check_memory_budget, memory_status_string, preflight_memory_check};
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image, update_output_metadata_size};
use crate::progress::{ProgressCallback, ProgressEvent};

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
const DECODER_C_COND: usize = 1024;
const DECODER_CLIP_EMBD: usize = 1024;
const DECODER_PATCH_SIZE: usize = 2;

/// Latent compression ratio for Stage C (prior).
/// Wuerstchen operates in a 42x compressed latent space.
const LATENT_DIM_SCALE: f64 = 42.67;

/// Scale factor from Prior output spatial dims to Decoder latent dims.
const LATENT_DIM_SCALE_DECODER: f64 = 10.67;

/// Loaded Wuerstchen model components, ready for inference.
struct LoadedWuerstchen {
    prior: WPrior,
    decoder: WDiffNeXt,
    vqgan: PaellaVQ,
    prior_clip: stable_diffusion::clip::ClipTextTransformer,
    decoder_clip: stable_diffusion::clip::ClipTextTransformer,
    prior_tokenizer: tokenizers::Tokenizer,
    decoder_tokenizer: tokenizers::Tokenizer,
    device: Device,
    dtype: DType,
}

/// Wuerstchen v2 inference engine.
///
/// Three-stage cascade: CLIP-G encode -> Prior (Stage C) -> Decoder (Stage B) -> VQ-GAN (Stage A).
pub struct WuerstchenEngine {
    base: EngineBase<LoadedWuerstchen>,
    prompt_cache: Mutex<LruCache<String, CachedTensorPair>>,
}

impl WuerstchenEngine {
    pub fn new(model_name: String, paths: ModelPaths, load_strategy: LoadStrategy) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy),
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_prompt_pair_cached(
        &self,
        prior_clip: &stable_diffusion::clip::ClipTextTransformer,
        prior_tokenizer: &tokenizers::Tokenizer,
        decoder_clip: &stable_diffusion::clip::ClipTextTransformer,
        decoder_tokenizer: &tokenizers::Tokenizer,
        prompt: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let prior_clip_config = stable_diffusion::clip::Config::wuerstchen_prior();
        let dec_clip_config = stable_diffusion::clip::Config::wuerstchen();
        let cache_key = prompt_text_key(prompt);
        let ((prior_text_embeddings, decoder_text_embeddings), cache_hit) =
            get_or_insert_cached_tensor_pair(&self.prompt_cache, cache_key, device, dtype, || {
                self.base
                    .progress
                    .stage_start("Encoding prompt (Prior CLIP-G, 1280-dim)");
                let encode_start = Instant::now();
                let (prior_tokens, prior_tokens_len) = Self::tokenize(
                    prior_tokenizer,
                    prompt,
                    prior_clip_config.max_position_embeddings,
                    device,
                )?;
                let prior_text_embeddings = prior_clip
                    .forward_with_mask(&prior_tokens, prior_tokens_len - 1)?
                    .to_dtype(dtype)?;
                self.base.progress.stage_done(
                    "Encoding prompt (Prior CLIP-G, 1280-dim)",
                    encode_start.elapsed(),
                );

                self.base
                    .progress
                    .stage_start("Encoding prompt (Decoder CLIP, 1024-dim)");
                let dec_encode_start = Instant::now();
                let (dec_tokens, dec_tokens_len) = Self::tokenize(
                    decoder_tokenizer,
                    prompt,
                    dec_clip_config.max_position_embeddings,
                    device,
                )?;
                let decoder_text_embeddings = decoder_clip
                    .forward_with_mask(&dec_tokens, dec_tokens_len - 1)?
                    .to_dtype(dtype)?;
                self.base.progress.stage_done(
                    "Encoding prompt (Decoder CLIP, 1024-dim)",
                    dec_encode_start.elapsed(),
                );
                Ok((prior_text_embeddings, decoder_text_embeddings))
            })?;
        if cache_hit {
            self.base.progress.cache_hit("prompt conditioning");
        }
        Ok((prior_text_embeddings, decoder_text_embeddings))
    }

    /// Validate and return required Wuerstchen paths.
    /// Returns (decoder_path, prior_clip_encoder, prior_clip_tokenizer, decoder_clip_encoder, decoder_clip_tokenizer)
    fn validate_paths(
        &self,
    ) -> Result<(
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
    )> {
        let decoder = self
            .base
            .paths
            .decoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Decoder (Stage B) path required for Wuerstchen"))?
            .clone();
        // Prior CLIP-G (1280-dim) — stored in clip_encoder_2
        let prior_clip_encoder = self
            .base
            .paths
            .clip_encoder_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Prior CLIP-G encoder path required for Wuerstchen"))?
            .clone();
        let prior_clip_tokenizer = self
            .base
            .paths
            .clip_tokenizer_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Prior CLIP-G tokenizer path required for Wuerstchen"))?
            .clone();
        // Decoder CLIP (1024-dim) — stored in clip_encoder.
        // Fall back to Prior CLIP if decoder CLIP not available (old configs from
        // before the dual-CLIP change). Quality will be degraded but won't crash.
        let decoder_clip_encoder = self.base.paths.clip_encoder.clone().unwrap_or_else(|| {
            tracing::warn!(
                "Decoder CLIP encoder path not configured — falling back to Prior CLIP. \
                     Run `mold rm wuerstchen-v2:fp16 && mold pull wuerstchen-v2:fp16` to fix."
            );
            prior_clip_encoder.clone()
        });
        let decoder_clip_tokenizer = self
            .base
            .paths
            .clip_tokenizer
            .clone()
            .unwrap_or_else(|| prior_clip_tokenizer.clone());

        for (label, path) in [
            ("prior (Stage C)", &self.base.paths.transformer),
            ("decoder (Stage B)", &decoder),
            ("vqgan (Stage A)", &self.base.paths.vae),
            ("prior clip_encoder", &prior_clip_encoder),
            ("prior clip_tokenizer", &prior_clip_tokenizer),
            ("decoder clip_encoder", &decoder_clip_encoder),
            ("decoder clip_tokenizer", &decoder_clip_tokenizer),
        ] {
            if !path.exists() {
                bail!("{label} file not found: {}", path.display());
            }
        }

        Ok((
            decoder,
            prior_clip_encoder,
            prior_clip_tokenizer,
            decoder_clip_encoder,
            decoder_clip_tokenizer,
        ))
    }

    /// Load all Wuerstchen model components (Eager mode).
    pub fn load(&mut self) -> Result<()> {
        if self.base.loaded.is_some() {
            return Ok(());
        }

        if self.base.load_strategy == LoadStrategy::Sequential {
            return Ok(());
        }

        let (decoder_path, prior_clip_path, prior_clip_tok_path, dec_clip_path, dec_clip_tok_path) =
            self.validate_paths()?;

        tracing::info!(model = %self.base.model_name, "loading Wuerstchen model components...");

        let device = crate::device::create_device(&self.base.progress)?;
        // Wuerstchen's candle impl mixes dtypes internally (gen_r_embedding produces F32
        // that gets fed to F16 TimestepBlock weights). Use F32 for all backends to avoid
        // dtype mismatches. The model is small enough (~5.6GB) that F32 is fine.
        let dtype = DType::F32;

        // Load Prior (Stage C)
        self.base.progress.stage_start("Loading Prior (Stage C)");
        let prior_start = Instant::now();
        let prior_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[&self.base.paths.transformer],
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
        self.base
            .progress
            .stage_done("Loading Prior (Stage C)", prior_start.elapsed());

        // Load Decoder (Stage B)
        self.base.progress.stage_start("Loading Decoder (Stage B)");
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
        self.base
            .progress
            .stage_done("Loading Decoder (Stage B)", decoder_start.elapsed());

        // Load VQ-GAN (Stage A)
        self.base.progress.stage_start("Loading VQ-GAN (Stage A)");
        let vqgan_start = Instant::now();
        let vqgan_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&self.base.paths.vae], dtype, &device)?
        };
        let vqgan = PaellaVQ::new(vqgan_vb)?;
        self.base
            .progress
            .stage_done("Loading VQ-GAN (Stage A)", vqgan_start.elapsed());

        // Load Prior CLIP-G encoder (1280-dim, 32 layers)
        self.base
            .progress
            .stage_start("Loading Prior CLIP-G encoder (1280-dim)");
        let prior_clip_start = Instant::now();
        let prior_clip_config = stable_diffusion::clip::Config::wuerstchen_prior();
        let prior_clip = stable_diffusion::build_clip_transformer(
            &prior_clip_config,
            &prior_clip_path,
            &device,
            DType::F32,
        )?;
        self.base.progress.stage_done(
            "Loading Prior CLIP-G encoder (1280-dim)",
            prior_clip_start.elapsed(),
        );

        // Load Decoder CLIP encoder (1024-dim, 24 layers)
        self.base
            .progress
            .stage_start("Loading Decoder CLIP encoder (1024-dim)");
        let dec_clip_start = Instant::now();
        let dec_clip_config = stable_diffusion::clip::Config::wuerstchen();
        let decoder_clip = stable_diffusion::build_clip_transformer(
            &dec_clip_config,
            &dec_clip_path,
            &device,
            DType::F32,
        )?;
        self.base.progress.stage_done(
            "Loading Decoder CLIP encoder (1024-dim)",
            dec_clip_start.elapsed(),
        );

        // Load tokenizers
        let prior_tokenizer = tokenizers::Tokenizer::from_file(&prior_clip_tok_path)
            .map_err(|e| anyhow::anyhow!("failed to load Prior CLIP-G tokenizer: {e}"))?;
        let decoder_tokenizer = tokenizers::Tokenizer::from_file(&dec_clip_tok_path)
            .map_err(|e| anyhow::anyhow!("failed to load Decoder CLIP tokenizer: {e}"))?;

        self.base.loaded = Some(LoadedWuerstchen {
            prior,
            decoder,
            vqgan,
            prior_clip,
            decoder_clip,
            prior_tokenizer,
            decoder_tokenizer,
            device,
            dtype,
        });

        tracing::info!(model = %self.base.model_name, "all Wuerstchen components loaded successfully");
        Ok(())
    }

    /// Tokenize a prompt for a CLIP text encoder.
    /// Returns (tokens_tensor, tokens_len) where tokens_len is the number of
    /// real tokens before padding (used for forward_with_mask).
    fn tokenize(
        tokenizer: &tokenizers::Tokenizer,
        prompt: &str,
        max_len: usize,
        device: &Device,
    ) -> Result<(Tensor, usize)> {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(max_len);
        let tokens_len = ids.len();
        while ids.len() < max_len {
            ids.push(49407); // CLIP EOS/PAD token
        }
        let ids = ids.into_iter().map(|i| i as i64).collect::<Vec<_>>();
        Ok((Tensor::new(ids, device)?.unsqueeze(0)?, tokens_len))
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
        self.base.progress.stage_start(&label);
        let start = Instant::now();

        for (step_idx, &t) in timesteps.iter().enumerate() {
            if step_idx + 1 >= timesteps.len() {
                break; // last timestep is 0.0, not used for denoising
            }
            let step_start = Instant::now();

            let noise_pred = if use_cfg {
                // CFG: batch [latents, latents] with [text_embeddings, uncond]
                // text first (index 0), uncond second (index 1)
                let latent_input = Tensor::cat(&[&*latents, &*latents], 0)?;
                let r = (Tensor::ones(2, DType::F32, device)? * t)?;
                let uncond = Tensor::zeros_like(text_embeddings)?;
                let c_input = Tensor::cat(&[text_embeddings, &uncond], 0)?;
                let pred = prior.forward(&latent_input, &r, &c_input)?;
                let chunks = pred.chunk(2, 0)?;
                let (pred_text, pred_uncond) = (&chunks[0], &chunks[1]);
                (pred_uncond + ((pred_text - pred_uncond)? * guidance)?)?
            } else {
                let r = (Tensor::ones(1, DType::F32, device)? * t)?;
                prior.forward(&*latents, &r, text_embeddings)?
            };

            *latents = scheduler.step(&noise_pred, t, &*latents)?;

            self.base.progress.emit(ProgressEvent::DenoiseStep {
                step: step_idx + 1,
                total: timesteps.len() - 1,
                elapsed: step_start.elapsed(),
            });
        }

        self.base.progress.stage_done(&label, start.elapsed());
        Ok(())
    }

    /// Run the Stage B (Decoder) denoising loop.
    ///
    /// `image_embeddings` is the scaled Prior output (effnet slot in WDiffNeXt).
    /// `text_embeddings` is the 1024-dim Decoder CLIP output (clip slot in WDiffNeXt).
    /// No CFG is applied at the decoder stage, matching the upstream candle example.
    fn denoise_decoder(
        &self,
        decoder: &WDiffNeXt,
        image_embeddings: &Tensor,
        text_embeddings: &Tensor,
        latents: &mut Tensor,
        steps: usize,
        device: &Device,
    ) -> Result<()> {
        let scheduler = DDPMWScheduler::new(steps, DDPMWSchedulerConfig::default())?;
        let timesteps = scheduler.timesteps().to_vec();

        let label = format!("Stage B Decoder ({} steps)", timesteps.len() - 1);
        self.base.progress.stage_start(&label);
        let start = Instant::now();

        for (step_idx, &t) in timesteps.iter().enumerate() {
            if step_idx + 1 >= timesteps.len() {
                break;
            }
            let step_start = Instant::now();

            let r = (Tensor::ones(1, DType::F32, device)? * t)?;
            let noise_pred =
                decoder.forward(&*latents, &r, image_embeddings, Some(text_embeddings))?;

            *latents = scheduler.step(&noise_pred, t, &*latents)?;

            self.base.progress.emit(ProgressEvent::DenoiseStep {
                step: step_idx + 1,
                total: timesteps.len() - 1,
                elapsed: step_start.elapsed(),
            });
        }

        self.base.progress.stage_done(&label, start.elapsed());
        Ok(())
    }

    /// Generate an image using sequential loading strategy.
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let (decoder_path, prior_clip_path, prior_clip_tok_path, dec_clip_path, dec_clip_tok_path) =
            self.validate_paths()?;

        if let Some(warning) = check_memory_budget(&self.base.paths, LoadStrategy::Sequential) {
            self.base.progress.info(&warning);
        }

        let device = crate::device::create_device(&self.base.progress)?;
        // Wuerstchen's candle impl mixes dtypes internally (gen_r_embedding produces F32
        // that gets fed to F16 TimestepBlock weights). Use F32 for all backends to avoid
        // dtype mismatches. The model is small enough (~5.6GB) that F32 is fine.
        let dtype = DType::F32;

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

        self.base
            .progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: Prior CLIP-G encode (1280-dim) ---
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        let prior_tokenizer = tokenizers::Tokenizer::from_file(&prior_clip_tok_path)
            .map_err(|e| anyhow::anyhow!("failed to load Prior CLIP-G tokenizer: {e}"))?;

        self.base
            .progress
            .stage_start("Loading Prior CLIP-G encoder (1280-dim)");
        let clip_start = Instant::now();
        let prior_clip_config = stable_diffusion::clip::Config::wuerstchen_prior();
        let prior_clip = stable_diffusion::build_clip_transformer(
            &prior_clip_config,
            &prior_clip_path,
            &device,
            DType::F32,
        )?;
        self.base.progress.stage_done(
            "Loading Prior CLIP-G encoder (1280-dim)",
            clip_start.elapsed(),
        );

        let decoder_tokenizer = tokenizers::Tokenizer::from_file(&dec_clip_tok_path)
            .map_err(|e| anyhow::anyhow!("failed to load Decoder CLIP tokenizer: {e}"))?;

        self.base
            .progress
            .stage_start("Loading Decoder CLIP encoder (1024-dim)");
        let dec_clip_start = Instant::now();
        let dec_clip_config = stable_diffusion::clip::Config::wuerstchen();
        let decoder_clip = stable_diffusion::build_clip_transformer(
            &dec_clip_config,
            &dec_clip_path,
            &device,
            DType::F32,
        )?;
        self.base.progress.stage_done(
            "Loading Decoder CLIP encoder (1024-dim)",
            dec_clip_start.elapsed(),
        );

        let (prior_text_embeddings, decoder_text_embeddings) = self.encode_prompt_pair_cached(
            &prior_clip,
            &prior_tokenizer,
            &decoder_clip,
            &decoder_tokenizer,
            &req.prompt,
            &device,
            dtype,
        )?;

        drop(prior_clip);
        drop(prior_tokenizer);
        self.base.progress.info("Freed Prior CLIP-G encoder");
        tracing::info!("Prior CLIP-G encoder dropped (sequential mode)");

        drop(decoder_clip);
        drop(decoder_tokenizer);
        self.base.progress.info("Freed Decoder CLIP encoder");
        tracing::info!("Decoder CLIP encoder dropped (sequential mode)");

        // --- Phase 2: Prior (Stage C) ---
        let prior_size = std::fs::metadata(&self.base.paths.transformer)
            .map(|m| m.len())
            .unwrap_or(0);
        preflight_memory_check("Prior (Stage C)", prior_size)?;
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        self.base.progress.stage_start("Loading Prior (Stage C)");
        let prior_start = Instant::now();
        let prior_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[&self.base.paths.transformer],
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
        self.base
            .progress
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
            &prior_text_embeddings,
            &mut prior_latents,
            prior_steps,
            guidance,
            &device,
        )?;

        // Scale prior output: convert from Prior latent space to Decoder conditioning space
        prior_latents = ((prior_latents * 42.)? - 1.)?;

        drop(prior);
        drop(prior_text_embeddings);
        device.synchronize()?;
        self.base.progress.info("Freed Prior (Stage C)");

        // --- Phase 3: Decoder (Stage B) ---
        // 3b. Load Decoder (Stage B) model and denoise
        let decoder_size = std::fs::metadata(&decoder_path)
            .map(|m| m.len())
            .unwrap_or(0);
        preflight_memory_check("Decoder (Stage B)", decoder_size)?;
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        self.base.progress.stage_start("Loading Decoder (Stage B)");
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
        self.base
            .progress
            .stage_done("Loading Decoder (Stage B)", dec_start.elapsed());

        // Decoder latent dims derived from prior output spatial dims
        let stage_b_h = (prior_latents.dim(2)? as f64 * LATENT_DIM_SCALE_DECODER) as usize;
        let stage_b_w = (prior_latents.dim(3)? as f64 * LATENT_DIM_SCALE_DECODER) as usize;
        let mut decoder_latents =
            crate::engine::seeded_randn(seed + 1, &[1, 4, stage_b_h, stage_b_w], &device, dtype)?;

        self.denoise_decoder(
            &decoder,
            &prior_latents,
            &decoder_text_embeddings,
            &mut decoder_latents,
            decoder_steps,
            &device,
        )?;

        drop(decoder);
        drop(prior_latents);
        drop(decoder_text_embeddings);
        device.synchronize()?;
        self.base.progress.info("Freed Decoder (Stage B)");

        // --- Phase 4: VQ-GAN decode (Stage A) ---
        self.base.progress.stage_start("Loading VQ-GAN (Stage A)");
        let vqgan_start = Instant::now();
        let vqgan_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&self.base.paths.vae], dtype, &device)?
        };
        let vqgan = PaellaVQ::new(vqgan_vb)?;
        self.base
            .progress
            .stage_done("Loading VQ-GAN (Stage A)", vqgan_start.elapsed());

        self.base.progress.stage_start("VQ-GAN decode");
        let decode_start = Instant::now();
        let img = vqgan.decode(&(&decoder_latents * 0.3764)?)?;
        let img = img.clamp(0f32, 1f32)?;
        let img = (img * 255.)?.to_dtype(DType::U8)?;
        let img = img.squeeze(0)?;
        self.base
            .progress
            .stage_done("VQ-GAN decode", decode_start.elapsed());

        // Use actual tensor dims — VQ-GAN output may differ from requested dims
        // due to the 42x compression rounding in the cascade.
        let (_, actual_h, actual_w) = img.dims3()?;
        let mut output_metadata = build_output_metadata(req, seed, None);
        update_output_metadata_size(&mut output_metadata, actual_w as u32, actual_h as u32);
        let image_bytes = encode_image(
            &img,
            req.output_format,
            actual_w as u32,
            actual_h as u32,
            output_metadata.as_ref(),
        )?;

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
            tracing::warn!("inpainting not yet supported for Wuerstchen -- ignoring mask");
        }

        if self.base.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

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

        // 1. Encode prompt with Prior CLIP-G (1280-dim)
        let (prior_text_embeddings, decoder_text_embeddings) = self.encode_prompt_pair_cached(
            &loaded.prior_clip,
            &loaded.prior_tokenizer,
            &loaded.decoder_clip,
            &loaded.decoder_tokenizer,
            &req.prompt,
            &loaded.device,
            loaded.dtype,
        )?;

        // 3. Stage C (Prior): denoise in highly compressed latent space
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
            &prior_text_embeddings,
            &mut prior_latents,
            prior_steps,
            guidance,
            &loaded.device,
        )?;

        // Scale prior output: convert from Prior latent space to Decoder conditioning space
        prior_latents = ((prior_latents * 42.)? - 1.)?;

        // 4. Stage B (Decoder): decode prior latents to VQ-GAN latent space
        // Decoder latent dims derived from prior output spatial dims
        let stage_b_h = (prior_latents.dim(2)? as f64 * LATENT_DIM_SCALE_DECODER) as usize;
        let stage_b_w = (prior_latents.dim(3)? as f64 * LATENT_DIM_SCALE_DECODER) as usize;
        let mut decoder_latents = crate::engine::seeded_randn(
            seed + 1,
            &[1, 4, stage_b_h, stage_b_w],
            &loaded.device,
            loaded.dtype,
        )?;

        self.denoise_decoder(
            &loaded.decoder,
            &prior_latents,
            &decoder_text_embeddings,
            &mut decoder_latents,
            decoder_steps,
            &loaded.device,
        )?;

        // 5. Stage A (VQ-GAN): decode to pixel space
        self.base.progress.stage_start("VQ-GAN decode");
        let decode_start = Instant::now();
        let img = loaded.vqgan.decode(&(&decoder_latents * 0.3764)?)?;
        let img = img.clamp(0f32, 1f32)?;
        let img = (img * 255.)?.to_dtype(DType::U8)?;
        let img = img.squeeze(0)?;
        self.base
            .progress
            .stage_done("VQ-GAN decode", decode_start.elapsed());

        // 6. Encode to image format
        // Use actual tensor dims — VQ-GAN output may differ from requested dims
        // due to the 42x compression rounding in the cascade.
        let (_, actual_h, actual_w) = img.dims3()?;
        let mut output_metadata = build_output_metadata(req, seed, None);
        update_output_metadata_size(&mut output_metadata, actual_w as u32, actual_h as u32);
        let image_bytes = encode_image(
            &img,
            req.output_format,
            actual_w as u32,
            actual_h as u32,
            output_metadata.as_ref(),
        )?;

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
        self.base.model_name()
    }

    fn is_loaded(&self) -> bool {
        self.base.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        WuerstchenEngine::load(self)
    }

    fn unload(&mut self) {
        self.base.unload();
        clear_cache(&self.prompt_cache);
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.base.set_on_progress(callback);
    }

    fn clear_on_progress(&mut self) {
        self.base.clear_on_progress();
    }
}
