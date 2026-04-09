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
    clear_cache, get_or_insert_cached_tensor_pair, restore_cached_tensor_pair, CachedTensorPair,
    LruCache, DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{check_memory_budget, memory_status_string, preflight_memory_check};
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image, update_output_metadata_size};
use crate::img_utils;
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

/// VQ-GAN output scaling factor: decoder latents are multiplied by this before VQ-GAN decode.
/// For img2img, VQ-GAN encode output is divided by this to get decoder latent space.
const VQGAN_SCALE: f64 = 0.3764;

/// Loaded Wuerstchen model components, ready for inference.
struct LoadedWuerstchen {
    /// None after being dropped for VQ-GAN decode VRAM; reloaded on next generate.
    prior: Option<WPrior>,
    /// None after being dropped for VQ-GAN decode VRAM; reloaded on next generate.
    decoder: Option<WDiffNeXt>,
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
    fn debug_tensor_stats(name: &str, tensor: &Tensor) {
        if std::env::var_os("MOLD_WUERSTCHEN_DEBUG").is_none() {
            return;
        }

        let stats = || -> Result<String> {
            let dims = tensor.dims().to_vec();
            let dtype = tensor.dtype();
            let flat = tensor
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;

            if flat.is_empty() {
                return Ok(format!(
                    "[wuerstchen-debug] {name}: shape={dims:?} dtype={dtype:?} <empty>"
                ));
            }

            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            let mut nan_count = 0usize;
            let mut inf_count = 0usize;
            let mut finite_count = 0usize;

            for &value in &flat {
                if value.is_nan() {
                    nan_count += 1;
                    continue;
                }
                if value.is_infinite() {
                    inf_count += 1;
                    continue;
                }
                min = min.min(value);
                max = max.max(value);
                let value = value as f64;
                sum += value;
                sum_sq += value * value;
                finite_count += 1;
            }

            let (mean, std) = if finite_count > 0 {
                let mean = sum / finite_count as f64;
                let variance = (sum_sq / finite_count as f64) - (mean * mean);
                (mean, variance.max(0.0).sqrt())
            } else {
                (f64::NAN, f64::NAN)
            };
            let checksum16: f64 = flat.iter().take(16).map(|&v| v as f64).sum();

            Ok(format!(
                "[wuerstchen-debug] {name}: shape={dims:?} dtype={dtype:?} min={min:.4} max={max:.4} mean={mean:.4} std={std:.4} nan={nan_count} inf={inf_count} checksum16={checksum16:.4}"
            ))
        };

        match stats() {
            Ok(message) => eprintln!("{message}"),
            Err(err) => eprintln!("[wuerstchen-debug] {name}: <failed: {err}>"),
        }
    }

    fn prompt_cache_key(
        prompt: &str,
        negative_prompt: &str,
        use_prior_cfg: bool,
        use_decoder_cfg: bool,
    ) -> String {
        format!(
            "{prompt}\u{1f}{negative_prompt}\u{1f}prior_cfg={use_prior_cfg}\u{1f}decoder_cfg={use_decoder_cfg}"
        )
    }

    fn pad_token_id(
        tokenizer: &tokenizers::Tokenizer,
        clip_config: &stable_diffusion::clip::Config,
    ) -> Result<u32> {
        let vocab = tokenizer.get_vocab(true);
        let token = clip_config.pad_with.as_deref().unwrap_or("<|endoftext|>");
        vocab
            .get(token)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("tokenizer missing pad/eos token '{token}'"))
    }

    fn encode_clip_prompt(
        clip: &stable_diffusion::clip::ClipTextTransformer,
        tokenizer: &tokenizers::Tokenizer,
        clip_config: &stable_diffusion::clip::Config,
        prompt: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let (tokens, tokens_len) = Self::tokenize(
            tokenizer,
            prompt,
            clip_config.max_position_embeddings,
            clip_config,
            device,
        )?;
        Ok(clip
            .forward_with_mask(&tokens, tokens_len - 1)?
            .to_dtype(dtype)?)
    }

    fn decoder_guidance() -> f64 {
        std::env::var("MOLD_WUERSTCHEN_DECODER_GUIDANCE")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(0.0)
    }

    fn effective_prior_steps(requested_steps: usize) -> usize {
        if requested_steps < 10 {
            // Very low step counts produce noise; warn but respect the request.
            tracing::warn!(
                steps = requested_steps,
                "Wuerstchen prior works best with ≥20 steps"
            );
        }
        requested_steps
    }

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
        negative_prompt: &str,
        device: &Device,
        dtype: DType,
        prior_guidance: f64,
        decoder_guidance: f64,
    ) -> Result<(Tensor, Tensor)> {
        let use_prior_cfg = prior_guidance > 1.0;
        let use_decoder_cfg = decoder_guidance > 1.0;
        let prior_clip_config = stable_diffusion::clip::Config::wuerstchen_prior();
        let dec_clip_config = stable_diffusion::clip::Config::wuerstchen();
        let cache_key =
            Self::prompt_cache_key(prompt, negative_prompt, use_prior_cfg, use_decoder_cfg);
        let ((prior_text_embeddings, decoder_text_embeddings), cache_hit) =
            get_or_insert_cached_tensor_pair(&self.prompt_cache, cache_key, device, dtype, || {
                self.base
                    .progress
                    .stage_start("Encoding prompt (Prior CLIP-G, 1280-dim)");
                let encode_start = Instant::now();
                let prior_text_embeddings = Self::encode_clip_prompt(
                    prior_clip,
                    prior_tokenizer,
                    &prior_clip_config,
                    prompt,
                    device,
                    dtype,
                )?;
                let prior_text_embeddings = if use_prior_cfg {
                    let prior_negative_embeddings = Self::encode_clip_prompt(
                        prior_clip,
                        prior_tokenizer,
                        &prior_clip_config,
                        negative_prompt,
                        device,
                        dtype,
                    )?;
                    Tensor::cat(&[&prior_text_embeddings, &prior_negative_embeddings], 0)?
                } else {
                    prior_text_embeddings
                };
                self.base.progress.stage_done(
                    "Encoding prompt (Prior CLIP-G, 1280-dim)",
                    encode_start.elapsed(),
                );

                self.base
                    .progress
                    .stage_start("Encoding prompt (Decoder CLIP, 1024-dim)");
                let dec_encode_start = Instant::now();
                let decoder_text_embeddings = Self::encode_clip_prompt(
                    decoder_clip,
                    decoder_tokenizer,
                    &dec_clip_config,
                    prompt,
                    device,
                    dtype,
                )?;
                let decoder_text_embeddings = if use_decoder_cfg {
                    let decoder_negative_embeddings = Self::encode_clip_prompt(
                        decoder_clip,
                        decoder_tokenizer,
                        &dec_clip_config,
                        negative_prompt,
                        device,
                        dtype,
                    )?;
                    Tensor::cat(&[&decoder_text_embeddings, &decoder_negative_embeddings], 0)?
                } else {
                    decoder_text_embeddings
                };
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

    /// Reload Prior and Decoder if they were dropped after VQ-GAN decode.
    fn reload_models_if_needed(&mut self) -> Result<()> {
        let needs_reload = self
            .base
            .loaded
            .as_ref()
            .map(|l| l.prior.is_none() || l.decoder.is_none())
            .unwrap_or(false);

        if needs_reload {
            let (decoder_path, _, _, _, _) = self.validate_paths()?;
            let loaded = self.base.loaded.as_ref().unwrap();
            let device = loaded.device.clone();
            let dtype = loaded.dtype;
            let _ = loaded;

            self.base.progress.stage_start("Reloading Prior (Stage C)");
            let reload_start = Instant::now();
            let prior_vb = crate::weight_loader::load_safetensors_with_progress(
                &[&self.base.paths.transformer],
                dtype,
                &device,
                "Wuerstchen Prior",
                &self.base.progress,
            )?;
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
                .stage_done("Reloading Prior (Stage C)", reload_start.elapsed());

            self.base
                .progress
                .stage_start("Reloading Decoder (Stage B)");
            let reload_start = Instant::now();
            let decoder_vb = crate::weight_loader::load_safetensors_with_progress(
                &[&decoder_path],
                DType::F32,
                &device,
                "Wuerstchen Decoder",
                &self.base.progress,
            )?;
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
                .stage_done("Reloading Decoder (Stage B)", reload_start.elapsed());

            let loaded = self.base.loaded.as_mut().unwrap();
            loaded.prior = Some(prior);
            loaded.decoder = Some(decoder);
        }
        Ok(())
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
        // Use F16 on GPU for ~2x throughput and ~2x less VRAM.
        // gen_r_embedding computes sincos basis in F32 internally, then casts to
        // model dtype before the matmul — patched in candle-transformers-mold 0.9.4.
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::F16
        };

        // Load Prior (Stage C)
        self.base.progress.stage_start("Loading Prior (Stage C)");
        let prior_start = Instant::now();
        let prior_vb = crate::weight_loader::load_safetensors_with_progress(
            &[&self.base.paths.transformer],
            dtype,
            &device,
            "Wuerstchen Prior",
            &self.base.progress,
        )?;
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

        // Load Decoder (Stage B) — F32 because the 256x256 latent space
        // overflows F16 range during denoising (image_embeddings ±200).
        self.base.progress.stage_start("Loading Decoder (Stage B)");
        let decoder_start = Instant::now();
        let decoder_vb = crate::weight_loader::load_safetensors_with_progress(
            &[&decoder_path],
            DType::F32,
            &device,
            "Wuerstchen Decoder",
            &self.base.progress,
        )?;
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

        // Load VQ-GAN (Stage A) — always F32 for pixel-space decoding
        self.base.progress.stage_start("Loading VQ-GAN (Stage A)");
        let vqgan_start = Instant::now();
        let vqgan_vb = crate::weight_loader::load_safetensors_with_progress(
            &[&self.base.paths.vae],
            DType::F32,
            &device,
            "VQ-GAN",
            &self.base.progress,
        )?;
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
            prior: Some(prior),
            decoder: Some(decoder),
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
        clip_config: &stable_diffusion::clip::Config,
        device: &Device,
    ) -> Result<(Tensor, usize)> {
        let pad_id = Self::pad_token_id(tokenizer, clip_config)?;
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(max_len);
        if ids.is_empty() {
            ids.push(pad_id);
        }
        let tokens_len = ids.len();
        while ids.len() < max_len {
            ids.push(pad_id);
        }
        Ok((
            Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?,
            tokens_len,
        ))
    }

    /// Run the Stage C (Prior) denoising loop.
    #[allow(clippy::too_many_arguments)]
    fn denoise_prior(
        &self,
        prior: &WPrior,
        text_embeddings: &Tensor,
        latents: &mut Tensor,
        // TODO: use for per-step RNG reseeding to close RMSE gap vs candle reference
        _base_seed: u64,
        steps: usize,
        guidance: f64,
        device: &Device,
        dtype: DType,
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
                // CFG: batch [latents, latents] with [text_embeddings, negative_prompt]
                // text first (index 0), negative/unconditional second (index 1)
                let latent_input = Tensor::cat(&[&*latents, &*latents], 0)?;
                let r = (Tensor::ones(2, dtype, device)? * t)?;
                let pred = prior.forward(&latent_input, &r, text_embeddings)?;
                let chunks = pred.chunk(2, 0)?;
                let (pred_text, pred_uncond) = (&chunks[0], &chunks[1]);
                (pred_uncond + ((pred_text - pred_uncond)? * guidance)?)?
            } else {
                let r = (Tensor::ones(1, dtype, device)? * t)?;
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
    /// Applies decoder CFG using Diffusers-style conditioning when guidance > 1.0.
    ///
    /// `start_step` allows starting from a later timestep for img2img (0 = full txt2img).
    /// `inpaint_ctx` blends preserved regions back after each step for inpainting.
    #[allow(clippy::too_many_arguments)]
    fn denoise_decoder(
        &self,
        decoder: &WDiffNeXt,
        image_embeddings: &Tensor,
        text_embeddings: &Tensor,
        latents: &mut Tensor,
        // TODO: use for per-step RNG reseeding to close RMSE gap vs candle reference
        _base_seed: u64,
        steps: usize,
        start_step: usize,
        guidance: f64,
        inpaint_ctx: Option<&img_utils::InpaintContext>,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        let use_cfg = guidance > 1.0;
        let scheduler = DDPMWScheduler::new(steps, DDPMWSchedulerConfig::default())?;
        let timesteps = scheduler.timesteps().to_vec();
        // Drop the final 0.0 timestep (not used for denoising), then skip start_step
        let active_timesteps = &timesteps[start_step..timesteps.len() - 1];

        let label = format!("Stage B Decoder ({} steps)", active_timesteps.len());
        self.base.progress.stage_start(&label);
        let start = Instant::now();

        for (step_idx, &t) in active_timesteps.iter().enumerate() {
            let step_start = Instant::now();

            let noise_pred = if use_cfg {
                let latent_input = Tensor::cat(&[&*latents, &*latents], 0)?;
                let r = (Tensor::ones(2, dtype, device)? * t)?;
                let effnet_input = Tensor::cat(
                    &[image_embeddings, &Tensor::zeros_like(image_embeddings)?],
                    0,
                )?;
                let pred =
                    decoder.forward(&latent_input, &r, &effnet_input, Some(text_embeddings))?;
                let chunks = pred.chunk(2, 0)?;
                let (pred_text, pred_uncond) = (&chunks[0], &chunks[1]);
                (pred_uncond + ((pred_text - pred_uncond)? * guidance)?)?
            } else {
                let r = (Tensor::ones(1, dtype, device)? * t)?;
                decoder.forward(&*latents, &r, image_embeddings, Some(text_embeddings))?
            };

            *latents = scheduler.step(&noise_pred, t, &*latents)?;

            // Inpainting: blend preserved regions back at current noise level
            if let Some(ctx) = inpaint_ctx {
                let noised_original = Self::ddpmw_add_noise(&ctx.original_latents, &ctx.noise, t)?;
                *latents = crate::img2img::blend_inpaint_latents(&*latents, ctx, &noised_original)?;
            }

            self.base.progress.emit(ProgressEvent::DenoiseStep {
                step: step_idx + 1,
                total: active_timesteps.len(),
                elapsed: step_start.elapsed(),
            });
        }

        self.base.progress.stage_done(&label, start.elapsed());
        Ok(())
    }

    /// DDPM noise addition for Wuerstchen's continuous timesteps.
    ///
    /// DDPMWScheduler doesn't expose `add_noise()`, so we implement the standard
    /// DDPM forward process: `noised = sqrt(alpha_cumprod) * original + sqrt(1 - alpha_cumprod) * noise`
    /// using the same cosine schedule as DDPMWScheduler.
    fn ddpmw_add_noise(original: &Tensor, noise: &Tensor, t: f64) -> Result<Tensor> {
        // Replicate DDPMWScheduler::alpha_cumprod with default config (scaler=1.0, s=0.008)
        let s = 0.008f64;
        let init_alpha_cumprod = (s / (1.0 + s) * std::f64::consts::PI).cos().powi(2);
        let alpha_cumprod = ((t + s) / (1.0 + s) * std::f64::consts::PI * 0.5)
            .cos()
            .powi(2)
            / init_alpha_cumprod;
        let alpha_cumprod = alpha_cumprod.clamp(0.0001, 0.9999);

        let sqrt_alpha = alpha_cumprod.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha_cumprod).sqrt();

        let noised = ((original * sqrt_alpha)? + (noise * sqrt_one_minus_alpha)?)?;
        Ok(noised)
    }

    /// Prepare img2img latents: VQ-GAN encode source image, add noise at the start timestep.
    /// Returns (noised_latents, start_step, encoded_latents, noise).
    #[allow(clippy::too_many_arguments)]
    fn prepare_img2img_latents(
        &self,
        vqgan: &PaellaVQ,
        source_bytes: &[u8],
        width: u32,
        height: u32,
        strength: f64,
        decoder_steps: usize,
        seed: u64,
        device: &Device,
    ) -> Result<(Tensor, usize, Tensor, Tensor)> {
        self.base
            .progress
            .stage_start("Encoding source image (VQ-GAN)");
        let encode_start = Instant::now();

        // VQ-GAN expects [0, 1] normalized input in F32
        let source_tensor = img_utils::decode_source_image(
            source_bytes,
            width,
            height,
            img_utils::NormalizeRange::ZeroToOne,
            device,
            DType::F32,
        )?;

        let encoded = vqgan.encode(&source_tensor)?;
        // Scale from VQ-GAN latent space to decoder latent space (inverse of decode scaling)
        let encoded = (&encoded / VQGAN_SCALE)?;

        self.base
            .progress
            .stage_done("Encoding source image (VQ-GAN)", encode_start.elapsed());

        let start_step = crate::img2img::img2img_start_index(decoder_steps, strength);

        // Generate deterministic noise matching decoder latent shape
        let noise = crate::engine::seeded_randn(seed, encoded.dims(), device, DType::F32)?;

        // Build scheduler to get timesteps for noise addition
        let scheduler = DDPMWScheduler::new(decoder_steps, DDPMWSchedulerConfig::default())?;
        let timesteps = scheduler.timesteps().to_vec();

        // Add noise at the start timestep
        let noised = if start_step < timesteps.len() - 1 {
            Self::ddpmw_add_noise(&encoded, &noise, timesteps[start_step])?
        } else {
            encoded.clone()
        };

        tracing::info!(
            start_step,
            total_steps = decoder_steps,
            strength,
            "img2img: starting decoder from step {start_step}"
        );

        Ok((noised, start_step, encoded, noise))
    }

    /// Generate an image using sequential loading strategy.
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let (decoder_path, prior_clip_path, prior_clip_tok_path, dec_clip_path, dec_clip_tok_path) =
            self.validate_paths()?;

        if let Some(warning) = check_memory_budget(&self.base.paths, LoadStrategy::Sequential) {
            self.base.progress.info(&warning);
        }

        let device = crate::device::create_device(&self.base.progress)?;
        // Use F16 on GPU for ~2x throughput on the Prior stage.
        // Decoder and VQ-GAN use F32 explicitly (see their load calls below).
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::F16
        };

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);
        let width = req.width as usize;
        let height = req.height as usize;
        let prior_guidance = req.guidance;
        let decoder_guidance = Self::decoder_guidance();
        let negative_prompt = req.negative_prompt.as_deref().unwrap_or("");
        let prior_steps = Self::effective_prior_steps(req.steps as usize);
        let decoder_steps = 12;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            prior_steps,
            decoder_steps,
            prior_guidance,
            decoder_guidance,
            "starting sequential Wuerstchen generation"
        );

        self.base
            .progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: Encode prompt (check cache first to skip encoder load) ---
        let use_prior_cfg = prior_guidance > 1.0;
        let use_decoder_cfg = decoder_guidance > 1.0;
        let cache_key =
            Self::prompt_cache_key(&req.prompt, negative_prompt, use_prior_cfg, use_decoder_cfg);
        let (prior_text_embeddings, decoder_text_embeddings) =
            if let Some((prior_emb, decoder_emb)) =
                restore_cached_tensor_pair(&self.prompt_cache, &cache_key, &device, dtype)?
            {
                self.base.progress.cache_hit("prompt conditioning");
                (prior_emb, decoder_emb)
            } else {
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

                let (prior_emb, decoder_emb) = self.encode_prompt_pair_cached(
                    &prior_clip,
                    &prior_tokenizer,
                    &decoder_clip,
                    &decoder_tokenizer,
                    &req.prompt,
                    negative_prompt,
                    &device,
                    dtype,
                    prior_guidance,
                    decoder_guidance,
                )?;

                drop(prior_clip);
                drop(prior_tokenizer);
                self.base.progress.info("Freed Prior CLIP-G encoder");

                (prior_emb, decoder_emb)
            };
        Self::debug_tensor_stats("prior_text_embeddings", &prior_text_embeddings);
        Self::debug_tensor_stats("decoder_text_embeddings", &decoder_text_embeddings);
        tracing::info!("CLIP encoders processed (sequential mode)");

        let is_img2img = req.source_image.is_some();

        // --- Phase 2: img2img path (VQ-GAN encode, skip Prior) or txt2img path (Prior) ---
        let (image_embeddings, mut decoder_latents, decoder_start_step, inpaint_ctx) =
            if let Some(ref source_bytes) = req.source_image {
                self.base
                    .progress
                    .info("img2img mode: skipping Prior, encoding source via VQ-GAN");

                // Load VQ-GAN for encoding
                self.base.progress.stage_start("Loading VQ-GAN (Stage A)");
                let vqgan_start = Instant::now();
                let vqgan_vb = crate::weight_loader::load_safetensors_with_progress(
                    &[&self.base.paths.vae],
                    DType::F32,
                    &device,
                    "VQ-GAN",
                    &self.base.progress,
                )?;
                let vqgan = PaellaVQ::new(vqgan_vb)?;
                self.base
                    .progress
                    .stage_done("Loading VQ-GAN (Stage A)", vqgan_start.elapsed());

                let (noised, start_step, encoded, noise) = self.prepare_img2img_latents(
                    &vqgan,
                    source_bytes,
                    req.width,
                    req.height,
                    req.strength,
                    decoder_steps,
                    seed,
                    &device,
                )?;

                let (_, _, enc_h, enc_w) = encoded.dims4()?;
                let inpaint_ctx = crate::img2img::maybe_build_inpaint_context(
                    req.mask_image.as_deref(),
                    &encoded,
                    &noise,
                    enc_h,
                    enc_w,
                    &device,
                    DType::F32,
                )?;

                // Use zeros for effnet conditioning (no Prior output)
                // The Decoder will rely on text conditioning + noised latents
                let (_, _, _enc_h, _enc_w) = noised.dims4()?;
                let prior_latent_h = (height as f64 / LATENT_DIM_SCALE).ceil() as usize;
                let prior_latent_w = (width as f64 / LATENT_DIM_SCALE).ceil() as usize;
                let image_embeddings = Tensor::zeros(
                    (1, PRIOR_C_IN, prior_latent_h, prior_latent_w),
                    DType::F32,
                    &device,
                )?;

                drop(vqgan);
                device.synchronize()?;
                self.base
                    .progress
                    .info("Freed VQ-GAN (will reload for decode)");

                Self::debug_tensor_stats("decoder_latents_init", &noised);
                (image_embeddings, noised, start_step, inpaint_ctx)
            } else {
                // --- txt2img: run Prior (Stage C) ---
                let prior_size = std::fs::metadata(&self.base.paths.transformer)
                    .map(|m| m.len())
                    .unwrap_or(0);
                preflight_memory_check("Prior (Stage C)", prior_size)?;
                if let Some(status) = memory_status_string() {
                    self.base.progress.info(&status);
                }

                self.base.progress.stage_start("Loading Prior (Stage C)");
                let prior_start = Instant::now();
                let prior_vb = crate::weight_loader::load_safetensors_with_progress(
                    &[&self.base.paths.transformer],
                    dtype,
                    &device,
                    "Wuerstchen Prior",
                    &self.base.progress,
                )?;
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
                device.set_seed(seed)?;
                let mut prior_latents =
                    Tensor::randn(0f32, 1f32, (1, PRIOR_C_IN, latent_h, latent_w), &device)?
                        .to_dtype(dtype)?;
                Self::debug_tensor_stats("prior_latents_init", &prior_latents);

                self.denoise_prior(
                    &prior,
                    &prior_text_embeddings,
                    &mut prior_latents,
                    seed,
                    prior_steps,
                    prior_guidance,
                    &device,
                    dtype,
                )?;

                // Scale prior output: convert from Prior latent space to Decoder conditioning space
                Self::debug_tensor_stats("prior_latents_denoised", &prior_latents);
                prior_latents = ((prior_latents * 42.)? - 1.)?;
                Self::debug_tensor_stats("image_embeddings", &prior_latents);

                drop(prior);
                device.synchronize()?;
                self.base.progress.info("Freed Prior (Stage C)");

                // Decoder latent dims derived from prior output spatial dims
                let prior_latents = prior_latents.to_dtype(DType::F32)?;
                let stage_b_h = (prior_latents.dim(2)? as f64 * LATENT_DIM_SCALE_DECODER) as usize;
                let stage_b_w = (prior_latents.dim(3)? as f64 * LATENT_DIM_SCALE_DECODER) as usize;
                device.set_seed(seed.wrapping_add(1))?;
                let decoder_latents =
                    Tensor::randn(0f32, 1f32, (1, 4, stage_b_h, stage_b_w), &device)?;
                Self::debug_tensor_stats("decoder_latents_init", &decoder_latents);

                (prior_latents, decoder_latents, 0, None)
            };
        drop(prior_text_embeddings);

        // --- Phase 3: Decoder (Stage B) ---
        let decoder_size = std::fs::metadata(&decoder_path)
            .map(|m| m.len())
            .unwrap_or(0);
        preflight_memory_check("Decoder (Stage B)", decoder_size)?;
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        // Decoder uses F32 — the 256x256 latent space overflows F16 range
        self.base.progress.stage_start("Loading Decoder (Stage B)");
        let dec_start = Instant::now();
        let decoder_vb = crate::weight_loader::load_safetensors_with_progress(
            &[&decoder_path],
            DType::F32,
            &device,
            "Wuerstchen Decoder",
            &self.base.progress,
        )?;
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

        // Cast text embeddings to F32 for Decoder
        let decoder_text_embeddings = decoder_text_embeddings.to_dtype(DType::F32)?;

        self.denoise_decoder(
            &decoder,
            &image_embeddings,
            &decoder_text_embeddings,
            &mut decoder_latents,
            seed,
            decoder_steps,
            decoder_start_step,
            decoder_guidance,
            inpaint_ctx.as_ref(),
            &device,
            DType::F32,
        )?;
        Self::debug_tensor_stats("decoder_latents_denoised", &decoder_latents);

        drop(decoder);
        drop(image_embeddings);
        drop(decoder_text_embeddings);
        drop(inpaint_ctx);
        device.synchronize()?;
        self.base.progress.info("Freed Decoder (Stage B)");

        // --- Phase 4: VQ-GAN decode (Stage A) ---
        // VQ-GAN uses F32 for pixel-space decoding regardless of model dtype
        let vqgan_load_label = if is_img2img {
            "Reloading VQ-GAN (Stage A)"
        } else {
            "Loading VQ-GAN (Stage A)"
        };
        self.base.progress.stage_start(vqgan_load_label);
        let vqgan_start = Instant::now();
        let vqgan_vb = crate::weight_loader::load_safetensors_with_progress(
            &[&self.base.paths.vae],
            DType::F32,
            &device,
            "VQ-GAN",
            &self.base.progress,
        )?;
        let vqgan = PaellaVQ::new(vqgan_vb)?;
        self.base
            .progress
            .stage_done(vqgan_load_label, vqgan_start.elapsed());

        self.base.progress.stage_start("VQ-GAN decode");
        let decode_start = Instant::now();
        Self::debug_tensor_stats("decoder_latents_pre_vq", &decoder_latents);
        let img = vqgan.decode(&(&decoder_latents * VQGAN_SCALE)?)?;
        Self::debug_tensor_stats("image_pre_postprocess", &img);
        let img = img.clamp(0f32, 1f32)?;
        Self::debug_tensor_stats("image_postprocess", &img);
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
            video: None,
        })
    }
}

impl InferenceEngine for WuerstchenEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if req.scheduler.is_some() {
            tracing::warn!("scheduler selection not supported for Wuerstchen, ignoring");
        }

        if self.base.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Reload Prior/Decoder if dropped after previous VQ-GAN decode
        self.reload_models_if_needed()?;

        let loaded = self
            .base
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);
        let width = req.width as usize;
        let height = req.height as usize;
        let prior_guidance = req.guidance;
        let decoder_guidance = Self::decoder_guidance();
        let negative_prompt = req.negative_prompt.as_deref().unwrap_or("");
        let prior_steps = Self::effective_prior_steps(req.steps as usize);
        let decoder_steps = 12;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            prior_steps,
            decoder_steps,
            prior_guidance,
            decoder_guidance,
            "starting Wuerstchen generation"
        );

        // 1. Encode prompt with Prior CLIP-G (1280-dim)
        let (prior_text_embeddings, decoder_text_embeddings) = self.encode_prompt_pair_cached(
            &loaded.prior_clip,
            &loaded.prior_tokenizer,
            &loaded.decoder_clip,
            &loaded.decoder_tokenizer,
            &req.prompt,
            negative_prompt,
            &loaded.device,
            loaded.dtype,
            prior_guidance,
            decoder_guidance,
        )?;
        Self::debug_tensor_stats("prior_text_embeddings", &prior_text_embeddings);
        Self::debug_tensor_stats("decoder_text_embeddings", &decoder_text_embeddings);

        // 2. Prepare latents: img2img (VQ-GAN encode, skip Prior) or txt2img (Prior)
        let (image_embeddings, mut decoder_latents, decoder_start_step, inpaint_ctx) =
            if let Some(ref source_bytes) = req.source_image {
                self.base
                    .progress
                    .info("img2img mode: skipping Prior, encoding source via VQ-GAN");

                let (noised, start_step, encoded, noise) = self.prepare_img2img_latents(
                    &loaded.vqgan,
                    source_bytes,
                    req.width,
                    req.height,
                    req.strength,
                    decoder_steps,
                    seed,
                    &loaded.device,
                )?;

                let (_, _, enc_h, enc_w) = encoded.dims4()?;
                let inpaint_ctx = crate::img2img::maybe_build_inpaint_context(
                    req.mask_image.as_deref(),
                    &encoded,
                    &noise,
                    enc_h,
                    enc_w,
                    &loaded.device,
                    DType::F32,
                )?;

                // Use zeros for effnet conditioning (no Prior output)
                let prior_latent_h = (height as f64 / LATENT_DIM_SCALE).ceil() as usize;
                let prior_latent_w = (width as f64 / LATENT_DIM_SCALE).ceil() as usize;
                let image_embeddings = Tensor::zeros(
                    (1, PRIOR_C_IN, prior_latent_h, prior_latent_w),
                    DType::F32,
                    &loaded.device,
                )?;

                Self::debug_tensor_stats("decoder_latents_init", &noised);
                (image_embeddings, noised, start_step, inpaint_ctx)
            } else {
                // txt2img: run Stage C (Prior) to generate image embeddings
                let latent_h = (height as f64 / LATENT_DIM_SCALE).ceil() as usize;
                let latent_w = (width as f64 / LATENT_DIM_SCALE).ceil() as usize;
                loaded.device.set_seed(seed)?;
                let mut prior_latents = Tensor::randn(
                    0f32,
                    1f32,
                    (1, PRIOR_C_IN, latent_h, latent_w),
                    &loaded.device,
                )?
                .to_dtype(loaded.dtype)?;
                Self::debug_tensor_stats("prior_latents_init", &prior_latents);

                let prior = loaded
                    .prior
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Prior not loaded"))?;
                self.denoise_prior(
                    prior,
                    &prior_text_embeddings,
                    &mut prior_latents,
                    seed,
                    prior_steps,
                    prior_guidance,
                    &loaded.device,
                    loaded.dtype,
                )?;

                // Scale prior output: convert from Prior latent space to Decoder conditioning space
                Self::debug_tensor_stats("prior_latents_denoised", &prior_latents);
                prior_latents = ((prior_latents * 42.)? - 1.)?;
                Self::debug_tensor_stats("image_embeddings", &prior_latents);

                // Stage B (Decoder): decode prior latents to VQ-GAN latent space
                let prior_latents = prior_latents.to_dtype(DType::F32)?;
                let stage_b_h = (prior_latents.dim(2)? as f64 * LATENT_DIM_SCALE_DECODER) as usize;
                let stage_b_w = (prior_latents.dim(3)? as f64 * LATENT_DIM_SCALE_DECODER) as usize;
                loaded.device.set_seed(seed.wrapping_add(1))?;
                let decoder_latents =
                    Tensor::randn(0f32, 1f32, (1, 4, stage_b_h, stage_b_w), &loaded.device)?;
                Self::debug_tensor_stats("decoder_latents_init", &decoder_latents);

                (prior_latents, decoder_latents, 0, None)
            };

        // 3. Stage B (Decoder): denoise
        // Cast text embeddings to F32 for Decoder (F16 overflows)
        let decoder_text_embeddings = decoder_text_embeddings.to_dtype(DType::F32)?;

        let decoder = loaded
            .decoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Decoder not loaded"))?;
        self.denoise_decoder(
            decoder,
            &image_embeddings,
            &decoder_text_embeddings,
            &mut decoder_latents,
            seed,
            decoder_steps,
            decoder_start_step,
            decoder_guidance,
            inpaint_ctx.as_ref(),
            &loaded.device,
            DType::F32,
        )?;
        Self::debug_tensor_stats("decoder_latents_denoised", &decoder_latents);

        // Drop Prior and Decoder before VQ-GAN decode to free VRAM.
        drop(inpaint_ctx);
        let _ = loaded;
        let loaded = self.base.loaded.as_mut().unwrap();
        loaded.prior = None;
        loaded.decoder = None;
        loaded.device.synchronize()?;
        tracing::info!("Prior + Decoder dropped to free VRAM for VQ-GAN decode");
        let _ = loaded;
        let loaded = self.base.loaded.as_ref().unwrap();

        // 4. Stage A (VQ-GAN): decode to pixel space
        self.base.progress.stage_start("VQ-GAN decode");
        let decode_start = Instant::now();
        Self::debug_tensor_stats("decoder_latents_pre_vq", &decoder_latents);
        let img = loaded.vqgan.decode(&(&decoder_latents * VQGAN_SCALE)?)?;
        Self::debug_tensor_stats("image_pre_postprocess", &img);
        let img = img.clamp(0f32, 1f32)?;
        Self::debug_tensor_stats("image_postprocess", &img);
        let img = (img * 255.)?.to_dtype(DType::U8)?;
        let img = img.squeeze(0)?;
        self.base
            .progress
            .stage_done("VQ-GAN decode", decode_start.elapsed());

        // 5. Encode to image format
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
            video: None,
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

    fn model_paths(&self) -> Option<&mold_core::ModelPaths> {
        Some(&self.base.paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::LoadStrategy;
    use mold_core::ModelPaths;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_test_dir(prefix: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("{prefix}-{}-{suffix}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, b"test").unwrap();
        path
    }

    fn wuerstchen_model_paths(
        transformer: PathBuf,
        decoder: Option<PathBuf>,
        vae: PathBuf,
        prior_clip_encoder: Option<PathBuf>,
        prior_clip_tokenizer: Option<PathBuf>,
        decoder_clip_encoder: Option<PathBuf>,
        decoder_clip_tokenizer: Option<PathBuf>,
    ) -> ModelPaths {
        ModelPaths {
            transformer,
            transformer_shards: vec![],
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: decoder_clip_encoder,
            t5_tokenizer: None,
            clip_tokenizer: decoder_clip_tokenizer,
            clip_encoder_2: prior_clip_encoder,
            clip_tokenizer_2: prior_clip_tokenizer,
            text_encoder_files: vec![],
            text_tokenizer: None,
            decoder,
        }
    }

    fn test_tokenizer() -> tokenizers::Tokenizer {
        let tokenizer_json = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<|endoftext|>": 7,
      "hello": 11
    },
    "unk_token": "<|endoftext|>"
  }
}"#;
        tokenizers::Tokenizer::from_bytes(tokenizer_json.as_bytes()).unwrap()
    }

    #[test]
    fn prompt_cache_key_includes_negative_prompt_and_cfg() {
        let base = WuerstchenEngine::prompt_cache_key("hello", "", false, false);
        let neg = WuerstchenEngine::prompt_cache_key("hello", "bad", false, false);
        let prior_cfg = WuerstchenEngine::prompt_cache_key("hello", "", true, false);
        let decoder_cfg = WuerstchenEngine::prompt_cache_key("hello", "", false, true);

        assert_ne!(base, neg);
        assert_ne!(base, prior_cfg);
        assert_ne!(base, decoder_cfg);
        assert_ne!(prior_cfg, decoder_cfg);
    }

    #[test]
    fn tokenize_uses_clip_pad_token() {
        let tokenizer = test_tokenizer();
        let clip_config = stable_diffusion::clip::Config::wuerstchen();
        let (tokens, tokens_len) =
            WuerstchenEngine::tokenize(&tokenizer, "hello", 4, &clip_config, &Device::Cpu).unwrap();
        let ids = tokens.squeeze(0).unwrap().to_vec1::<u32>().unwrap();

        assert_eq!(tokens_len, 1);
        assert_eq!(ids, vec![11, 7, 7, 7]);
    }

    #[test]
    fn tokenize_falls_back_to_pad_token_for_empty_prompt() {
        let tokenizer = test_tokenizer();
        let clip_config = stable_diffusion::clip::Config::wuerstchen();
        let (tokens, tokens_len) =
            WuerstchenEngine::tokenize(&tokenizer, "", 3, &clip_config, &Device::Cpu).unwrap();
        let ids = tokens.squeeze(0).unwrap().to_vec1::<u32>().unwrap();

        assert_eq!(tokens_len, 1);
        assert_eq!(ids, vec![7, 7, 7]);
    }

    #[test]
    fn effective_prior_steps_passes_through() {
        assert_eq!(WuerstchenEngine::effective_prior_steps(30), 30);
        assert_eq!(WuerstchenEngine::effective_prior_steps(60), 60);
        assert_eq!(WuerstchenEngine::effective_prior_steps(20), 20);
    }

    #[test]
    fn ddpmw_add_noise_matches_reference_formula() {
        let dev = Device::Cpu;
        let original = Tensor::from_vec(vec![2.0f32, -1.0], (1, 2), &dev).unwrap();
        let noise = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &dev).unwrap();
        let t = 0.5f64;

        let actual = WuerstchenEngine::ddpmw_add_noise(&original, &noise, t)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let s = 0.008f64;
        let init_alpha_cumprod = (s / (1.0 + s) * std::f64::consts::PI).cos().powi(2);
        let alpha_cumprod = ((t + s) / (1.0 + s) * std::f64::consts::PI * 0.5)
            .cos()
            .powi(2)
            / init_alpha_cumprod;
        let alpha_cumprod = alpha_cumprod.clamp(0.0001, 0.9999);
        let sqrt_alpha = alpha_cumprod.sqrt() as f32;
        let sqrt_one_minus_alpha = (1.0 - alpha_cumprod).sqrt() as f32;
        let expected = vec![
            2.0f32 * sqrt_alpha + 3.0 * sqrt_one_minus_alpha,
            -1.0f32 * sqrt_alpha + 4.0 * sqrt_one_minus_alpha,
        ];

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn validate_paths_falls_back_to_prior_clip_when_decoder_clip_missing() {
        let dir = temp_test_dir("mold-wuerstchen-validate-ok");
        let transformer = touch(&dir, "prior.safetensors");
        let decoder = touch(&dir, "decoder.safetensors");
        let vae = touch(&dir, "vqgan.safetensors");
        let prior_clip_encoder = touch(&dir, "prior-clip.safetensors");
        let prior_clip_tokenizer = touch(&dir, "prior-tokenizer.json");

        let engine = WuerstchenEngine::new(
            "wuerstchen-v2:fp16".to_string(),
            wuerstchen_model_paths(
                transformer,
                Some(decoder.clone()),
                vae,
                Some(prior_clip_encoder.clone()),
                Some(prior_clip_tokenizer.clone()),
                None,
                None,
            ),
            LoadStrategy::Sequential,
        );

        let (
            decoder_path,
            resolved_prior_clip_encoder,
            resolved_prior_clip_tokenizer,
            resolved_decoder_clip_encoder,
            resolved_decoder_clip_tokenizer,
        ) = engine.validate_paths().unwrap();

        assert_eq!(decoder_path, decoder);
        assert_eq!(resolved_prior_clip_encoder, prior_clip_encoder);
        assert_eq!(resolved_prior_clip_tokenizer, prior_clip_tokenizer);
        assert_eq!(resolved_decoder_clip_encoder, resolved_prior_clip_encoder);
        assert_eq!(
            resolved_decoder_clip_tokenizer,
            resolved_prior_clip_tokenizer
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn validate_paths_requires_decoder_and_existing_files() {
        let dir = temp_test_dir("mold-wuerstchen-validate-missing");
        let transformer = touch(&dir, "prior.safetensors");
        let vae = touch(&dir, "vqgan.safetensors");
        let prior_clip_encoder = touch(&dir, "prior-clip.safetensors");
        let prior_clip_tokenizer = touch(&dir, "prior-tokenizer.json");

        let missing_decoder_engine = WuerstchenEngine::new(
            "wuerstchen-v2:fp16".to_string(),
            wuerstchen_model_paths(
                transformer.clone(),
                None,
                vae.clone(),
                Some(prior_clip_encoder.clone()),
                Some(prior_clip_tokenizer.clone()),
                None,
                None,
            ),
            LoadStrategy::Sequential,
        );
        let err = missing_decoder_engine.validate_paths().unwrap_err();
        assert!(err.to_string().contains("Decoder (Stage B) path required"));

        let missing_file_engine = WuerstchenEngine::new(
            "wuerstchen-v2:fp16".to_string(),
            wuerstchen_model_paths(
                transformer,
                Some(dir.join("missing-decoder.safetensors")),
                vae,
                Some(prior_clip_encoder),
                Some(prior_clip_tokenizer),
                None,
                None,
            ),
            LoadStrategy::Sequential,
        );
        let err = missing_file_engine.validate_paths().unwrap_err();
        assert!(err.to_string().contains("decoder (Stage B) file not found"));

        fs::remove_dir_all(dir).ok();
    }
}
