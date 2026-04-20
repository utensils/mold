use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp};
use candle_transformers::models::mmdit::model::{Config as MMDiTConfig, MMDiT};
use candle_transformers::quantized_var_builder;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::sync::Mutex;
use std::time::Instant;

use crate::cache::{
    clear_cache, get_or_insert_cached_tensor_pair, prompt_text_key, restore_cached_tensor_pair,
    CachedTensorPair, LruCache, DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{
    check_memory_budget, fmt_gb, free_vram_bytes, memory_status_string, preflight_memory_check,
};
use crate::encoders;
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy, OptionRestoreGuard};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image};
use crate::img_utils;
use crate::progress::{ProgressCallback, ProgressReporter};

use super::quantized_mmdit::QuantizedMMDiT;
use super::sampling::{self, SkipLayerGuidanceConfig};
use super::transformer::SD3Transformer;
use super::vae::{build_sd3_vae_autoencoder, sd3_vae_vb_rename};

/// Loaded SD3 model components, ready for inference.
struct LoadedSD3 {
    /// None after being dropped for VAE decode VRAM; reloaded on next generate.
    transformer: Option<SD3Transformer>,
    triple_encoder: encoders::sd3_clip::SD3TripleEncoder,
    vae_vb_path: std::path::PathBuf,
    device: Device,
    dtype: DType,
    _is_quantized: bool,
    is_turbo: bool,
    is_medium: bool,
}

/// SD3.5 inference engine backed by candle.
///
/// Supports SD3.5 Large (8.1B, depth=38), SD3.5 Large Turbo (8.1B, 4 steps),
/// and SD3.5 Medium (2.5B, depth=24, SLG support).
/// Both BF16 safetensors and GGUF quantized transformers are supported.
pub struct SD3Engine {
    base: EngineBase<LoadedSD3>,
    is_turbo: bool,
    is_medium: bool,
    t5_variant: Option<String>,
    prompt_cache: Mutex<LruCache<String, CachedTensorPair>>,
    pending_placement: Option<mold_core::types::DevicePlacement>,
}

impl SD3Engine {
    /// Create a new SD3Engine. Does not load models until `load()` is called.
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        is_turbo: bool,
        is_medium: bool,
        t5_variant: Option<String>,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy, gpu_ordinal),
            is_turbo,
            is_medium,
            t5_variant,
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            pending_placement: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_conditioning(
        progress: &ProgressReporter,
        prompt_cache: &Mutex<LruCache<String, CachedTensorPair>>,
        triple_encoder: &mut encoders::sd3_clip::SD3TripleEncoder,
        prompt: &str,
        negative_prompt: &str,
        device: &Device,
        dtype: DType,
        is_quantized: bool,
    ) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
        let cache_key = prompt_text_key(prompt);
        let ((context, y), cache_hit) = get_or_insert_cached_tensor_pair(
            prompt_cache,
            cache_key,
            device,
            if is_quantized { DType::F32 } else { dtype },
            || {
                progress.stage_start("Encoding prompt (SD3 triple)");
                let encode_start = Instant::now();
                let (context_cond, y_cond) = triple_encoder.encode(prompt, device, dtype)?;
                let (context_uncond, y_uncond) =
                    triple_encoder.encode(negative_prompt, device, dtype)?;
                progress.stage_done("Encoding prompt (SD3 triple)", encode_start.elapsed());

                let pair = if is_quantized {
                    (
                        candle_core::Tensor::cat(&[&context_cond, &context_uncond], 0)?
                            .to_dtype(DType::F32)?,
                        candle_core::Tensor::cat(&[&y_cond, &y_uncond], 0)?.to_dtype(DType::F32)?,
                    )
                } else {
                    (
                        candle_core::Tensor::cat(&[&context_cond, &context_uncond], 0)?,
                        candle_core::Tensor::cat(&[&y_cond, &y_uncond], 0)?,
                    )
                };
                Ok(pair)
            },
        )?;
        if cache_hit {
            progress.cache_hit("prompt conditioning");
            return Ok((context, y));
        }
        Ok((context, y))
    }

    fn img2img_source_normalize_range() -> img_utils::NormalizeRange {
        img_utils::NormalizeRange::MinusOneToOne
    }

    /// Detect if the transformer is quantized (GGUF).
    fn detect_is_quantized(&self) -> bool {
        self.base
            .paths
            .transformer
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    }

    /// Get the MMDiT config for this model variant.
    fn mmdit_config(&self) -> MMDiTConfig {
        if self.is_medium {
            MMDiTConfig::sd3_5_medium()
        } else {
            MMDiTConfig::sd3_5_large()
        }
    }

    /// Validate that all required paths exist.
    fn validate_paths(
        &self,
    ) -> Result<(
        std::path::PathBuf, // clip_l_path
        std::path::PathBuf, // clip_l_tokenizer
        std::path::PathBuf, // clip_g_path
        std::path::PathBuf, // clip_g_tokenizer
        std::path::PathBuf, // t5_encoder_path
        std::path::PathBuf, // t5_tokenizer_path
    )> {
        let clip_l_path = self
            .base
            .paths
            .clip_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-L encoder path required for SD3 models"))?
            .clone();
        let clip_l_tokenizer = self
            .base
            .paths
            .clip_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-L tokenizer path required for SD3 models"))?
            .clone();
        let clip_g_path = self
            .base
            .paths
            .clip_encoder_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-G encoder path required for SD3 models"))?
            .clone();
        let clip_g_tokenizer = self
            .base
            .paths
            .clip_tokenizer_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-G tokenizer path required for SD3 models"))?
            .clone();
        let t5_encoder_path = self
            .base
            .paths
            .t5_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("T5 encoder path required for SD3 models"))?
            .clone();
        let t5_tokenizer_path = self
            .base
            .paths
            .t5_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("T5 tokenizer path required for SD3 models"))?
            .clone();

        for (label, path) in [
            ("transformer", &self.base.paths.transformer),
            ("vae", &self.base.paths.vae),
            ("clip_encoder (CLIP-L)", &clip_l_path),
            ("clip_tokenizer (CLIP-L)", &clip_l_tokenizer),
            ("clip_encoder_2 (CLIP-G)", &clip_g_path),
            ("clip_tokenizer_2 (CLIP-G)", &clip_g_tokenizer),
            ("t5_encoder", &t5_encoder_path),
            ("t5_tokenizer", &t5_tokenizer_path),
        ] {
            if !path.exists() {
                bail!("{label} file not found: {}", path.display());
            }
        }

        Ok((
            clip_l_path,
            clip_l_tokenizer,
            clip_g_path,
            clip_g_tokenizer,
            t5_encoder_path,
            t5_tokenizer_path,
        ))
    }

    /// Load all model components into GPU memory (Eager mode).
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

        tracing::info!(model = %self.base.model_name, "loading SD3 model components...");

        let (
            clip_l_path,
            clip_l_tokenizer,
            clip_g_path,
            clip_g_tokenizer,
            t5_encoder_path,
            t5_tokenizer_path,
        ) = self.validate_paths()?;

        let device = crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)?;
        let gpu_dtype = if crate::device::is_gpu(&device) {
            DType::F16
        } else {
            DType::F32
        };

        let is_quantized = self.detect_is_quantized();
        let mmdit_config = self.mmdit_config();

        // --- Load MMDiT transformer on GPU first ---
        let xformer_label = if is_quantized {
            "Loading SD3 MMDiT transformer (GPU, quantized)"
        } else {
            "Loading SD3 MMDiT transformer (GPU, FP16)"
        };
        self.base.progress.stage_start(xformer_label);
        let xformer_stage = Instant::now();

        let transformer = if is_quantized {
            // GGUF files from city96 use unprefixed tensor names (no "model.diffusion_model.")
            let vb = quantized_var_builder::VarBuilder::from_gguf(
                &self.base.paths.transformer,
                &device,
            )?;
            SD3Transformer::Quantized(QuantizedMMDiT::new(&mmdit_config, vb)?)
        } else {
            // BF16 safetensors from stabilityai use "model.diffusion_model." prefix
            let vb = crate::weight_loader::load_safetensors_with_progress(
                std::slice::from_ref(&self.base.paths.transformer),
                gpu_dtype,
                &device,
                "SD3 transformer",
                &self.base.progress,
            )?;
            SD3Transformer::BF16(MMDiT::new(
                &mmdit_config,
                false,
                vb.pp("model.diffusion_model"),
            )?)
        };
        self.base
            .progress
            .stage_done(xformer_label, xformer_stage.elapsed());

        // --- Decide encoder placement based on remaining VRAM ---
        let free = free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        if free > 0 {
            self.base
                .progress
                .info(&format!("Free VRAM after transformer: {}", fmt_gb(free)));
        }

        // --- Load triple encoder (CLIP-L + CLIP-G + T5) ---
        // For T5, use variant resolution logic
        self.base.progress.stage_start("Selecting T5 encoder");
        let t5_resolve_start = Instant::now();
        let t5_preference = self.t5_variant.as_deref();
        let (resolved_t5_path, t5_on_gpu, _t5_auto_device_label) =
            crate::encoders::variant_resolution::resolve_t5_variant(
                &self.base.progress,
                t5_preference,
                &device,
                free,
                &t5_encoder_path,
            )?;
        self.base
            .progress
            .stage_done("Selecting T5 encoder", t5_resolve_start.elapsed());

        // Tier 1: honor `placement.text_encoders` — all three encoders share the knob.
        let tier1 = self
            .pending_placement
            .as_ref()
            .map(|p| p.text_encoders)
            .unwrap_or_default();
        let auto_encoder_device = if t5_on_gpu { device.clone() } else { Device::Cpu };
        let encoder_device_owned = crate::device::resolve_device(
            Some(tier1),
            || Ok(auto_encoder_device.clone()),
        )?;
        let encoder_device = &encoder_device_owned;
        let t5_on_gpu = !encoder_device.is_cpu();
        let t5_device_label = if t5_on_gpu { "GPU" } else { "CPU" };
        let encoder_dtype = if t5_on_gpu { gpu_dtype } else { DType::F32 };

        let encoder_label = format!("Loading SD3 triple encoder ({t5_device_label})");
        self.base.progress.stage_start(&encoder_label);
        let encoder_stage = Instant::now();

        let triple_encoder = encoders::sd3_clip::SD3TripleEncoder::load(
            &clip_l_path,
            &clip_l_tokenizer,
            &clip_g_path,
            &clip_g_tokenizer,
            &resolved_t5_path,
            &t5_tokenizer_path,
            encoder_device,
            encoder_dtype,
            &self.base.progress,
        )?;

        self.base
            .progress
            .stage_done(&encoder_label, encoder_stage.elapsed());

        self.base.loaded = Some(LoadedSD3 {
            transformer: Some(transformer),
            triple_encoder,
            vae_vb_path: self.base.paths.vae.clone(),
            device,
            dtype: gpu_dtype,
            _is_quantized: is_quantized,
            is_turbo: self.is_turbo,
            is_medium: self.is_medium,
        });

        tracing::info!(model = %self.base.model_name, "all SD3 model components loaded successfully");
        Ok(())
    }

    /// Get SLG config if applicable (Medium only).
    fn slg_config(&self) -> Option<SkipLayerGuidanceConfig> {
        if self.is_medium {
            Some(SkipLayerGuidanceConfig {
                scale: 2.5,
                start: 0.01,
                end: 0.2,
                layers: vec![7, 8, 9],
            })
        } else {
            None
        }
    }

    /// Generate an image using sequential loading strategy.
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let (
            clip_l_path,
            clip_l_tokenizer,
            clip_g_path,
            clip_g_tokenizer,
            t5_encoder_path,
            t5_tokenizer_path,
        ) = self.validate_paths()?;

        if let Some(warning) = check_memory_budget(&self.base.paths, LoadStrategy::Sequential) {
            self.base.progress.info(&warning);
        }

        let device = crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)?;
        let gpu_dtype = if crate::device::is_gpu(&device) {
            DType::F16
        } else {
            DType::F32
        };

        let is_quantized = self.detect_is_quantized();
        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            guidance = req.guidance,
            "starting sequential SD3 generation"
        );

        self.base
            .progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: Encode prompt (check cache first to skip encoder load) ---
        let neg = req.negative_prompt.as_deref().unwrap_or("");
        let cache_key = prompt_text_key(&req.prompt);
        let (context, y) = if let Some((context, y)) =
            restore_cached_tensor_pair(&self.prompt_cache, &cache_key, &device, gpu_dtype)?
        {
            self.base.progress.cache_hit("prompt conditioning");
            (context, y)
        } else {
            let free = free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
            self.base.progress.stage_start("Selecting T5 encoder");
            let t5_resolve_start = Instant::now();
            let t5_preference = self.t5_variant.as_deref();
            let (resolved_t5_path, t5_on_gpu, _t5_auto_device_label) =
                crate::encoders::variant_resolution::resolve_t5_variant(
                    &self.base.progress,
                    t5_preference,
                    &device,
                    free,
                    &t5_encoder_path,
                )?;
            self.base
                .progress
                .stage_done("Selecting T5 encoder", t5_resolve_start.elapsed());

            let tier1 = self
                .pending_placement
                .as_ref()
                .map(|p| p.text_encoders)
                .unwrap_or_default();
            let auto_encoder_device = if t5_on_gpu { device.clone() } else { Device::Cpu };
            let encoder_device_owned = crate::device::resolve_device(
                Some(tier1),
                || Ok(auto_encoder_device.clone()),
            )?;
            let encoder_device = &encoder_device_owned;
            let t5_on_gpu = !encoder_device.is_cpu();
            let t5_device_label = if t5_on_gpu { "GPU" } else { "CPU" };
            let encoder_dtype = if t5_on_gpu { gpu_dtype } else { DType::F32 };

            let t5_size = std::fs::metadata(&resolved_t5_path)
                .map(|m| m.len())
                .unwrap_or(0);
            preflight_memory_check("SD3 triple encoder", t5_size)?;
            if let Some(status) = memory_status_string() {
                self.base.progress.info(&status);
            }

            let encoder_label = format!("Loading SD3 triple encoder ({t5_device_label})");
            self.base.progress.stage_start(&encoder_label);
            let encoder_stage = Instant::now();
            let mut triple_encoder = encoders::sd3_clip::SD3TripleEncoder::load(
                &clip_l_path,
                &clip_l_tokenizer,
                &clip_g_path,
                &clip_g_tokenizer,
                &resolved_t5_path,
                &t5_tokenizer_path,
                encoder_device,
                encoder_dtype,
                &self.base.progress,
            )?;
            self.base
                .progress
                .stage_done(&encoder_label, encoder_stage.elapsed());

            let (context, y) = Self::encode_conditioning(
                &self.base.progress,
                &self.prompt_cache,
                &mut triple_encoder,
                &req.prompt,
                neg,
                &device,
                gpu_dtype,
                is_quantized,
            )?;

            drop(triple_encoder);
            self.base.progress.info("Freed SD3 triple encoder");

            (context, y)
        };

        // --- Phase 2: img2img — encode source image if provided ---
        let noise_dtype = if is_quantized { DType::F32 } else { gpu_dtype };
        let latent_h = height / 16 * 2;
        let latent_w = width / 16 * 2;
        let time_shift = 3.0;

        // Build sigma schedule
        let num_steps = req.steps as usize;
        let mut sigmas: Vec<f64> = (0..=num_steps)
            .map(|s| s as f64 / num_steps as f64)
            .rev()
            .map(|t| sampling::time_snr_shift(time_shift, t))
            .collect();

        if req.source_image.is_some() {
            let (trimmed, start_index) =
                crate::img2img::trim_schedule_tail(&sigmas, req.steps as usize, req.strength);
            sigmas = trimmed;
            tracing::info!(
                strength = req.strength,
                start_index,
                start_sigma = sigmas[0],
                schedule = ?sigmas,
                remaining_steps = sigmas.len().saturating_sub(1),
                "img2img: truncated schedule from strength"
            );
        }

        let (initial_latents, inpaint_ctx) = if let Some(ref source_bytes) = req.source_image {
            let start_t = sigmas[0];

            // Load VAE early for source image encoding
            self.base.progress.stage_start("Loading VAE for encoding");
            let vae_stage = Instant::now();
            let vae_vb = crate::weight_loader::load_safetensors_with_progress(
                std::slice::from_ref(&self.base.paths.vae),
                gpu_dtype,
                &device,
                "VAE",
                &self.base.progress,
            )?;
            let vae_vb = vae_vb.rename_f(sd3_vae_vb_rename).pp("first_stage_model");
            let autoencoder = build_sd3_vae_autoencoder(vae_vb)?;
            self.base
                .progress
                .stage_done("Loading VAE for encoding", vae_stage.elapsed());

            self.base
                .progress
                .stage_start("Encoding source image (VAE)");
            let encode_start = Instant::now();
            let source_tensor = img_utils::decode_source_image(
                source_bytes,
                req.width,
                req.height,
                Self::img2img_source_normalize_range(),
                &device,
                gpu_dtype,
            )?;
            let dist = autoencoder.encode(&source_tensor)?;
            // SD3 VAE encode scaling: reverse of decode's x / 1.5305 + 0.0609.
            // Use the posterior mean so img2img remains deterministic.
            let encoded = ((dist.mode()? - 0.0609)? * 1.5305)?;
            self.base
                .progress
                .stage_done("Encoding source image (VAE)", encode_start.elapsed());

            // Drop VAE to free VRAM for transformer (will reload for decode)
            drop(autoencoder);
            device.synchronize()?;
            self.base
                .progress
                .info("Freed VAE encoder to make room for transformer");

            let encoded = encoded.to_dtype(noise_dtype)?;
            let prepared = crate::img2img::prepare_flow_match_img2img(
                &encoded,
                seed,
                &[1, 16, latent_h, latent_w],
                start_t,
                req.mask_image.as_deref(),
                latent_h,
                latent_w,
                &device,
                noise_dtype,
            )?;
            (Some(prepared.initial_latents), prepared.inpaint_ctx)
        } else {
            (None, None)
        };

        // --- Phase 3: Load transformer + denoise ---
        let mmdit_config = self.mmdit_config();

        let xformer_size = std::fs::metadata(&self.base.paths.transformer)
            .map(|m| m.len())
            .unwrap_or(0);
        preflight_memory_check("SD3 MMDiT transformer", xformer_size)?;
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        let xformer_label = if is_quantized {
            "Loading SD3 MMDiT transformer (GPU, quantized)"
        } else {
            "Loading SD3 MMDiT transformer (GPU, FP16)"
        };
        self.base.progress.stage_start(xformer_label);
        let xformer_stage = Instant::now();

        let transformer = if is_quantized {
            // GGUF files from city96 use unprefixed tensor names
            let vb = quantized_var_builder::VarBuilder::from_gguf(
                &self.base.paths.transformer,
                &device,
            )?;
            SD3Transformer::Quantized(QuantizedMMDiT::new(&mmdit_config, vb)?)
        } else {
            // BF16 safetensors from stabilityai use "model.diffusion_model." prefix
            let vb = crate::weight_loader::load_safetensors_with_progress(
                std::slice::from_ref(&self.base.paths.transformer),
                gpu_dtype,
                &device,
                "SD3 transformer",
                &self.base.progress,
            )?;
            SD3Transformer::BF16(MMDiT::new(
                &mmdit_config,
                false,
                vb.pp("model.diffusion_model"),
            )?)
        };
        self.base
            .progress
            .stage_done(xformer_label, xformer_stage.elapsed());

        // Denoise
        let slg_config = self.slg_config();
        let actual_steps = sigmas.len().saturating_sub(1);
        let denoise_label = format!("Denoising ({actual_steps} steps)");
        self.base.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        let x = sampling::euler_sample(
            &transformer,
            &y,
            &context,
            num_steps,
            req.guidance,
            time_shift,
            height,
            width,
            slg_config.as_ref(),
            is_quantized,
            seed,
            &self.base.progress,
            initial_latents.as_ref(),
            Some(sigmas),
            inpaint_ctx.as_ref(),
        )?;

        self.base
            .progress
            .stage_done(&denoise_label, denoise_start.elapsed());

        // Drop transformer to free memory for VAE
        drop(transformer);
        drop(context);
        drop(y);
        drop(inpaint_ctx);
        device.synchronize()?;
        self.base.progress.info("Freed SD3 MMDiT transformer");

        // --- Phase 4: VAE decode ---
        self.base.progress.stage_start("Loading VAE (GPU)");
        let vae_stage = Instant::now();
        let vae_vb = crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(&self.base.paths.vae),
            gpu_dtype,
            &device,
            "VAE",
            &self.base.progress,
        )?;
        let vae_vb = vae_vb.rename_f(sd3_vae_vb_rename).pp("first_stage_model");
        let autoencoder = build_sd3_vae_autoencoder(vae_vb)?;
        self.base
            .progress
            .stage_done("Loading VAE (GPU)", vae_stage.elapsed());

        self.base.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();

        // SD3 VAE scaling: x / 1.5305 + 0.0609
        // Cast to VAE dtype (quantized path outputs F32, VAE is F16/BF16)
        let x = ((x / 1.5305)? + 0.0609)?.to_dtype(gpu_dtype)?;
        let img = autoencoder.decode(&x)?;

        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?;

        self.base
            .progress
            .stage_done("VAE decode", vae_decode_start.elapsed());

        let output_metadata = build_output_metadata(req, seed, None);
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
            "sequential SD3 generation complete"
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
            gpu: None,
        })
    }
}

impl SD3Engine {
    fn generate_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if req.scheduler.is_some() {
            tracing::warn!("scheduler selection not supported for SD3 (flow-matching), ignoring");
        }

        // Sequential mode: load-use-drop each component
        if self.base.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components
        let progress = &self.base.progress;
        let prompt_cache = &self.prompt_cache;
        let mmdit_config = self.mmdit_config();
        let transformer_path = self.base.paths.transformer.clone();

        let mut loaded = OptionRestoreGuard::take(&mut self.base.loaded)
            .ok_or_else(|| anyhow::anyhow!("model not loaded -- call load() first"))?;
        let loaded_dtype = loaded.dtype;
        let loaded_device = loaded.device.clone();
        let is_quantized = loaded._is_quantized;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            guidance = req.guidance,
            turbo = loaded.is_turbo,
            medium = loaded.is_medium,
            "starting SD3 generation"
        );

        (|| -> Result<GenerateResponse> {
            if !loaded.triple_encoder.is_loaded() {
                progress.stage_start("Reloading SD3 triple encoder");
                let reload_start = Instant::now();
                loaded.triple_encoder.reload(loaded_dtype, progress)?;
                progress.stage_done("Reloading SD3 triple encoder", reload_start.elapsed());
            }

            let neg = req.negative_prompt.as_deref().unwrap_or("");
            let (context, y) = Self::encode_conditioning(
                progress,
                prompt_cache,
                &mut loaded.triple_encoder,
                &req.prompt,
                neg,
                &loaded_device,
                loaded_dtype,
                is_quantized,
            )?;

            if loaded.triple_encoder.on_gpu {
                loaded.triple_encoder.drop_weights();
                tracing::info!("SD3 triple encoder dropped from GPU to free VRAM for denoising");
            }

            // --- img2img: build schedule and encode source image ---
            let noise_dtype = if is_quantized {
                DType::F32
            } else {
                loaded_dtype
            };
            let latent_h = height / 16 * 2;
            let latent_w = width / 16 * 2;
            let time_shift = 3.0;
            let num_steps = req.steps as usize;

            let mut sigmas: Vec<f64> = (0..=num_steps)
                .map(|s| s as f64 / num_steps as f64)
                .rev()
                .map(|t| sampling::time_snr_shift(time_shift, t))
                .collect();

            if req.source_image.is_some() {
                let (trimmed, start_index) =
                    crate::img2img::trim_schedule_tail(&sigmas, req.steps as usize, req.strength);
                sigmas = trimmed;
                tracing::info!(
                    strength = req.strength,
                    start_index,
                    start_sigma = sigmas[0],
                    schedule = ?sigmas,
                    remaining_steps = sigmas.len().saturating_sub(1),
                    "img2img: truncated schedule from strength"
                );
            }

            let (initial_latents, inpaint_ctx, early_vae) =
                if let Some(ref source_bytes) = req.source_image {
                    let start_t = sigmas[0];

                    // Drop transformer to make room for VAE encoding
                    loaded.transformer = None;
                    loaded.device.synchronize()?;

                    progress.stage_start("Loading VAE for encoding");
                    let vae_stage = Instant::now();
                    let vae_vb = crate::weight_loader::load_safetensors_with_progress(
                        std::slice::from_ref(&loaded.vae_vb_path),
                        loaded_dtype,
                        &loaded.device,
                        "VAE",
                        progress,
                    )?;
                    let vae_vb = vae_vb.rename_f(sd3_vae_vb_rename).pp("first_stage_model");
                    let autoencoder = build_sd3_vae_autoencoder(vae_vb)?;
                    progress.stage_done("Loading VAE for encoding", vae_stage.elapsed());

                    progress.stage_start("Encoding source image (VAE)");
                    let encode_start = Instant::now();
                    let source_tensor = img_utils::decode_source_image(
                        source_bytes,
                        req.width,
                        req.height,
                        Self::img2img_source_normalize_range(),
                        &loaded_device,
                        loaded_dtype,
                    )?;
                    let dist = autoencoder.encode(&source_tensor)?;
                    // SD3 VAE encode scaling: reverse of decode's x / 1.5305 + 0.0609.
                    // Use the posterior mean so img2img remains deterministic.
                    let encoded = ((dist.mode()? - 0.0609)? * 1.5305)?;
                    progress.stage_done("Encoding source image (VAE)", encode_start.elapsed());

                    // Drop VAE to free VRAM for transformer reload
                    drop(autoencoder);
                    loaded.device.synchronize()?;

                    let encoded = encoded.to_dtype(noise_dtype)?;
                    let prepared = crate::img2img::prepare_flow_match_img2img(
                        &encoded,
                        seed,
                        &[1, 16, latent_h, latent_w],
                        start_t,
                        req.mask_image.as_deref(),
                        latent_h,
                        latent_w,
                        &loaded_device,
                        noise_dtype,
                    )?;
                    (
                        Some(prepared.initial_latents),
                        prepared.inpaint_ctx,
                        None::<()>,
                    )
                } else {
                    (None, None, None)
                };

            // Reload transformer if needed (dropped for img2img VAE encoding, or prior VAE decode)
            if loaded.transformer.is_none() {
                progress.stage_start("Reloading SD3 transformer");
                let reload_start = Instant::now();
                let transformer = if is_quantized {
                    let vb = quantized_var_builder::VarBuilder::from_gguf(
                        &transformer_path,
                        &loaded_device,
                    )?;
                    SD3Transformer::Quantized(QuantizedMMDiT::new(&mmdit_config, vb)?)
                } else {
                    let vb = crate::weight_loader::load_safetensors_with_progress(
                        std::slice::from_ref(&transformer_path),
                        loaded_dtype,
                        &loaded_device,
                        "SD3 transformer",
                        progress,
                    )?;
                    let vb = vb.pp("model.diffusion_model");
                    SD3Transformer::BF16(MMDiT::new(&mmdit_config, false, vb)?)
                };
                loaded.transformer = Some(transformer);
                progress.stage_done("Reloading SD3 transformer", reload_start.elapsed());
            }

            let slg_config = if loaded.is_medium {
                Some(SkipLayerGuidanceConfig {
                    scale: 2.5,
                    start: 0.01,
                    end: 0.2,
                    layers: vec![7, 8, 9],
                })
            } else {
                None
            };

            let actual_steps = sigmas.len().saturating_sub(1);
            let denoise_label = format!("Denoising ({actual_steps} steps)");
            progress.stage_start(&denoise_label);
            let denoise_start = Instant::now();

            let transformer = loaded
                .transformer
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("SD3 transformer not loaded"))?;
            let x = sampling::euler_sample(
                transformer,
                &y,
                &context,
                num_steps,
                req.guidance,
                time_shift,
                height,
                width,
                slg_config.as_ref(),
                loaded._is_quantized,
                seed,
                progress,
                initial_latents.as_ref(),
                Some(sigmas),
                inpaint_ctx.as_ref(),
            )?;

            progress.stage_done(&denoise_label, denoise_start.elapsed());
            drop(context);
            drop(y);
            drop(inpaint_ctx);
            let _ = early_vae;

            // Drop transformer before VAE decode to free VRAM.
            loaded.transformer = None;
            loaded.device.synchronize()?;
            tracing::info!("SD3 transformer dropped to free VRAM for VAE decode");

            progress.stage_start("VAE decode");
            let vae_decode_start = Instant::now();

            let vae_vb = crate::weight_loader::load_safetensors_with_progress(
                std::slice::from_ref(&loaded.vae_vb_path),
                loaded.dtype,
                &loaded.device,
                "VAE",
                progress,
            )?;
            let vae_vb = vae_vb.rename_f(sd3_vae_vb_rename).pp("first_stage_model");
            let autoencoder = build_sd3_vae_autoencoder(vae_vb)?;

            let x = ((x / 1.5305)? + 0.0609)?.to_dtype(loaded.dtype)?;
            let img = autoencoder.decode(&x)?;

            let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
            let img = img.i(0)?;

            progress.stage_done("VAE decode", vae_decode_start.elapsed());

            let output_metadata = build_output_metadata(req, seed, None);
            let image_bytes = encode_image(
                &img,
                req.output_format,
                req.width,
                req.height,
                output_metadata.as_ref(),
            )?;

            let generation_time_ms = start.elapsed().as_millis() as u64;
            tracing::info!(generation_time_ms, seed, "SD3 generation complete");

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
                gpu: None,
            })
        })()
    }
}

impl InferenceEngine for SD3Engine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.pending_placement = req.placement.clone();
        let result = self.generate_inner(req);
        self.pending_placement = None;
        result
    }

    fn model_name(&self) -> &str {
        self.base.model_name()
    }

    fn is_loaded(&self) -> bool {
        self.base.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        SD3Engine::load(self)
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

    #[allow(clippy::too_many_arguments)]
    fn sd3_model_paths(
        transformer: PathBuf,
        vae: PathBuf,
        clip_l_path: Option<PathBuf>,
        clip_l_tokenizer: Option<PathBuf>,
        clip_g_path: Option<PathBuf>,
        clip_g_tokenizer: Option<PathBuf>,
        t5_encoder: Option<PathBuf>,
        t5_tokenizer: Option<PathBuf>,
    ) -> ModelPaths {
        ModelPaths {
            transformer,
            transformer_shards: vec![],
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder,
            clip_encoder: clip_l_path,
            t5_tokenizer,
            clip_tokenizer: clip_l_tokenizer,
            clip_encoder_2: clip_g_path,
            clip_tokenizer_2: clip_g_tokenizer,
            text_encoder_files: vec![],
            text_tokenizer: None,
            decoder: None,
        }
    }

    #[test]
    fn sd3_img2img_uses_minus_one_to_one_source_normalization() {
        assert_eq!(
            SD3Engine::img2img_source_normalize_range(),
            img_utils::NormalizeRange::MinusOneToOne
        );
    }

    #[test]
    fn sd3_mmdit_config_tracks_large_vs_medium_variants() {
        let base_dir = temp_test_dir("mold-sd3-config");
        let large = SD3Engine::new(
            "sd3.5-large:bf16".to_string(),
            sd3_model_paths(
                base_dir.join("transformer.safetensors"),
                base_dir.join("vae.safetensors"),
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            false,
            false,
            None,
            LoadStrategy::Sequential,
            0,
        );
        let medium = SD3Engine::new(
            "sd3.5-medium:bf16".to_string(),
            sd3_model_paths(
                base_dir.join("transformer.safetensors"),
                base_dir.join("vae.safetensors"),
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            false,
            true,
            None,
            LoadStrategy::Sequential,
            0,
        );

        let large_cfg = large.mmdit_config();
        let medium_cfg = medium.mmdit_config();

        assert_eq!(large_cfg.depth, 38);
        assert_eq!(large_cfg.pos_embed_max_size, 192);
        assert_eq!(medium_cfg.depth, 24);
        assert_eq!(medium_cfg.pos_embed_max_size, 384);
        assert!(large.slg_config().is_none());
        let slg = medium.slg_config().unwrap();
        assert_eq!(slg.scale, 2.5);
        assert_eq!(slg.layers, vec![7, 8, 9]);

        fs::remove_dir_all(base_dir).ok();
    }

    #[test]
    fn sd3_validate_paths_accepts_existing_files() {
        let dir = temp_test_dir("mold-sd3-validate-ok");
        let transformer = touch(&dir, "transformer.gguf");
        let vae = touch(&dir, "vae.safetensors");
        let clip_l = touch(&dir, "clip-l.safetensors");
        let clip_l_tok = touch(&dir, "clip-l-tokenizer.json");
        let clip_g = touch(&dir, "clip-g.safetensors");
        let clip_g_tok = touch(&dir, "clip-g-tokenizer.json");
        let t5 = touch(&dir, "t5.safetensors");
        let t5_tok = touch(&dir, "t5-tokenizer.json");

        let engine = SD3Engine::new(
            "sd3.5-large-turbo:q8".to_string(),
            sd3_model_paths(
                transformer,
                vae,
                Some(clip_l),
                Some(clip_l_tok),
                Some(clip_g),
                Some(clip_g_tok),
                Some(t5),
                Some(t5_tok.clone()),
            ),
            true,
            false,
            None,
            LoadStrategy::Sequential,
            0,
        );

        let (_, _, _, _, _, resolved_t5_tok) = engine.validate_paths().unwrap();
        assert_eq!(resolved_t5_tok, t5_tok);
        assert!(engine.detect_is_quantized());

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn sd3_validate_paths_requires_t5_encoder() {
        let dir = temp_test_dir("mold-sd3-validate-missing");
        let engine = SD3Engine::new(
            "sd3.5-large:bf16".to_string(),
            sd3_model_paths(
                dir.join("transformer.safetensors"),
                dir.join("vae.safetensors"),
                Some(dir.join("clip-l.safetensors")),
                Some(dir.join("clip-l-tokenizer.json")),
                Some(dir.join("clip-g.safetensors")),
                Some(dir.join("clip-g-tokenizer.json")),
                None,
                Some(dir.join("t5-tokenizer.json")),
            ),
            false,
            false,
            None,
            LoadStrategy::Sequential,
            0,
        );

        let err = engine.validate_paths().unwrap_err();
        assert!(err.to_string().contains("T5 encoder path required"));
        assert!(!engine.detect_is_quantized());

        fs::remove_dir_all(dir).ok();
    }
}
