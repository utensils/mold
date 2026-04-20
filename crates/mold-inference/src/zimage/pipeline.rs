use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::z_image::{
    calculate_shift, postprocess_image, AutoEncoderKL, Config, FlowMatchEulerDiscreteScheduler,
    SchedulerConfig, VaeConfig, ZImageTransformer2DModel,
};
use candle_transformers::quantized_var_builder;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::sync::Mutex;
use std::time::Instant;

use super::gguf_dense::load_gguf_dense_transformer;
use super::transformer::ZImageTransformer;
use crate::cache::{
    clear_cache, get_or_insert_cached_tensor, prompt_text_key, restore_cached_tensor, CachedTensor,
    LruCache, DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{
    check_memory_budget, fmt_gb, free_vram_bytes, memory_status_string, preflight_memory_check,
    should_use_gpu,
};
// Re-exported for tests (test harness is disabled via `test = false` in Cargo.toml,
// but tests reference this constant via `super::*`).
#[cfg(test)]
use crate::device::QWEN3_FP16_VRAM_THRESHOLD;
use crate::encoders;
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image};
use crate::img_utils;
use crate::progress::{ProgressCallback, ProgressEvent, ProgressReporter};

/// Minimum free VRAM (bytes) required to place Z-Image VAE on GPU.
/// The VAE itself is small (~160MB), but decode at 1024x1024 needs ~6GB workspace
/// for conv2d im2col expansions through the upsampling blocks.
const VAE_DECODE_VRAM_THRESHOLD: u64 = 6_500_000_000;

/// Z-Image scheduler shift constants from the reference implementation.
const BASE_IMAGE_SEQ_LEN: usize = 256;
const MAX_IMAGE_SEQ_LEN: usize = 4096;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

fn build_zimage_scheduler(
    num_steps: usize,
    image_seq_len: usize,
    strength: Option<f64>,
) -> (FlowMatchEulerDiscreteScheduler, usize) {
    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
    let mu = calculate_shift(
        image_seq_len,
        BASE_IMAGE_SEQ_LEN,
        MAX_IMAGE_SEQ_LEN,
        BASE_SHIFT,
        MAX_SHIFT,
    );
    let sigmas: Vec<f64> = (0..=num_steps)
        .map(|v| v as f64 / num_steps as f64)
        .rev()
        .map(|t| {
            if !(0.0..1.0).contains(&t) {
                t
            } else {
                let e_mu = mu.exp();
                e_mu / (e_mu + (1.0 / t - 1.0))
            }
        })
        .collect();
    scheduler.timesteps = sigmas[..sigmas.len().saturating_sub(1)]
        .iter()
        .map(|sigma| sigma * scheduler.config.num_train_timesteps as f64)
        .collect();
    scheduler.sigmas = sigmas;
    let start_index = strength
        .map(|strength| crate::img2img::img2img_start_index(num_steps, strength))
        .unwrap_or(0);
    if start_index > 0 {
        scheduler.timesteps = scheduler.timesteps[start_index..].to_vec();
        scheduler.sigmas = scheduler.sigmas[start_index..].to_vec();
    }
    scheduler.reset();
    (scheduler, start_index)
}

fn model_timestep(scheduler: &FlowMatchEulerDiscreteScheduler) -> f64 {
    1.0 - scheduler.current_sigma()
}

fn zimage_debug_enabled() -> bool {
    std::env::var_os("MOLD_ZIMAGE_DEBUG").is_some()
}

fn tensor_stats_summary(name: &str, tensor: &Tensor) -> Result<String> {
    let flat = tensor.to_dtype(DType::F32)?.flatten_all()?;
    let mean = flat.mean_all()?.to_scalar::<f32>()?;
    let min = flat.min(0)?.to_scalar::<f32>()?;
    let max = flat.max(0)?.to_scalar::<f32>()?;
    let rms = flat.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
    Ok(format!(
        "{name}: mean={mean:.5} min={min:.5} max={max:.5} rms={rms:.5}"
    ))
}

/// Loaded Z-Image model components, ready for inference.
struct LoadedZImage {
    /// Transformer is wrapped in Option so it can be dropped to free VRAM for VAE decode,
    /// then reloaded from disk for the next generation (similar to FLUX's T5/CLIP offload).
    transformer: Option<ZImageTransformer>,
    text_encoder: encoders::qwen3::Qwen3Encoder,
    vae: AutoEncoderKL,
    transformer_cfg: Config,
    /// GPU device for transformer + denoising
    device: Device,
    /// Device where the VAE lives (may be CPU if VRAM is extremely tight)
    vae_device: Device,
    dtype: DType,
    /// Whether the transformer source file is GGUF (needed for reload/logging).
    is_gguf: bool,
    /// Path to the VAE safetensors file (needed for CPU fallback reload on OOM).
    vae_path: std::path::PathBuf,
}

/// Z-Image inference engine backed by candle's z_image module.
pub struct ZImageEngine {
    base: EngineBase<LoadedZImage>,
    /// Qwen3 variant preference: None/"auto" = VRAM-based, "bf16" = force BF16, "q8"/etc = specific.
    qwen3_variant: Option<String>,
    prompt_cache: Mutex<LruCache<String, CachedTensor>>,
    /// Per-request placement override.
    pending_placement: Option<mold_core::types::DevicePlacement>,
}

impl ZImageEngine {
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        qwen3_variant: Option<String>,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy, gpu_ordinal),
            qwen3_variant,
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            pending_placement: None,
        }
    }

    fn encode_prompt_cached(
        progress: &ProgressReporter,
        prompt_cache: &Mutex<LruCache<String, CachedTensor>>,
        encoder: &mut encoders::qwen3::Qwen3Encoder,
        prompt: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let cache_key = prompt_text_key(prompt);
        let (cap_feats, cache_hit) =
            get_or_insert_cached_tensor(prompt_cache, cache_key, device, dtype, || {
                progress.stage_start("Encoding prompt (Qwen3)");
                let encode_start = Instant::now();
                let (cap_feats, _token_count) = encoder.encode(prompt, device, dtype)?;
                progress.stage_done("Encoding prompt (Qwen3)", encode_start.elapsed());
                Ok(cap_feats)
            })?;
        if cache_hit {
            progress.cache_hit("prompt conditioning");
        }
        let token_count = cap_feats.dim(1)?;
        let cap_mask = Tensor::ones((1, token_count), DType::U8, device)?;
        Ok((cap_feats, cap_mask))
    }

    /// Resolve transformer shard paths: use `transformer_shards` if non-empty,
    /// otherwise treat `transformer` as a single file.
    fn transformer_paths(&self) -> Vec<std::path::PathBuf> {
        if !self.base.paths.transformer_shards.is_empty() {
            self.base.paths.transformer_shards.clone()
        } else {
            vec![self.base.paths.transformer.clone()]
        }
    }

    /// Detect if the transformer is GGUF quantized.
    fn detect_is_gguf(&self) -> bool {
        self.base
            .paths
            .transformer
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    }

    /// Validate tokenizer path and transformer/VAE paths exist.
    fn validate_paths(&self) -> Result<std::path::PathBuf> {
        let text_tokenizer_path =
            self.base.paths.text_tokenizer.as_ref().ok_or_else(|| {
                anyhow::anyhow!("text tokenizer path required for Z-Image models")
            })?;
        if !text_tokenizer_path.exists() {
            bail!(
                "text tokenizer file not found: {}",
                text_tokenizer_path.display()
            );
        }

        let xformer_paths = self.transformer_paths();
        for path in &xformer_paths {
            if !path.exists() {
                bail!("transformer file not found: {}", path.display());
            }
        }
        if !self.base.paths.vae.exists() {
            bail!("VAE file not found: {}", self.base.paths.vae.display());
        }

        Ok(text_tokenizer_path.clone())
    }

    /// Load transformer from disk.
    fn load_transformer(
        &self,
        device: &Device,
        dtype: DType,
        cfg: &Config,
    ) -> Result<ZImageTransformer> {
        let is_gguf = self.detect_is_gguf();
        let xformer_paths = self.transformer_paths();

        if is_gguf {
            let vb =
                quantized_var_builder::VarBuilder::from_gguf(&self.base.paths.transformer, device)?;
            Ok(ZImageTransformer::Dense(load_gguf_dense_transformer(
                cfg, dtype, vb,
            )?))
        } else {
            let xformer_vb = crate::weight_loader::load_safetensors_with_progress(
                &xformer_paths,
                dtype,
                device,
                "Z-Image transformer",
                &self.base.progress,
            )?;
            Ok(ZImageTransformer::Dense(ZImageTransformer2DModel::new(
                cfg, xformer_vb,
            )?))
        }
    }

    /// Load VAE from disk.
    fn load_vae(&self, device: &Device, dtype: DType) -> Result<AutoEncoderKL> {
        let vae_cfg = VaeConfig::z_image();
        let vae_vb = crate::weight_loader::load_safetensors_with_progress(
            &[self.base.paths.vae.as_path()],
            dtype,
            device,
            "VAE",
            &self.base.progress,
        )?;
        Ok(AutoEncoderKL::new(&vae_cfg, vae_vb)?)
    }

    /// Load all model components (Eager mode).
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

        tracing::info!(model = %self.base.model_name, "loading Z-Image model components...");

        let is_gguf = self.detect_is_gguf();
        let text_tokenizer_path = self.validate_paths()?;

        let device = crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)?;
        let dtype = crate::engine::gpu_dtype(&device);
        let transformer_cfg = Config::z_image_turbo();

        // Load transformer
        let xformer_label = if is_gguf {
            "Loading Z-Image transformer (GPU, GGUF -> dense)".to_string()
        } else {
            let xformer_paths = self.transformer_paths();
            format!(
                "Loading Z-Image transformer ({} shards)",
                xformer_paths.len()
            )
        };
        self.base.progress.stage_start(&xformer_label);
        let xformer_start = Instant::now();

        let transformer = self.load_transformer(&device, dtype, &transformer_cfg)?;

        self.base
            .progress
            .stage_done(&xformer_label, xformer_start.elapsed());
        tracing::info!(quantized = is_gguf, "Z-Image transformer loaded");

        // --- Decide where to place VAE and Qwen3 text encoder based on remaining VRAM ---
        let free = free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let is_cuda = device.is_cuda();
        let is_metal = device.is_metal();
        if free > 0 {
            self.base
                .progress
                .info(&format!("Free VRAM after transformer: {}", fmt_gb(free)));
            tracing::info!(free_vram = free, "free VRAM after loading transformer");
        }

        // VAE decode at 1024x1024 needs ~6GB workspace for conv2d im2col.
        // On tight VRAM, load VAE on CPU to guarantee decode succeeds.
        let vae_on_gpu = should_use_gpu(is_cuda, is_metal, free, VAE_DECODE_VRAM_THRESHOLD);
        let vae_device = if vae_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let vae_dtype = if vae_on_gpu { dtype } else { DType::F32 };
        let vae_device_label = if vae_on_gpu { "GPU" } else { "CPU" };

        if !vae_on_gpu && (is_cuda || is_metal) {
            self.base.progress.info(&format!(
                "VAE on CPU ({} free < {} threshold for decode workspace)",
                fmt_gb(free),
                fmt_gb(VAE_DECODE_VRAM_THRESHOLD),
            ));
        }

        // Load VAE
        let vae_label = format!("Loading VAE ({})", vae_device_label);
        self.base.progress.stage_start(&vae_label);
        let vae_start = Instant::now();
        let vae = self.load_vae(&vae_device, vae_dtype)?;
        self.base
            .progress
            .stage_done(&vae_label, vae_start.elapsed());
        tracing::info!(device = vae_device_label, "Z-Image VAE loaded");

        // --- Qwen3 text encoder: auto-select variant based on VRAM ---
        self.base.progress.stage_start("Selecting Qwen3 encoder");
        let qwen3_resolve_start = Instant::now();
        let qwen3_preference = self.qwen3_variant.as_deref();
        let (resolved_paths, is_qwen3_gguf, te_on_gpu, _te_auto_device_label) = {
            let bf16_paths = self.base.paths.text_encoder_files.clone();
            let have_bf16 = !bf16_paths.is_empty() && bf16_paths.iter().all(|p| p.exists());
            crate::encoders::variant_resolution::resolve_qwen3_variant(
                &self.base.progress,
                qwen3_preference,
                &device,
                free,
                &bf16_paths,
                have_bf16,
                false,
                crate::encoders::variant_resolution::Qwen3Size::B4,
            )?
        };
        self.base
            .progress
            .stage_done("Selecting Qwen3 encoder", qwen3_resolve_start.elapsed());

        let tier1 = self
            .pending_placement
            .as_ref()
            .map(|p| p.text_encoders)
            .unwrap_or_default();
        let auto_te_device = if te_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let te_device = crate::device::resolve_device(
            Some(tier1),
            || Ok(auto_te_device.clone()),
        )?;
        let te_on_gpu = !te_device.is_cpu();
        let te_device_label = if te_on_gpu { "GPU" } else { "CPU" };
        let te_dtype = if te_on_gpu { dtype } else { DType::F32 };

        // Load text encoder
        let bf16_cfg = encoders::qwen3_bf16::Qwen3BF16Config::qwen3_4b();
        let te_label = if is_qwen3_gguf {
            format!("Loading Qwen3 text encoder (GGUF, {})", te_device_label)
        } else {
            format!(
                "Loading Qwen3 text encoder ({} shards, {})",
                resolved_paths.len(),
                te_device_label,
            )
        };
        self.base.progress.stage_start(&te_label);
        let te_start = Instant::now();

        let text_encoder = if is_qwen3_gguf {
            encoders::qwen3::Qwen3Encoder::load_gguf(
                &resolved_paths[0],
                &text_tokenizer_path,
                &te_device,
                &bf16_cfg,
            )?
        } else {
            encoders::qwen3::Qwen3Encoder::load_bf16(
                &resolved_paths,
                &text_tokenizer_path,
                &te_device,
                te_dtype,
                &bf16_cfg,
                &self.base.progress,
            )?
        };

        self.base.progress.stage_done(&te_label, te_start.elapsed());
        tracing::info!(device = %te_device_label, quantized = is_qwen3_gguf, "Qwen3 text encoder loaded");

        self.base.loaded = Some(LoadedZImage {
            transformer: Some(transformer),
            text_encoder,
            vae,
            transformer_cfg,
            device,
            vae_device,
            dtype,
            is_gguf,
            vae_path: self.base.paths.vae.clone(),
        });

        tracing::info!(model = %self.base.model_name, "all Z-Image components loaded successfully");
        Ok(())
    }

    /// Reload the transformer from disk (called when it was dropped to free VRAM for VAE decode).
    fn reload_transformer(&self, loaded: &mut LoadedZImage) -> Result<()> {
        let transformer =
            self.load_transformer(&loaded.device, loaded.dtype, &loaded.transformer_cfg)?;
        loaded.transformer = Some(transformer);
        Ok(())
    }

    /// Generate an image using sequential loading strategy.
    ///
    /// Loads components one at a time and drops them when done:
    /// 1. Load Qwen3 → encode → drop Qwen3
    /// 2. Load transformer → denoise → drop transformer
    /// 3. Load VAE → decode → drop VAE
    ///
    /// Peak memory: max(Qwen3_size, transformer_size) instead of sum(all).
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let text_tokenizer_path = self.validate_paths()?;
        let is_gguf = self.detect_is_gguf();
        let transformer_cfg = Config::z_image_turbo();

        // Check memory budget
        if let Some(warning) = check_memory_budget(&self.base.paths, LoadStrategy::Sequential) {
            self.base.progress.info(&warning);
        }

        let device = crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)?;
        let dtype = crate::engine::gpu_dtype(&device);

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting sequential Z-Image generation"
        );

        self.base
            .progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: Qwen3 text encoding (check cache first to skip encoder load) ---
        let cache_key = prompt_text_key(&req.prompt);
        let (cap_feats, cap_mask) = if let Some(cap_feats) =
            restore_cached_tensor(&self.prompt_cache, &cache_key, &device, dtype)?
        {
            self.base.progress.cache_hit("prompt conditioning");
            let token_count = cap_feats.dim(1)?;
            let cap_mask = Tensor::ones((1, token_count), DType::U8, &device)?;
            (cap_feats, cap_mask)
        } else {
            let free = free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
            self.base.progress.stage_start("Selecting Qwen3 encoder");
            let qwen3_resolve_start = Instant::now();
            let qwen3_preference = self.qwen3_variant.as_deref();
            let (resolved_paths, is_qwen3_gguf, te_on_gpu, _te_auto_device_label) = {
                let bf16_paths = self.base.paths.text_encoder_files.clone();
                let have_bf16 = !bf16_paths.is_empty() && bf16_paths.iter().all(|p| p.exists());
                crate::encoders::variant_resolution::resolve_qwen3_variant(
                    &self.base.progress,
                    qwen3_preference,
                    &device,
                    free,
                    &bf16_paths,
                    have_bf16,
                    false,
                    crate::encoders::variant_resolution::Qwen3Size::B4,
                )?
            };
            self.base
                .progress
                .stage_done("Selecting Qwen3 encoder", qwen3_resolve_start.elapsed());

            let tier1 = self
                .pending_placement
                .as_ref()
                .map(|p| p.text_encoders)
                .unwrap_or_default();
            let auto_te_device = if te_on_gpu {
                device.clone()
            } else {
                Device::Cpu
            };
            let te_device = crate::device::resolve_device(
                Some(tier1),
                || Ok(auto_te_device.clone()),
            )?;
            let te_on_gpu = !te_device.is_cpu();
            let te_device_label = if te_on_gpu { "GPU" } else { "CPU" };
            let te_dtype = if te_on_gpu { dtype } else { DType::F32 };

            let bf16_cfg = encoders::qwen3_bf16::Qwen3BF16Config::qwen3_4b();
            let te_label = if is_qwen3_gguf {
                format!("Loading Qwen3 text encoder (GGUF, {})", te_device_label)
            } else {
                format!(
                    "Loading Qwen3 text encoder ({} shards, {})",
                    resolved_paths.len(),
                    te_device_label,
                )
            };
            let te_size: u64 = resolved_paths
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .sum();
            preflight_memory_check("Qwen3 text encoder", te_size)?;

            if let Some(status) = memory_status_string() {
                self.base.progress.info(&status);
            }

            self.base.progress.stage_start(&te_label);
            let te_start = Instant::now();

            let mut text_encoder = if is_qwen3_gguf {
                encoders::qwen3::Qwen3Encoder::load_gguf(
                    &resolved_paths[0],
                    &text_tokenizer_path,
                    &te_device,
                    &bf16_cfg,
                )?
            } else {
                encoders::qwen3::Qwen3Encoder::load_bf16(
                    &resolved_paths,
                    &text_tokenizer_path,
                    &te_device,
                    te_dtype,
                    &bf16_cfg,
                    &self.base.progress,
                )?
            };
            self.base.progress.stage_done(&te_label, te_start.elapsed());

            let (cap_feats, cap_mask) = Self::encode_prompt_cached(
                &self.base.progress,
                &self.prompt_cache,
                &mut text_encoder,
                &req.prompt,
                &device,
                dtype,
            )?;

            drop(text_encoder);
            self.base.progress.info("Freed Qwen3 text encoder");
            tracing::info!("Qwen3 text encoder dropped (sequential mode)");

            (cap_feats, cap_mask)
        };

        // Calculate latent dimensions up front so img2img can encode the source image
        // before the transformer is loaded. This keeps the encode path on GPU and
        // avoids the multi-minute CPU fallback.
        let vae_align = 16;
        let latent_h = 2 * (height / vae_align);
        let latent_w = 2 * (width / vae_align);

        let patch_size = transformer_cfg.all_patch_size[0];
        let image_seq_len = (latent_h / patch_size) * (latent_w / patch_size);
        let (mut scheduler, start_index) = build_zimage_scheduler(
            req.steps as usize,
            image_seq_len,
            req.source_image.as_ref().map(|_| req.strength),
        );

        if req.source_image.is_some() {
            tracing::info!(
                strength = req.strength,
                start_index,
                start_sigma = scheduler.sigmas[0],
                remaining_sigmas = scheduler.sigmas.len(),
                remaining_steps = scheduler.sigmas.len().saturating_sub(1),
                "img2img: truncated schedule from strength"
            );
        }

        // --- Phase 2: Build initial latents ---
        let (mut latents, inpaint_ctx) = if let Some(ref source_bytes) = req.source_image {
            let start_sigma = scheduler.sigmas[0];

            // Encode before loading the transformer so we can keep the VAE on GPU.
            let encode_vae_device = if device.is_cuda() || device.is_metal() {
                device.clone()
            } else {
                Device::Cpu
            };
            let encode_vae_dtype = if encode_vae_device.is_cpu() {
                DType::F32
            } else {
                dtype
            };
            let encode_label = if encode_vae_device.is_cpu() {
                "Loading VAE for source encoding (CPU)"
            } else {
                "Loading VAE for source encoding (GPU)"
            };

            self.base.progress.stage_start(encode_label);
            let vae_enc_start = Instant::now();
            let encode_vae = self.load_vae(&encode_vae_device, encode_vae_dtype)?;
            self.base
                .progress
                .stage_done(encode_label, vae_enc_start.elapsed());

            self.base
                .progress
                .stage_start("Encoding source image (VAE)");
            let encode_start = Instant::now();
            let source_tensor = img_utils::decode_source_image(
                source_bytes,
                req.width,
                req.height,
                img_utils::NormalizeRange::MinusOneToOne,
                &encode_vae_device,
                encode_vae_dtype,
            )?;
            let encoded = encode_vae.encode(&source_tensor)?;
            self.base
                .progress
                .stage_done("Encoding source image (VAE)", encode_start.elapsed());

            // Drop encoding VAE before loading transformer
            drop(encode_vae);

            // Generate noise on the target device
            let encoded = encoded.to_dtype(dtype)?.to_device(&device)?;
            let prepared = crate::img2img::prepare_flow_match_img2img(
                &encoded,
                seed,
                &[1, 16, latent_h, latent_w],
                start_sigma,
                req.mask_image.as_deref(),
                latent_h,
                latent_w,
                &device,
                dtype,
            )?;
            // Add frame dimension: (B, C, H, W) -> (B, C, 1, H, W)
            (prepared.initial_latents.unsqueeze(2)?, prepared.inpaint_ctx)
        } else {
            // txt2img: pure noise
            let noise =
                crate::engine::seeded_randn(seed, &[1, 16, latent_h, latent_w], &device, dtype)?;
            (noise.unsqueeze(2)?, None)
        };

        // --- Phase 3: Load transformer and denoise ---
        let xformer_paths = self.transformer_paths();
        let xformer_size: u64 = xformer_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        preflight_memory_check("Z-Image transformer", xformer_size)?;

        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        let xformer_label = if is_gguf {
            "Loading Z-Image transformer (GPU, GGUF -> dense)".to_string()
        } else {
            format!(
                "Loading Z-Image transformer ({} shards)",
                xformer_paths.len()
            )
        };
        self.base.progress.stage_start(&xformer_label);
        let xformer_start = Instant::now();
        let transformer = self.load_transformer(&device, dtype, &transformer_cfg)?;
        self.base
            .progress
            .stage_done(&xformer_label, xformer_start.elapsed());

        let num_steps = scheduler.sigmas.len().saturating_sub(1);
        let denoise_label = format!("Denoising ({} steps)", num_steps);
        self.base.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        for step in 0..num_steps {
            let step_start = Instant::now();
            let t = model_timestep(&scheduler);
            let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &device)?.to_dtype(dtype)?;
            if zimage_debug_enabled() {
                tracing::debug!(
                    step = step + 1,
                    total = num_steps,
                    sigma = scheduler.current_sigma(),
                    timestep = t,
                    "{}",
                    tensor_stats_summary("latents_in", &latents)?
                );
            }
            let noise_pred = transformer.forward(&latents, &t_tensor, &cap_feats, &cap_mask)?;
            if zimage_debug_enabled() {
                tracing::debug!(
                    step = step + 1,
                    total = num_steps,
                    "{}",
                    tensor_stats_summary("noise_pred_raw", &noise_pred)?
                );
            }
            let noise_pred = noise_pred.neg()?;
            let noise_pred_4d = noise_pred.squeeze(2)?;
            let latents_4d = latents.squeeze(2)?;
            let prev_latents = scheduler.step(&noise_pred_4d, &latents_4d)?;
            latents = prev_latents.unsqueeze(2)?;
            if zimage_debug_enabled() {
                tracing::debug!(
                    step = step + 1,
                    total = num_steps,
                    sigma_next = scheduler.current_sigma(),
                    "{}",
                    tensor_stats_summary("latents_out", &latents)?
                );
            }

            // Inpainting: blend preserved regions back at current noise level
            if let Some(ref ctx) = inpaint_ctx {
                let latents_4d = latents.squeeze(2)?;
                let blended = crate::img2img::apply_flow_match_inpaint(
                    &latents_4d,
                    ctx,
                    scheduler.sigmas[step + 1],
                )?;
                latents = blended.unsqueeze(2)?;
            }

            self.base.progress.emit(ProgressEvent::DenoiseStep {
                step: step + 1,
                total: num_steps,
                elapsed: step_start.elapsed(),
            });
        }

        self.base
            .progress
            .stage_done(&denoise_label, denoise_start.elapsed());

        // Drop transformer and text embeddings to free memory for VAE decode
        drop(transformer);
        self.base.progress.info("Freed Z-Image transformer");
        drop(cap_feats);
        drop(cap_mask);
        device.synchronize()?;
        tracing::info!("Transformer dropped (sequential mode)");

        // --- Phase 3: Load VAE and decode ---
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }
        // With sequential loading, we can always try GPU for VAE since transformer is freed
        let free_for_vae = free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let vae_on_gpu = should_use_gpu(
            device.is_cuda(),
            device.is_metal(),
            free_for_vae,
            VAE_DECODE_VRAM_THRESHOLD,
        );
        let vae_device = if vae_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let vae_dtype = if vae_on_gpu { dtype } else { DType::F32 };
        let vae_device_label = if vae_on_gpu { "GPU" } else { "CPU" };

        let vae_label = format!("Loading VAE ({})", vae_device_label);
        self.base.progress.stage_start(&vae_label);
        let vae_start = Instant::now();
        let vae = self.load_vae(&vae_device, vae_dtype)?;
        self.base
            .progress
            .stage_done(&vae_label, vae_start.elapsed());

        self.base.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();

        let latents = latents
            .squeeze(2)?
            .to_device(&vae_device)?
            .to_dtype(vae_dtype)?;
        let image = vae.decode(&latents)?;
        let image = postprocess_image(&image)?;
        let image = image.i(0)?;

        self.base
            .progress
            .stage_done("VAE decode", vae_decode_start.elapsed());

        // VAE dropped here
        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &image,
            req.output_format,
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            generation_time_ms,
            seed,
            "sequential Z-Image generation complete"
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

impl ZImageEngine {
    fn generate_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if req.scheduler.is_some() {
            tracing::warn!(
                "scheduler selection not supported for Z-Image (flow-matching), ignoring"
            );
        }
        // Sequential mode: load-use-drop each component
        if self.base.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components
        if self.base.loaded.is_none() {
            bail!("model not loaded — call load() first");
        }

        // Borrow progress reporter separately from loaded state.
        let progress = &self.base.progress;

        let start = Instant::now();

        // Reload transformer if it was dropped (offloaded) after previous VAE decode
        let loaded_ref = self
            .base
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
        let needs_reload = loaded_ref.transformer.is_none();
        if needs_reload {
            {
                let mut loaded_mut = self
                    .base
                    .loaded
                    .take()
                    .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
                let xformer_label = if loaded_mut.is_gguf {
                    "Reloading Z-Image transformer (GPU, GGUF -> dense)"
                } else {
                    "Reloading Z-Image transformer (GPU, BF16)"
                };
                progress.stage_start(xformer_label);
                let reload_start = Instant::now();
                self.reload_transformer(&mut loaded_mut)?;
                progress.stage_done(xformer_label, reload_start.elapsed());
                self.base.loaded = Some(loaded_mut);
            }
        }

        let loaded = self
            .base
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting Z-Image generation"
        );

        // 1. Encode prompt with Qwen3 (check cache first to avoid unnecessary reload)
        let cache_key = prompt_text_key(&req.prompt);
        let (cap_feats, cap_mask) = if let Some(cap_feats) =
            restore_cached_tensor(&self.prompt_cache, &cache_key, &loaded.device, loaded.dtype)?
        {
            progress.cache_hit("prompt conditioning");
            let token_count = cap_feats.dim(1)?;
            let cap_mask = Tensor::ones((1, token_count), DType::U8, &loaded.device)?;
            (cap_feats, cap_mask)
        } else {
            // Cache miss — reload encoder if it was dropped after a previous generation
            if loaded.text_encoder.model.is_none() {
                let te_label = if loaded.text_encoder.is_quantized {
                    "Reloading Qwen3 encoder (GGUF)"
                } else {
                    "Reloading Qwen3 encoder (BF16)"
                };
                progress.stage_start(te_label);
                let reload_start = Instant::now();
                loaded.text_encoder.reload(progress)?;
                progress.stage_done(te_label, reload_start.elapsed());
            }

            let (cap_feats, cap_mask) = Self::encode_prompt_cached(
                progress,
                &self.prompt_cache,
                &mut loaded.text_encoder,
                &req.prompt,
                &loaded.device,
                loaded.dtype,
            )?;
            tracing::info!(token_count = cap_feats.dim(1)?, "text encoding complete");

            // Drop text encoder to free memory for denoising + VAE decode.
            // Always drop on GPU. On Metal (unified memory), also drop CPU-loaded
            // weights since they share the same physical RAM as GPU allocations.
            // On CUDA, keep CPU-loaded weights resident to avoid expensive reloads.
            if loaded.text_encoder.on_gpu || loaded.device.is_metal() {
                loaded.text_encoder.drop_weights();
                tracing::info!(
                    on_gpu = loaded.text_encoder.on_gpu,
                    "Qwen3 text encoder dropped to free memory for denoising"
                );
            }

            (cap_feats, cap_mask)
        };

        // 3. Calculate latent dimensions: 2 * (image_size / 16)
        let vae_align = 16;
        let latent_h = 2 * (height / vae_align);
        let latent_w = 2 * (width / vae_align);

        // 5. Initialize scheduler
        let patch_size = loaded.transformer_cfg.all_patch_size[0];
        let image_seq_len = (latent_h / patch_size) * (latent_w / patch_size);
        let (mut scheduler, start_index) = build_zimage_scheduler(
            req.steps as usize,
            image_seq_len,
            req.source_image.as_ref().map(|_| req.strength),
        );

        if req.source_image.is_some() {
            tracing::info!(
                strength = req.strength,
                start_index,
                start_sigma = scheduler.sigmas[0],
                remaining_sigmas = scheduler.sigmas.len(),
                remaining_steps = scheduler.sigmas.len().saturating_sub(1),
                "img2img: truncated schedule from strength"
            );
        }

        // 6. Build initial latents — img2img encodes source image, txt2img uses pure noise
        let (mut latents, inpaint_ctx) = if let Some(ref source_bytes) = req.source_image {
            let start_sigma = scheduler.sigmas[0];

            // Encode source image through the pre-loaded VAE
            progress.stage_start("Encoding source image (VAE)");
            let encode_start = Instant::now();
            let vae_encode_device = &loaded.vae_device;
            let vae_encode_dtype = if loaded.vae_device.is_cpu() {
                DType::F32
            } else {
                loaded.dtype
            };
            let source_tensor = img_utils::decode_source_image(
                source_bytes,
                req.width,
                req.height,
                img_utils::NormalizeRange::MinusOneToOne,
                vae_encode_device,
                vae_encode_dtype,
            )?;
            let encoded = loaded.vae.encode(&source_tensor)?;
            progress.stage_done("Encoding source image (VAE)", encode_start.elapsed());

            let encoded = encoded.to_dtype(loaded.dtype)?.to_device(&loaded.device)?;

            let prepared = crate::img2img::prepare_flow_match_img2img(
                &encoded,
                seed,
                &[1, 16, latent_h, latent_w],
                start_sigma,
                req.mask_image.as_deref(),
                latent_h,
                latent_w,
                &loaded.device,
                loaded.dtype,
            )?;
            (prepared.initial_latents.unsqueeze(2)?, prepared.inpaint_ctx)
        } else {
            // txt2img: pure noise (B, 16, latent_h, latent_w) → add frame dim
            let noise = crate::engine::seeded_randn(
                seed,
                &[1, 16, latent_h, latent_w],
                &loaded.device,
                loaded.dtype,
            )?;
            (noise.unsqueeze(2)?, None)
        };

        // 7. Denoising loop
        let num_steps = scheduler.sigmas.len().saturating_sub(1);
        let denoise_label = format!("Denoising ({} steps)", num_steps);
        progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        // Scope the transformer borrow so it can be dropped before VAE decode
        {
            let transformer = loaded
                .transformer
                .as_ref()
                .expect("transformer must be loaded for denoising");

            for step in 0..num_steps {
                let step_start = Instant::now();
                let t = model_timestep(&scheduler);
                let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &loaded.device)?
                    .to_dtype(loaded.dtype)?;
                if zimage_debug_enabled() {
                    tracing::debug!(
                        step = step + 1,
                        total = num_steps,
                        sigma = scheduler.current_sigma(),
                        timestep = t,
                        "{}",
                        tensor_stats_summary("latents_in", &latents)?
                    );
                }

                // Forward pass through transformer
                let noise_pred = transformer.forward(&latents, &t_tensor, &cap_feats, &cap_mask)?;
                if zimage_debug_enabled() {
                    tracing::debug!(
                        step = step + 1,
                        total = num_steps,
                        "{}",
                        tensor_stats_summary("noise_pred_raw", &noise_pred)?
                    );
                }

                // Negate prediction (Z-Image specific)
                let noise_pred = noise_pred.neg()?;

                // Remove frame dimension for scheduler: (B, C, 1, H, W) → (B, C, H, W)
                let noise_pred_4d = noise_pred.squeeze(2)?;
                let latents_4d = latents.squeeze(2)?;

                // Scheduler step
                let prev_latents = scheduler.step(&noise_pred_4d, &latents_4d)?;

                // Add back frame dimension
                latents = prev_latents.unsqueeze(2)?;
                if zimage_debug_enabled() {
                    tracing::debug!(
                        step = step + 1,
                        total = num_steps,
                        sigma_next = scheduler.current_sigma(),
                        "{}",
                        tensor_stats_summary("latents_out", &latents)?
                    );
                }

                // Inpainting: blend preserved regions back at current noise level
                if let Some(ref ctx) = inpaint_ctx {
                    let latents_4d = latents.squeeze(2)?;
                    let blended = crate::img2img::apply_flow_match_inpaint(
                        &latents_4d,
                        ctx,
                        scheduler.sigmas[step + 1],
                    )?;
                    latents = blended.unsqueeze(2)?;
                }

                progress.emit(ProgressEvent::DenoiseStep {
                    step: step + 1,
                    total: num_steps,
                    elapsed: step_start.elapsed(),
                });
            }
        }

        progress.stage_done(&denoise_label, denoise_start.elapsed());
        tracing::info!("denoising complete");

        // Free text embeddings — no longer needed after denoising
        drop(cap_feats);
        drop(cap_mask);

        // Drop the transformer weights from GPU to free VRAM for VAE decode.
        // The transformer (~6.6GB for Q8) is only needed during denoising.
        // It will be reloaded from disk on the next generate() call.
        loaded.transformer = None;
        // Synchronize to ensure CUDA's caching allocator reclaims the freed memory
        // before VAE decode allocates large im2col workspace buffers (~6GB at 1024x1024).
        loaded.device.synchronize()?;
        tracing::info!("Z-Image transformer dropped from GPU to free VRAM for VAE decode");

        // 8. VAE decode — try GPU first, fall back to CPU on OOM
        progress.stage_start("VAE decode");
        let vae_start = Instant::now();

        // Remove frame dimension: (B, C, 1, H, W) → (B, C, H, W)
        let latents_4d = latents.squeeze(2)?;

        // Try VAE decode on the pre-assigned device
        let image = {
            let decode_latents = latents_4d.to_device(&loaded.vae_device)?.to_dtype(
                if loaded.vae_device.is_cpu() {
                    DType::F32
                } else {
                    loaded.dtype
                },
            )?;
            match loaded.vae.decode(&decode_latents) {
                Ok(img) => img,
                Err(e) if loaded.vae_device.is_cuda() => {
                    // OOM on GPU — reload VAE on CPU and retry
                    let err_msg = format!("{e}");
                    if err_msg.contains("OUT_OF_MEMORY") || err_msg.contains("out of memory") {
                        tracing::warn!("VAE decode OOM on GPU, falling back to CPU");
                        progress.info("VAE decode OOM on GPU — retrying on CPU");
                        loaded.device.synchronize()?;
                        // Load a fresh VAE on CPU (can't call self.load_vae_cpu() due to borrow)
                        let vae_cfg = VaeConfig::z_image();
                        let vae_vb = crate::weight_loader::load_safetensors_with_progress(
                            &[loaded.vae_path.as_path()],
                            DType::F32,
                            &Device::Cpu,
                            "VAE",
                            progress,
                        )?;
                        let cpu_vae = AutoEncoderKL::new(&vae_cfg, vae_vb)?;
                        let cpu_latents =
                            latents_4d.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                        cpu_vae.decode(&cpu_latents)?
                    } else {
                        return Err(e.into());
                    }
                }
                Err(e) => return Err(e.into()),
            }
        };

        // Post-process: [-1, 1] → [0, 255] (candle z_image utility)
        let image = postprocess_image(&image)?;
        let image = image.i(0)?; // Remove batch dimension → [3, H, W]

        progress.stage_done("VAE decode", vae_start.elapsed());

        // 9. Encode to output format
        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &image,
            req.output_format,
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "Z-Image generation complete");

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

impl InferenceEngine for ZImageEngine {
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
        // Sequential mode is always "ready" — it loads on demand
        self.base.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        ZImageEngine::load(self)
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
    use crate::device::should_use_gpu;
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

    fn zimage_model_paths(
        transformer: PathBuf,
        transformer_shards: Vec<PathBuf>,
        vae: PathBuf,
        text_tokenizer: Option<PathBuf>,
    ) -> ModelPaths {
        ModelPaths {
            transformer,
            transformer_shards,
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![],
            text_tokenizer,
            decoder: None,
        }
    }

    #[test]
    fn latent_dimensions() {
        // 1024px → 2 * (1024 / 16) = 128
        assert_eq!(2 * (1024 / 16), 128);
        // 512px → 2 * (512 / 16) = 64
        assert_eq!(2 * (512 / 16), 64);
        // 768px → 2 * (768 / 16) = 96
        assert_eq!(2 * (768 / 16), 96);
    }

    // --- VRAM threshold decision tests (with drop-and-reload) ---

    #[test]
    fn qwen3_on_gpu_on_24gb_with_q8_drop_reload() {
        // Q8 transformer (6.6GB) on 24GB card → ~17GB free
        // With drop-and-reload, threshold is 10.2GB → fits on GPU!
        assert!(should_use_gpu(
            true,
            false,
            17_000_000_000,
            QWEN3_FP16_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn qwen3_on_gpu_on_24gb_with_q4_drop_reload() {
        // Q4 transformer (3.9GB) on 24GB card → ~19GB free
        // With drop-and-reload, easily fits on GPU
        assert!(should_use_gpu(
            true,
            false,
            19_000_000_000,
            QWEN3_FP16_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn qwen3_on_cpu_with_bf16_transformer() {
        // BF16 transformer (24.6GB) on 24GB card → ~0GB free
        // Even with drop-and-reload, can't fit
        assert!(!should_use_gpu(
            true,
            false,
            400_000_000,
            QWEN3_FP16_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn qwen3_on_gpu_on_48gb_card() {
        assert!(should_use_gpu(
            true,
            false,
            40_000_000_000,
            QWEN3_FP16_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn qwen3_on_gpu_on_metal() {
        // Metal with no memory info falls back to true
        assert!(should_use_gpu(false, true, 0, QWEN3_FP16_VRAM_THRESHOLD));
    }

    #[test]
    fn vae_on_gpu_when_plenty_of_vram() {
        assert!(should_use_gpu(
            true,
            false,
            17_000_000_000,
            VAE_DECODE_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn vae_on_cpu_when_vram_tight() {
        assert!(!should_use_gpu(
            true,
            false,
            5_400_000_000,
            VAE_DECODE_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn vae_on_gpu_on_metal() {
        // Metal with no memory info falls back to true
        assert!(should_use_gpu(false, true, 0, VAE_DECODE_VRAM_THRESHOLD));
    }

    // --- Threshold sanity checks ---

    #[test]
    fn qwen3_threshold_allows_gpu_on_24gb_with_quantized_xformer() {
        // Key improvement: with drop-and-reload, BF16 Qwen3 fits on GPU
        // when quantized transformer is used on 24GB cards
        let threshold = std::hint::black_box(QWEN3_FP16_VRAM_THRESHOLD);
        assert!(threshold < 17_000_000_000);
    }

    #[test]
    fn qwen3_threshold_exceeds_encoder_size() {
        let threshold = std::hint::black_box(QWEN3_FP16_VRAM_THRESHOLD);
        assert!(threshold > 8_200_000_000);
    }

    #[test]
    fn vae_threshold_accounts_for_decode_workspace() {
        let threshold = std::hint::black_box(VAE_DECODE_VRAM_THRESHOLD);
        assert!(threshold > 160_000_000);
        assert!(threshold < 15_000_000_000);
    }

    #[test]
    fn zimage_scheduler_uses_shifted_reference_sigmas() {
        let image_seq_len = 1024;
        let (full, _) = build_zimage_scheduler(9, image_seq_len, None);
        let (scheduler, start_index) = build_zimage_scheduler(9, image_seq_len, Some(0.5));
        let expected_sigmas = full.sigmas[start_index..].to_vec();
        let expected_timesteps = expected_sigmas[..expected_sigmas.len() - 1]
            .iter()
            .map(|sigma| sigma * 1000.0)
            .collect::<Vec<_>>();

        assert_eq!(start_index, crate::img2img::img2img_start_index(9, 0.5));
        assert_eq!(scheduler.sigmas, expected_sigmas);
        assert_eq!(scheduler.timesteps, expected_timesteps);
        assert_eq!(scheduler.sigmas.last().copied(), Some(0.0));
    }

    #[test]
    fn zimage_model_timestep_matches_scheduler_timesteps() {
        let (scheduler, _) = build_zimage_scheduler(9, 1024, Some(0.5));
        let t = model_timestep(&scheduler);
        assert!(
            (t - (1.0 - scheduler.sigmas[0])).abs() < 1e-10,
            "expected model timestep to match 1-sigma semantics, got {t} vs {}",
            1.0 - scheduler.sigmas[0]
        );
    }

    #[test]
    fn zimage_zero_strength_preserves_terminal_zero_only() {
        let (scheduler, start_index) = build_zimage_scheduler(9, 1024, Some(0.0));

        assert_eq!(start_index, 9);
        assert_eq!(scheduler.sigmas, vec![0.0]);
        assert!(scheduler.timesteps.is_empty());
    }

    #[test]
    fn tensor_stats_summary_reports_expected_values() {
        let tensor =
            Tensor::from_vec(vec![1.0f32, -1.0, 3.0, -3.0], (1, 1, 2, 2), &Device::Cpu).unwrap();
        let summary = tensor_stats_summary("probe", &tensor).unwrap();

        assert!(summary.contains("probe:"));
        assert!(summary.contains("mean=0.00000"));
        assert!(summary.contains("min=-3.00000"));
        assert!(summary.contains("max=3.00000"));
        assert!(summary.contains("rms=2.23607"));
    }

    #[test]
    fn zimage_transformer_paths_prefer_shards_when_present() {
        let dir = temp_test_dir("mold-zimage-shards");
        let shard_a = touch(&dir, "transformer-00001-of-00002.safetensors");
        let shard_b = touch(&dir, "transformer-00002-of-00002.safetensors");
        let engine = ZImageEngine::new(
            "z-image-turbo:bf16".to_string(),
            zimage_model_paths(
                dir.join("transformer.safetensors"),
                vec![shard_a.clone(), shard_b.clone()],
                dir.join("vae.safetensors"),
                Some(dir.join("tokenizer.json")),
            ),
            None,
            LoadStrategy::Sequential,
            0,
        );

        assert_eq!(engine.transformer_paths(), vec![shard_a, shard_b]);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn zimage_validate_paths_accepts_existing_files() {
        let dir = temp_test_dir("mold-zimage-validate-ok");
        let shard_a = touch(&dir, "transformer-00001-of-00002.safetensors");
        let shard_b = touch(&dir, "transformer-00002-of-00002.safetensors");
        let vae = touch(&dir, "vae.safetensors");
        let tokenizer = touch(&dir, "tokenizer.json");
        let gguf = touch(&dir, "transformer.gguf");

        let sharded = ZImageEngine::new(
            "z-image-turbo:bf16".to_string(),
            zimage_model_paths(
                dir.join("transformer.safetensors"),
                vec![shard_a, shard_b],
                vae.clone(),
                Some(tokenizer.clone()),
            ),
            None,
            LoadStrategy::Sequential,
            0,
        );
        assert_eq!(sharded.validate_paths().unwrap(), tokenizer);
        assert!(!sharded.detect_is_gguf());

        let quantized = ZImageEngine::new(
            "z-image-turbo:q4".to_string(),
            zimage_model_paths(gguf, vec![], vae, Some(dir.join("tokenizer.json"))),
            None,
            LoadStrategy::Sequential,
            0,
        );
        assert!(quantized.detect_is_gguf());

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn zimage_validate_paths_requires_text_tokenizer() {
        let dir = temp_test_dir("mold-zimage-validate-missing");
        let engine = ZImageEngine::new(
            "z-image-turbo:q4".to_string(),
            zimage_model_paths(
                dir.join("transformer.gguf"),
                vec![],
                dir.join("vae.safetensors"),
                None,
            ),
            None,
            LoadStrategy::Sequential,
            0,
        );

        let err = engine.validate_paths().unwrap_err();
        assert!(err.to_string().contains("text tokenizer path required"));

        fs::remove_dir_all(dir).ok();
    }
}
