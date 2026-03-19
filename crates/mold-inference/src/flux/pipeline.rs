use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::flux;
use candle_transformers::quantized_var_builder;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::time::Instant;

use crate::device::{
    check_memory_budget, fits_in_memory, fmt_gb, free_vram_bytes, memory_status_string,
    preflight_memory_check, should_use_gpu, t5_vram_threshold, CLIP_VRAM_THRESHOLD,
    T5_VRAM_THRESHOLD,
};
use crate::encoders;
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::image::encode_image;
use crate::progress::{ProgressCallback, ProgressReporter};

use super::transformer::FluxTransformer;

/// Loaded FLUX model components, ready for inference.
/// FLUX transformer and VAE always run on GPU. T5 and CLIP run on GPU or CPU
/// depending on available VRAM (checked at load time after the transformer is loaded).
/// When T5/CLIP are loaded on GPU, they are dropped after encoding to free VRAM
/// for the denoising pass (their weights are only needed for prompt encoding).
struct LoadedFlux {
    flux_model: FluxTransformer,
    t5: encoders::t5::T5Encoder,
    clip: encoders::clip::ClipEncoder,
    vae: flux::autoencoder::AutoEncoder,
    /// GPU device for FLUX transformer + VAE
    device: Device,
    dtype: DType,
    is_schnell: bool,
    /// True if using quantized GGUF model (state tensors must be F32)
    is_quantized: bool,
    /// The actual T5 encoder path used (may be a quantized GGUF, not the original FP16 path).
    t5_encoder_path: std::path::PathBuf,
}

/// FLUX inference engine backed by candle.
pub struct FluxEngine {
    loaded: Option<LoadedFlux>,
    model_name: String,
    paths: ModelPaths,
    /// Optional explicit override for is_schnell; if None, auto-detect from transformer filename.
    is_schnell_override: Option<bool>,
    progress: ProgressReporter,
    /// T5 variant preference: None/"auto" = auto-select, "fp16" = force FP16, "q8"/"q5"/etc = specific quantized.
    t5_variant: Option<String>,
    /// How to load model components (Eager = all at once, Sequential = load-use-drop).
    load_strategy: LoadStrategy,
}

impl FluxEngine {
    /// Create a new FluxEngine. Does not load models until `load()` is called.
    /// `is_schnell_override` lets callers explicitly set the scheduler family.
    /// `t5_variant` controls T5 encoder selection: None/"auto" = VRAM-based auto-select,
    /// "fp16" = force FP16, "q8"/"q5"/etc = specific quantized variant.
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        is_schnell_override: Option<bool>,
        t5_variant: Option<String>,
        load_strategy: LoadStrategy,
    ) -> Self {
        Self {
            loaded: None,
            model_name,
            paths,
            is_schnell_override,
            progress: ProgressReporter::default(),
            t5_variant,
            load_strategy,
        }
    }

    /// Resolve which T5 encoder to use and where to place it.
    /// Returns (encoder_path, on_gpu, device_label).
    /// `default_t5_path` is the FP16 T5 encoder path (already validated to exist).
    fn resolve_t5_variant(
        &self,
        preference: Option<&str>,
        gpu_device: &Device,
        _cpu_device: &Device,
        free_vram: u64,
        default_t5_path: &std::path::Path,
    ) -> Result<(std::path::PathBuf, bool, String)> {
        use mold_core::download::{cached_file_path, download_single_file_sync};
        use mold_core::manifest::{find_t5_variant, known_t5_variants, T5_FP16_SIZE};

        let is_cuda = gpu_device.is_cuda();
        let is_metal = gpu_device.is_metal();

        match preference {
            // Explicit quantized variant requested
            Some(tag) if tag != "fp16" && tag != "auto" => {
                let variant = find_t5_variant(tag).ok_or_else(|| {
                    anyhow::anyhow!(
                        "unknown T5 variant '{}'. Valid: fp16, auto, q8, q6, q5, q4, q3",
                        tag,
                    )
                })?;
                let path = self.resolve_t5_gguf_path(variant)?;
                let threshold = t5_vram_threshold(variant.size_bytes);
                let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, threshold);
                let label = if on_gpu {
                    "GPU, quantized"
                } else {
                    "CPU, quantized"
                };
                self.progress.info(&format!(
                    "Using T5 {} ({}) on {} (explicit)",
                    variant.tag,
                    fmt_gb(variant.size_bytes),
                    if on_gpu { "GPU" } else { "CPU" },
                ));
                Ok((path, on_gpu, label.to_string()))
            }

            // Explicit FP16 requested
            Some("fp16") => {
                let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, T5_VRAM_THRESHOLD);
                let label = if on_gpu { "GPU" } else { "CPU" };
                self.progress
                    .info(&format!("Using FP16 T5 on {} (explicit)", label));
                Ok((default_t5_path.to_path_buf(), on_gpu, label.to_string()))
            }

            // Auto mode (default): try FP16 on GPU, then quantized on GPU, then FP16 on CPU
            _ => {
                // Can FP16 T5 fit on GPU?
                if fits_in_memory(is_cuda, is_metal, free_vram, T5_VRAM_THRESHOLD) {
                    if is_metal {
                        self.progress
                            .info("Loading FP16 T5 on GPU (unified memory)");
                    } else {
                        self.progress.info(&format!(
                            "Loading FP16 T5 on GPU ({} free > {} threshold)",
                            fmt_gb(free_vram),
                            fmt_gb(T5_VRAM_THRESHOLD),
                        ));
                    }
                    return Ok((default_t5_path.to_path_buf(), true, "GPU".to_string()));
                }

                // FP16 won't fit on GPU — try quantized variants (largest first)
                if is_cuda || is_metal {
                    for variant in known_t5_variants() {
                        let threshold = t5_vram_threshold(variant.size_bytes);
                        if fits_in_memory(is_cuda, is_metal, free_vram, threshold) {
                            // Check cache first, download if needed
                            let path = match cached_file_path(
                                variant.hf_repo,
                                variant.hf_filename,
                                Some("shared/t5-gguf"),
                            ) {
                                Some(p) => p,
                                None => {
                                    self.progress.info(&format!(
                                        "Downloading T5 {} ({})...",
                                        variant.tag,
                                        fmt_gb(variant.size_bytes),
                                    ));
                                    tracing::info!(
                                        variant = variant.tag,
                                        repo = variant.hf_repo,
                                        file = variant.hf_filename,
                                        "downloading quantized T5 encoder"
                                    );
                                    download_single_file_sync(
                                        variant.hf_repo,
                                        variant.hf_filename,
                                        Some("shared/t5-gguf"),
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "failed to download T5 {}: {e}",
                                            variant.tag
                                        )
                                    })?
                                }
                            };
                            self.progress.info(&format!(
                                "FP16 T5 ({}) exceeds remaining VRAM ({}). Using quantized T5 {} ({}) on GPU instead.",
                                fmt_gb(T5_FP16_SIZE),
                                fmt_gb(free_vram),
                                variant.tag,
                                fmt_gb(variant.size_bytes),
                            ));
                            return Ok((path, true, format!("GPU, quantized {}", variant.tag)));
                        }
                    }
                }

                // On Metal, never fall back to CPU (same memory pool). Use smallest quantized variant.
                if is_metal {
                    let variants = known_t5_variants();
                    if let Some(smallest) = variants.last() {
                        let path = self.resolve_t5_gguf_path(smallest)?;
                        self.progress.info(&format!(
                            "Memory tight — using smallest T5 {} ({}) on GPU to reduce page pressure",
                            smallest.tag,
                            fmt_gb(smallest.size_bytes),
                        ));
                        return Ok((path, true, format!("GPU, quantized {}", smallest.tag)));
                    }
                }

                // No quantized variant fits on GPU either — fall back to FP16 on CPU
                if is_cuda || is_metal {
                    self.progress.info(&format!(
                        "Loading FP16 T5 on CPU ({} free, no variant fits on GPU)",
                        fmt_gb(free_vram),
                    ));
                } else {
                    self.progress.info("No GPU detected, loading T5 on CPU");
                }
                Ok((default_t5_path.to_path_buf(), false, "CPU".to_string()))
            }
        }
    }

    /// Resolve the path for a quantized T5 GGUF file: check cache, download if needed.
    fn resolve_t5_gguf_path(
        &self,
        variant: &mold_core::manifest::T5Variant,
    ) -> Result<std::path::PathBuf> {
        use mold_core::download::{cached_file_path, download_single_file_sync};

        if let Some(path) =
            cached_file_path(variant.hf_repo, variant.hf_filename, Some("shared/t5-gguf"))
        {
            return Ok(path);
        }
        self.progress.info(&format!(
            "Downloading T5 {} ({})...",
            variant.tag,
            fmt_gb(variant.size_bytes),
        ));
        download_single_file_sync(variant.hf_repo, variant.hf_filename, Some("shared/t5-gguf"))
            .map_err(|e| anyhow::anyhow!("failed to download T5 {}: {e}", variant.tag))
    }

    /// Detect is_schnell from override, model name, or transformer filename.
    fn detect_is_schnell(&self) -> bool {
        self.is_schnell_override.unwrap_or_else(|| {
            self.model_name.contains("schnell")
                || self
                    .paths
                    .transformer
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.contains("schnell"))
                    .unwrap_or(false)
        })
    }

    /// Detect if the transformer is quantized (GGUF).
    fn detect_is_quantized(&self) -> bool {
        self.paths
            .transformer
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    }

    /// Validate that all required paths exist.
    fn validate_paths(
        &self,
    ) -> Result<(
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
    )> {
        let t5_encoder_path = self
            .paths
            .t5_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("T5 encoder path required for FLUX models"))?
            .clone();
        let t5_tokenizer_path = self
            .paths
            .t5_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("T5 tokenizer path required for FLUX models"))?
            .clone();
        let clip_encoder_path = self
            .paths
            .clip_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP encoder path required for FLUX models"))?
            .clone();
        let clip_tokenizer_path = self
            .paths
            .clip_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP tokenizer path required for FLUX models"))?
            .clone();

        for (label, path) in [
            ("transformer", &self.paths.transformer),
            ("vae", &self.paths.vae),
            ("t5_encoder", &t5_encoder_path),
            ("clip_encoder", &clip_encoder_path),
            ("t5_tokenizer", &t5_tokenizer_path),
            ("clip_tokenizer", &clip_tokenizer_path),
        ] {
            if !path.exists() {
                bail!("{label} file not found: {}", path.display());
            }
        }

        Ok((
            t5_encoder_path,
            t5_tokenizer_path,
            clip_encoder_path,
            clip_tokenizer_path,
        ))
    }

    /// Load all model components into GPU memory (Eager mode).
    pub fn load(&mut self) -> Result<()> {
        if self.loaded.is_some() {
            return Ok(());
        }

        // Sequential mode defers loading to generate_sequential()
        if self.load_strategy == LoadStrategy::Sequential {
            return Ok(());
        }

        let is_schnell = self.detect_is_schnell();
        tracing::info!(model = %self.model_name, "loading FLUX model components...");

        let (t5_encoder_path, t5_tokenizer_path, clip_encoder_path, clip_tokenizer_path) =
            self.validate_paths()?;

        let cpu = Device::Cpu;
        let device = crate::device::create_device(&self.progress)?;
        let gpu_dtype = crate::engine::gpu_dtype(&device);

        tracing::info!("GPU device: {:?}, GPU dtype: {:?}", device, gpu_dtype);

        // --- Load FLUX transformer + VAE on GPU first (variable size) ---
        // This must happen before T5/CLIP so we can measure remaining VRAM.
        let is_quantized = self.detect_is_quantized();

        let flux_cfg = if is_schnell {
            flux::model::Config::schnell()
        } else {
            flux::model::Config::dev()
        };

        let xformer_label = if is_quantized {
            "Loading FLUX transformer (GPU, quantized)"
        } else {
            "Loading FLUX transformer (GPU, BF16)"
        };
        self.progress.stage_start(xformer_label);
        let xformer_stage = Instant::now();
        tracing::info!(
            path = %self.paths.transformer.display(),
            quantized = is_quantized,
            "loading FLUX transformer on GPU..."
        );

        let flux_model = if is_quantized {
            let vb =
                quantized_var_builder::VarBuilder::from_gguf(&self.paths.transformer, &device)?;
            FluxTransformer::Quantized(flux::quantized_model::Flux::new(&flux_cfg, vb)?)
        } else {
            let flux_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&self.paths.transformer),
                    gpu_dtype,
                    &device,
                )?
            };
            FluxTransformer::BF16(flux::model::Flux::new(&flux_cfg, flux_vb)?)
        };
        self.progress
            .stage_done(xformer_label, xformer_stage.elapsed());
        tracing::info!("FLUX transformer loaded on GPU");

        // Load VAE on GPU (small, ~300MB)
        self.progress.stage_start("Loading VAE (GPU)");
        let vae_stage = Instant::now();
        tracing::info!(path = %self.paths.vae.display(), "loading VAE on GPU...");
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&self.paths.vae),
                gpu_dtype,
                &device,
            )?
        };
        let vae_cfg = if is_schnell {
            flux::autoencoder::Config::schnell()
        } else {
            flux::autoencoder::Config::dev()
        };
        let vae = flux::autoencoder::AutoEncoder::new(&vae_cfg, vae_vb)?;
        self.progress
            .stage_done("Loading VAE (GPU)", vae_stage.elapsed());
        tracing::info!("VAE loaded on GPU");

        // --- Decide where to place T5 and CLIP based on remaining VRAM ---
        let free = free_vram_bytes().unwrap_or(0);
        if free > 0 {
            self.progress.info(&format!(
                "Free VRAM after transformer+VAE: {}",
                fmt_gb(free)
            ));
            tracing::info!(
                free_vram = free,
                "free VRAM after loading transformer + VAE"
            );
        }

        // --- T5 encoder: auto-select variant based on VRAM or explicit preference ---
        self.progress.stage_start("Selecting T5 encoder");
        let t5_resolve_start = Instant::now();
        let t5_preference = self.t5_variant.as_deref();
        let (resolved_t5_path, t5_on_gpu, t5_device_label) =
            self.resolve_t5_variant(t5_preference, &device, &cpu, free, &t5_encoder_path)?;
        self.progress
            .stage_done("Selecting T5 encoder", t5_resolve_start.elapsed());
        let t5_device = if t5_on_gpu { &device } else { &cpu };
        let t5_dtype = if t5_on_gpu { gpu_dtype } else { DType::F32 };

        // Load T5 encoder
        let t5_stage_label = format!("Loading T5 encoder ({t5_device_label})");
        self.progress.stage_start(&t5_stage_label);
        let t5_stage = Instant::now();
        tracing::info!(
            path = %resolved_t5_path.display(),
            device = %t5_device_label,
            "loading T5 encoder..."
        );
        let t5 = encoders::t5::T5Encoder::load(
            &resolved_t5_path,
            &t5_tokenizer_path,
            t5_device,
            t5_dtype,
        )?;
        self.progress
            .stage_done(&t5_stage_label, t5_stage.elapsed());
        tracing::info!(device = %t5_device_label, "T5 encoder loaded");

        // Re-check VRAM after T5 (it may have consumed GPU memory)
        let free_after_t5 = free_vram_bytes().unwrap_or(0);
        let clip_on_gpu = should_use_gpu(
            device.is_cuda(),
            device.is_metal(),
            free_after_t5,
            CLIP_VRAM_THRESHOLD,
        );
        let clip_device = if clip_on_gpu { &device } else { &cpu };
        let clip_dtype = if clip_on_gpu { gpu_dtype } else { DType::F32 };
        let clip_device_label = if clip_on_gpu { "GPU" } else { "CPU" };

        // Load CLIP encoder
        let clip_stage_label = format!("Loading CLIP encoder ({clip_device_label})");
        self.progress.stage_start(&clip_stage_label);
        let clip_stage = Instant::now();
        tracing::info!(
            path = %clip_encoder_path.display(),
            device = clip_device_label,
            "loading CLIP encoder..."
        );
        let clip = encoders::clip::ClipEncoder::load(
            &clip_encoder_path,
            &clip_tokenizer_path,
            clip_device,
            clip_dtype,
        )?;
        self.progress
            .stage_done(&clip_stage_label, clip_stage.elapsed());
        tracing::info!(device = clip_device_label, "CLIP encoder loaded");

        self.loaded = Some(LoadedFlux {
            flux_model,
            t5,
            clip,
            vae,
            device,
            dtype: gpu_dtype,
            is_schnell,
            is_quantized,
            t5_encoder_path: resolved_t5_path,
        });

        tracing::info!(model = %self.model_name, "all model components loaded successfully");
        Ok(())
    }

    /// Generate an image using sequential loading strategy.
    ///
    /// Loads components one at a time and drops them when done to minimize peak memory:
    /// 1. Load T5 → encode → drop T5
    /// 2. Load CLIP → encode → drop CLIP
    /// 3. Load transformer + VAE → denoise → drop transformer
    /// 4. VAE decode → drop VAE
    ///
    /// Peak memory: max(T5_size, transformer_size + VAE_size) instead of sum(all).
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let is_schnell = self.detect_is_schnell();
        let is_quantized = self.detect_is_quantized();

        let (t5_encoder_path, t5_tokenizer_path, clip_encoder_path, clip_tokenizer_path) =
            self.validate_paths()?;

        // Check memory budget
        if let Some(warning) = check_memory_budget(&self.paths, LoadStrategy::Sequential) {
            self.progress.info(&warning);
        }

        let device = crate::device::create_device(&self.progress)?;
        let gpu_dtype = crate::engine::gpu_dtype(&device);

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting sequential FLUX generation"
        );

        self.progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: T5 encoding ---
        let free = free_vram_bytes().unwrap_or(0);
        self.progress.stage_start("Selecting T5 encoder");
        let t5_resolve_start = Instant::now();
        let t5_preference = self.t5_variant.as_deref();
        let (resolved_t5_path, t5_on_gpu, t5_device_label) =
            self.resolve_t5_variant(t5_preference, &device, &Device::Cpu, free, &t5_encoder_path)?;
        self.progress
            .stage_done("Selecting T5 encoder", t5_resolve_start.elapsed());

        let t5_device = if t5_on_gpu { &device } else { &Device::Cpu };
        let t5_dtype = if t5_on_gpu { gpu_dtype } else { DType::F32 };

        // Pre-flight: check if T5 fits in free memory
        let t5_size = std::fs::metadata(&resolved_t5_path)
            .map(|m| m.len())
            .unwrap_or(0);
        preflight_memory_check("T5 encoder", t5_size)?;
        if let Some(status) = memory_status_string() {
            self.progress.info(&status);
        }

        let t5_stage_label = format!("Loading T5 encoder ({t5_device_label})");
        self.progress.stage_start(&t5_stage_label);
        let t5_stage = Instant::now();
        let mut t5 = encoders::t5::T5Encoder::load(
            &resolved_t5_path,
            &t5_tokenizer_path,
            t5_device,
            t5_dtype,
        )?;
        self.progress
            .stage_done(&t5_stage_label, t5_stage.elapsed());

        self.progress.stage_start("Encoding prompt (T5)");
        let encode_t5 = Instant::now();
        let t5_emb = t5.encode(&req.prompt, &device, gpu_dtype)?;
        self.progress
            .stage_done("Encoding prompt (T5)", encode_t5.elapsed());

        // Drop T5 to free memory before loading CLIP
        drop(t5);
        self.progress.info("Freed T5 encoder");
        tracing::info!("T5 encoder dropped (sequential mode)");

        // --- Phase 2: CLIP encoding ---
        let free_for_clip = free_vram_bytes().unwrap_or(0);
        let clip_on_gpu = should_use_gpu(
            device.is_cuda(),
            device.is_metal(),
            free_for_clip,
            CLIP_VRAM_THRESHOLD,
        );
        let clip_device = if clip_on_gpu { &device } else { &Device::Cpu };
        let clip_dtype = if clip_on_gpu { gpu_dtype } else { DType::F32 };
        let clip_device_label = if clip_on_gpu { "GPU" } else { "CPU" };

        let clip_stage_label = format!("Loading CLIP encoder ({clip_device_label})");
        self.progress.stage_start(&clip_stage_label);
        let clip_stage = Instant::now();
        let clip = encoders::clip::ClipEncoder::load(
            &clip_encoder_path,
            &clip_tokenizer_path,
            clip_device,
            clip_dtype,
        )?;
        self.progress
            .stage_done(&clip_stage_label, clip_stage.elapsed());

        self.progress.stage_start("Encoding prompt (CLIP)");
        let encode_clip = Instant::now();
        let clip_emb = {
            let mut clip = clip;
            clip.encode(&req.prompt, &device, gpu_dtype)?
        };
        self.progress
            .stage_done("Encoding prompt (CLIP)", encode_clip.elapsed());

        // CLIP is dropped here (goes out of scope)
        self.progress.info("Freed CLIP encoder");
        tracing::info!("CLIP encoder dropped (sequential mode)");

        // --- Phase 3: Load transformer + VAE, denoise ---
        let xformer_size = std::fs::metadata(&self.paths.transformer)
            .map(|m| m.len())
            .unwrap_or(0);
        let vae_file_size = std::fs::metadata(&self.paths.vae)
            .map(|m| m.len())
            .unwrap_or(0);
        preflight_memory_check("FLUX transformer + VAE", xformer_size + vae_file_size)?;
        if let Some(status) = memory_status_string() {
            self.progress.info(&status);
        }

        let flux_cfg = if is_schnell {
            flux::model::Config::schnell()
        } else {
            flux::model::Config::dev()
        };

        let xformer_label = if is_quantized {
            "Loading FLUX transformer (GPU, quantized)"
        } else {
            "Loading FLUX transformer (GPU, BF16)"
        };
        self.progress.stage_start(xformer_label);
        let xformer_stage = Instant::now();

        let flux_model = if is_quantized {
            let vb =
                quantized_var_builder::VarBuilder::from_gguf(&self.paths.transformer, &device)?;
            FluxTransformer::Quantized(flux::quantized_model::Flux::new(&flux_cfg, vb)?)
        } else {
            let flux_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&self.paths.transformer),
                    gpu_dtype,
                    &device,
                )?
            };
            FluxTransformer::BF16(flux::model::Flux::new(&flux_cfg, flux_vb)?)
        };
        self.progress
            .stage_done(xformer_label, xformer_stage.elapsed());

        // Load VAE (small, keep it loaded through denoise + decode)
        self.progress.stage_start("Loading VAE (GPU)");
        let vae_stage = Instant::now();
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&self.paths.vae),
                gpu_dtype,
                &device,
            )?
        };
        let vae_cfg = if is_schnell {
            flux::autoencoder::Config::schnell()
        } else {
            flux::autoencoder::Config::dev()
        };
        let vae = flux::autoencoder::AutoEncoder::new(&vae_cfg, vae_vb)?;
        self.progress
            .stage_done("Loading VAE (GPU)", vae_stage.elapsed());

        // Generate noise and build state
        let noise_dtype = if is_quantized { DType::F32 } else { gpu_dtype };
        let latent_h = height / 16 * 2;
        let latent_w = width / 16 * 2;
        let img =
            crate::engine::seeded_randn(seed, &[1, 16, latent_h, latent_w], &device, noise_dtype)?;

        let (t5_emb_state, clip_emb_state, img_state) = if is_quantized {
            (
                t5_emb.to_dtype(DType::F32)?,
                clip_emb.to_dtype(DType::F32)?,
                img.to_dtype(DType::F32)?,
            )
        } else {
            (t5_emb, clip_emb, img)
        };

        let state = flux::sampling::State::new(&t5_emb_state, &clip_emb_state, &img_state)?;

        let timesteps = if is_schnell {
            flux::sampling::get_schedule(req.steps as usize, None)
        } else {
            flux::sampling::get_schedule(req.steps as usize, Some((state.img.dim(1)?, 0.5, 1.15)))
        };

        let denoise_label = format!("Denoising ({} steps)", timesteps.len());
        self.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        let img = flux_model.denoise(
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            req.guidance,
            &self.progress,
        )?;

        let img = flux::sampling::unpack(&img, height, width)?;
        self.progress
            .stage_done(&denoise_label, denoise_start.elapsed());

        // Drop transformer + state to free memory for VAE decode
        drop(flux_model);
        self.progress.info("Freed FLUX transformer");
        drop(state);
        drop(t5_emb_state);
        drop(clip_emb_state);
        drop(img_state);
        tracing::info!("Transformer dropped (sequential mode), decoding VAE...");

        // --- Phase 4: VAE decode ---
        self.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();
        let img = vae.decode(&img.to_dtype(gpu_dtype)?)?;

        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?;

        self.progress
            .stage_done("VAE decode", vae_decode_start.elapsed());
        // VAE dropped here

        let image_bytes = encode_image(&img, req.output_format, req.width, req.height)?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "sequential generation complete");

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

impl InferenceEngine for FluxEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        // Sequential mode: load-use-drop each component
        if self.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components
        // Borrow progress reporter separately from loaded state.
        let progress = &self.progress;

        // Grab path references before borrowing loaded mutably
        let t5_encoder_path = self
            .loaded
            .as_ref()
            .map(|l| l.t5_encoder_path.clone())
            .or_else(|| self.paths.t5_encoder.clone())
            .ok_or_else(|| anyhow::anyhow!("T5 encoder path required for FLUX models"))?;
        let clip_encoder_path = self
            .paths
            .clip_encoder
            .clone()
            .ok_or_else(|| anyhow::anyhow!("CLIP encoder path required for FLUX models"))?;

        let loaded = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed,
            width,
            height,
            steps = req.steps,
            "starting generation"
        );

        // If T5/CLIP were dropped after a previous generation (GPU offload), reload them.
        if loaded.t5.model.is_none() {
            progress.stage_start("Reloading T5 encoder (GPU)");
            let reload_start = Instant::now();
            loaded.t5.reload(&t5_encoder_path, loaded.dtype)?;
            progress.stage_done("Reloading T5 encoder (GPU)", reload_start.elapsed());
        }
        if loaded.clip.model.is_none() {
            progress.stage_start("Reloading CLIP encoder (GPU)");
            let reload_start = Instant::now();
            loaded.clip.reload(&clip_encoder_path, loaded.dtype)?;
            progress.stage_done("Reloading CLIP encoder (GPU)", reload_start.elapsed());
        }

        // 1. Encode prompt with T5 (may be on GPU or CPU depending on VRAM)
        progress.stage_start("Encoding prompt (T5)");
        let encode_t5 = Instant::now();
        let t5_emb = loaded
            .t5
            .encode(&req.prompt, &loaded.device, loaded.dtype)?;
        progress.stage_done("Encoding prompt (T5)", encode_t5.elapsed());
        tracing::info!("T5 encoding complete");

        // 2. Encode prompt with CLIP (may be on GPU or CPU depending on VRAM)
        progress.stage_start("Encoding prompt (CLIP)");
        let encode_clip = Instant::now();
        let clip_emb = loaded
            .clip
            .encode(&req.prompt, &loaded.device, loaded.dtype)?;
        progress.stage_done("Encoding prompt (CLIP)", encode_clip.elapsed());
        tracing::info!("CLIP encoding complete");

        // Drop T5/CLIP from GPU to free VRAM for the denoising pass.
        // Their weights are only needed for prompt encoding (already done above).
        // On CPU this is a no-op (CPU RAM is plentiful); they'll be reloaded next call.
        if loaded.t5.on_gpu {
            loaded.t5.drop_weights();
            tracing::info!("T5 encoder dropped from GPU to free VRAM for denoising");
        }
        if loaded.clip.on_gpu {
            loaded.clip.drop_weights();
            tracing::info!("CLIP encoder dropped from GPU to free VRAM for denoising");
        }

        // 3. Generate initial noise (F32 for quantized, gpu_dtype for BF16)
        let noise_dtype = if loaded.is_quantized {
            DType::F32
        } else {
            loaded.dtype
        };
        let latent_h = height / 16 * 2;
        let latent_w = width / 16 * 2;
        let img = crate::engine::seeded_randn(
            seed,
            &[1, 16, latent_h, latent_w],
            &loaded.device,
            noise_dtype,
        )?;

        // For quantized model, state tensors must be F32
        let (t5_emb_state, clip_emb_state, img_state) = if loaded.is_quantized {
            (
                t5_emb.to_dtype(DType::F32)?,
                clip_emb.to_dtype(DType::F32)?,
                img.to_dtype(DType::F32)?,
            )
        } else {
            (t5_emb, clip_emb, img)
        };

        // 4. Build sampling state
        let state = flux::sampling::State::new(&t5_emb_state, &clip_emb_state, &img_state)?;

        // 5. Get timestep schedule
        let timesteps = if loaded.is_schnell {
            flux::sampling::get_schedule(req.steps as usize, None)
        } else {
            flux::sampling::get_schedule(req.steps as usize, Some((state.img.dim(1)?, 0.5, 1.15)))
        };

        let denoise_label = format!("Denoising ({} steps)", timesteps.len());
        progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();
        tracing::info!(
            steps = timesteps.len(),
            quantized = loaded.is_quantized,
            "running denoising loop..."
        );

        // 6. Denoise — guidance from request (0.0 for schnell, 3.5+ for dev/finetuned)
        let img = loaded.flux_model.denoise(
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            req.guidance,
            progress,
        )?;

        // 7. Unpack latent to spatial
        let img = flux::sampling::unpack(&img, height, width)?;
        progress.stage_done(&denoise_label, denoise_start.elapsed());
        tracing::info!("denoising complete, decoding VAE...");

        // Free denoising intermediates (state, embeddings) before VAE decode.
        // These GPU tensors can be several GB and the VAE needs that VRAM.
        drop(state);
        drop(t5_emb_state);
        drop(clip_emb_state);
        drop(img_state);

        // 8. Decode with VAE — cast to VAE dtype (BF16) in case quantized model produced F32
        progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();
        let img = loaded.vae.decode(&img.to_dtype(loaded.dtype)?)?;

        // 9. Convert to u8 image: clamp to [-1, 1], map to [0, 255]
        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?; // remove batch dim: [3, H, W]

        progress.stage_done("VAE decode", vae_decode_start.elapsed());
        tracing::info!("VAE decode complete, encoding output image...");

        // 10. Convert candle tensor to image bytes
        let image_bytes = encode_image(&img, req.output_format, req.width, req.height)?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "generation complete");

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
        FluxEngine::load(self)
    }

    fn unload(&mut self) {
        self.loaded = None;
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }
}
