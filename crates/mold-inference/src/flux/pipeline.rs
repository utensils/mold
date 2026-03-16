use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::flux;
use candle_transformers::quantized_var_builder;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::time::Instant;

use crate::device::{
    fmt_gb, free_vram_bytes, should_use_gpu, t5_vram_threshold, CLIP_VRAM_THRESHOLD,
    T5_VRAM_THRESHOLD,
};
use crate::encoders;
use crate::engine::{rand_seed, InferenceEngine};
use crate::image::encode_image;
use crate::progress::{ProgressCallback, ProgressEvent};

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
    /// Optional progress callback for UI reporting.
    on_progress: Option<ProgressCallback>,
    /// T5 variant preference: None/"auto" = auto-select, "fp16" = force FP16, "q8"/"q5"/etc = specific quantized.
    t5_variant: Option<String>,
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
    ) -> Self {
        Self {
            loaded: None,
            model_name,
            paths,
            is_schnell_override,
            on_progress: None,
            t5_variant,
        }
    }

    /// Set a progress callback for receiving loading/inference status updates.
    pub fn set_on_progress<F: Fn(ProgressEvent) + Send + Sync + 'static>(&mut self, callback: F) {
        self.on_progress = Some(Box::new(callback));
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
                self.info(&format!(
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
                self.info(&format!("Using FP16 T5 on {} (explicit)", label));
                Ok((default_t5_path.to_path_buf(), on_gpu, label.to_string()))
            }

            // Auto mode (default): try FP16 on GPU, then quantized on GPU, then FP16 on CPU
            _ => {
                // Can FP16 T5 fit on GPU?
                if should_use_gpu(is_cuda, is_metal, free_vram, T5_VRAM_THRESHOLD) {
                    if is_metal {
                        self.info("Loading FP16 T5 on GPU (unified memory)");
                    } else {
                        self.info(&format!(
                            "Loading FP16 T5 on GPU ({} free > {} threshold)",
                            fmt_gb(free_vram),
                            fmt_gb(T5_VRAM_THRESHOLD),
                        ));
                    }
                    return Ok((default_t5_path.to_path_buf(), true, "GPU".to_string()));
                }

                // FP16 won't fit on GPU — try quantized variants (largest first)
                // (Only relevant for CUDA with discrete VRAM; Metal unified memory
                //  always passes the threshold check above.)
                if is_cuda {
                    for variant in known_t5_variants() {
                        let threshold = t5_vram_threshold(variant.size_bytes);
                        if free_vram > threshold {
                            // Check cache first, download if needed
                            let path = match cached_file_path(variant.hf_repo, variant.hf_filename)
                            {
                                Some(p) => p,
                                None => {
                                    self.info(&format!(
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
                                    download_single_file_sync(variant.hf_repo, variant.hf_filename)
                                        .map_err(|e| {
                                            anyhow::anyhow!(
                                                "failed to download T5 {}: {e}",
                                                variant.tag
                                            )
                                        })?
                                }
                            };
                            self.info(&format!(
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

                // No quantized variant fits on GPU either — fall back to FP16 on CPU
                if is_cuda {
                    self.info(&format!(
                        "Loading FP16 T5 on CPU ({} free, no variant fits on GPU)",
                        fmt_gb(free_vram),
                    ));
                } else {
                    self.info("No GPU detected, loading T5 on CPU");
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

        if let Some(path) = cached_file_path(variant.hf_repo, variant.hf_filename) {
            return Ok(path);
        }
        self.info(&format!(
            "Downloading T5 {} ({})...",
            variant.tag,
            fmt_gb(variant.size_bytes),
        ));
        download_single_file_sync(variant.hf_repo, variant.hf_filename)
            .map_err(|e| anyhow::anyhow!("failed to download T5 {}: {e}", variant.tag))
    }

    /// Load all model components into GPU memory.
    pub fn load(&mut self) -> Result<()> {
        if self.loaded.is_some() {
            return Ok(());
        }

        // Detect model family: explicit override → name → transformer filename
        let is_schnell = self.is_schnell_override.unwrap_or_else(|| {
            self.model_name.contains("schnell")
                || self
                    .paths
                    .transformer
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.contains("schnell"))
                    .unwrap_or(false)
        });
        tracing::info!(model = %self.model_name, "loading FLUX model components...");

        let cpu = Device::Cpu;
        let device = if candle_core::utils::cuda_is_available() {
            self.info("CUDA detected, using GPU");
            tracing::info!("CUDA detected, using GPU");
            Device::new_cuda(0)?
        } else if candle_core::utils::metal_is_available() {
            self.info("Metal detected, using GPU");
            tracing::info!("Metal detected, using MPS");
            Device::new_metal(0)?
        } else {
            self.info("No GPU detected, using CPU");
            tracing::warn!("No GPU detected, falling back to CPU");
            Device::Cpu
        };
        let gpu_dtype = if device.is_cuda() || device.is_metal() {
            DType::BF16
        } else {
            DType::F32
        };

        tracing::info!("GPU device: {:?}, GPU dtype: {:?}", device, gpu_dtype);

        // Validate T5 paths are present (required for FLUX)
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

        // Validate all paths exist before attempting unsafe mmap
        for (label, path) in [
            ("transformer", &self.paths.transformer),
            ("vae", &self.paths.vae),
            ("t5_encoder", &t5_encoder_path),
            ("clip_encoder", &self.paths.clip_encoder),
            ("t5_tokenizer", &t5_tokenizer_path),
            ("clip_tokenizer", &self.paths.clip_tokenizer),
        ] {
            if !path.exists() {
                bail!("{label} file not found: {}", path.display());
            }
        }

        // --- Load FLUX transformer + VAE on GPU first (variable size) ---
        // This must happen before T5/CLIP so we can measure remaining VRAM.

        // Load FLUX transformer on GPU — GGUF quantized or BF16 safetensors
        let is_quantized = self
            .paths
            .transformer
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);

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
        self.stage_start(xformer_label);
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
        self.stage_done(xformer_label, xformer_stage.elapsed());
        tracing::info!("FLUX transformer loaded on GPU");

        // Load VAE on GPU (small, ~300MB)
        self.stage_start("Loading VAE (GPU)");
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
        self.stage_done("Loading VAE (GPU)", vae_stage.elapsed());
        tracing::info!("VAE loaded on GPU");

        // --- Decide where to place T5 and CLIP based on remaining VRAM ---
        let free = free_vram_bytes().unwrap_or(0);
        if free > 0 {
            self.info(&format!(
                "Free VRAM after transformer+VAE: {}",
                fmt_gb(free)
            ));
            tracing::info!(
                free_vram = free,
                "free VRAM after loading transformer + VAE"
            );
        }

        // --- T5 encoder: auto-select variant based on VRAM or explicit preference ---
        // Emit a stage so the spinner shows something relevant during variant resolution
        // (which may involve downloading a quantized T5 GGUF).
        self.stage_start("Selecting T5 encoder");
        let t5_resolve_start = Instant::now();
        let t5_preference = self.t5_variant.as_deref();
        let (resolved_t5_path, t5_on_gpu, t5_device_label) =
            self.resolve_t5_variant(t5_preference, &device, &cpu, free, &t5_encoder_path)?;
        self.stage_done("Selecting T5 encoder", t5_resolve_start.elapsed());
        let t5_device = if t5_on_gpu { &device } else { &cpu };
        let t5_dtype = if t5_on_gpu { gpu_dtype } else { DType::F32 };

        // Load T5 encoder
        let t5_stage_label = format!("Loading T5 encoder ({t5_device_label})");
        self.stage_start(&t5_stage_label);
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
        self.stage_done(&t5_stage_label, t5_stage.elapsed());
        tracing::info!(device = %t5_device_label, "T5 encoder loaded");

        // Re-check VRAM after T5 (it may have consumed GPU memory)
        let free_after_t5 = free_vram_bytes().unwrap_or(0);
        let clip_on_gpu = should_use_gpu(device.is_cuda(), device.is_metal(), free_after_t5, CLIP_VRAM_THRESHOLD);
        let clip_device = if clip_on_gpu { &device } else { &cpu };
        let clip_dtype = if clip_on_gpu { gpu_dtype } else { DType::F32 };
        let clip_device_label = if clip_on_gpu { "GPU" } else { "CPU" };

        // Load CLIP encoder
        let clip_stage_label = format!("Loading CLIP encoder ({clip_device_label})");
        self.stage_start(&clip_stage_label);
        let clip_stage = Instant::now();
        tracing::info!(
            path = %self.paths.clip_encoder.display(),
            device = clip_device_label,
            "loading CLIP encoder..."
        );
        let clip = encoders::clip::ClipEncoder::load(
            &self.paths.clip_encoder,
            &self.paths.clip_tokenizer,
            clip_device,
            clip_dtype,
        )?;
        self.stage_done(&clip_stage_label, clip_stage.elapsed());
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
}

impl InferenceEngine for FluxEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        // Extract progress callback ref so we can use it while loaded is mutably borrowed.
        let on_progress = &self.on_progress;
        let emit = |event: ProgressEvent| {
            if let Some(cb) = on_progress {
                cb(event);
            }
        };
        let stage_start = |name: &str| {
            emit(ProgressEvent::StageStart {
                name: name.to_string(),
            });
        };
        let stage_done = |name: &str, elapsed: std::time::Duration| {
            emit(ProgressEvent::StageDone {
                name: name.to_string(),
                elapsed,
            });
        };

        // Grab path references before borrowing loaded mutably
        let t5_encoder_path = self
            .loaded
            .as_ref()
            .map(|l| l.t5_encoder_path.clone())
            .or_else(|| self.paths.t5_encoder.clone())
            .ok_or_else(|| anyhow::anyhow!("T5 encoder path required for FLUX models"))?;
        let clip_encoder_path = self.paths.clip_encoder.clone();

        let loaded = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);
        loaded.device.set_seed(seed)?;

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
            stage_start("Reloading T5 encoder (GPU)");
            let reload_start = Instant::now();
            loaded.t5.reload(&t5_encoder_path, loaded.dtype)?;
            stage_done("Reloading T5 encoder (GPU)", reload_start.elapsed());
        }
        if loaded.clip.model.is_none() {
            stage_start("Reloading CLIP encoder (GPU)");
            let reload_start = Instant::now();
            loaded.clip.reload(&clip_encoder_path, loaded.dtype)?;
            stage_done("Reloading CLIP encoder (GPU)", reload_start.elapsed());
        }

        // 1. Encode prompt with T5 (may be on GPU or CPU depending on VRAM)
        stage_start("Encoding prompt (T5)");
        let encode_t5 = Instant::now();
        let t5_emb = loaded
            .t5
            .encode(&req.prompt, &loaded.device, loaded.dtype)?;
        stage_done("Encoding prompt (T5)", encode_t5.elapsed());
        tracing::info!("T5 encoding complete");

        // 2. Encode prompt with CLIP (may be on GPU or CPU depending on VRAM)
        stage_start("Encoding prompt (CLIP)");
        let encode_clip = Instant::now();
        let clip_emb = loaded
            .clip
            .encode(&req.prompt, &loaded.device, loaded.dtype)?;
        stage_done("Encoding prompt (CLIP)", encode_clip.elapsed());
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
        let img =
            flux::sampling::get_noise(1, height, width, &loaded.device)?.to_dtype(noise_dtype)?;

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
        stage_start(&denoise_label);
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
        )?;

        // 7. Unpack latent to spatial
        let img = flux::sampling::unpack(&img, height, width)?;
        stage_done(&denoise_label, denoise_start.elapsed());
        tracing::info!("denoising complete, decoding VAE...");

        // Free denoising intermediates (state, embeddings) before VAE decode.
        // These GPU tensors can be several GB and the VAE needs that VRAM.
        drop(state);
        drop(t5_emb_state);
        drop(clip_emb_state);
        drop(img_state);

        // 8. Decode with VAE — cast to VAE dtype (BF16) in case quantized model produced F32
        stage_start("VAE decode");
        let vae_decode_start = Instant::now();
        let img = loaded.vae.decode(&img.to_dtype(loaded.dtype)?)?;

        // 9. Convert to u8 image: clamp to [-1, 1], map to [0, 255]
        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?; // remove batch dim: [3, H, W]

        stage_done("VAE decode", vae_decode_start.elapsed());
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
        self.loaded.is_some()
    }

    fn load(&mut self) -> Result<()> {
        FluxEngine::load(self)
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.on_progress = Some(callback);
    }
}
