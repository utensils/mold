//! Flux.2 Klein-4B inference engine.
//!
//! Follows the same Eager + Sequential loading pattern as FluxEngine and ZImageEngine.
//!
//! Key differences from FLUX.1:
//! - Uses Qwen3 text encoder (not T5 + CLIP)
//! - Qwen3 hidden states from layers 9, 18, 27 are stacked to produce joint_attention_dim=7680
//! - VAE has latent_channels=32 (not 16)
//! - Transformer has 128 input channels (not 64)
//! - 4D RoPE (not 3D)
//! - Klein is distilled (no guidance embedding)
//! - No pooled text vector input
//! - Linear timestep schedule (distilled, no time-shifting)

use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::sync::Mutex;
use std::time::Instant;

use super::sampling::{self, Flux2State};
use super::transformer::{Flux2Config, Flux2TransformerWrapper};
use super::vae::{Flux2AutoEncoder, Flux2VaeConfig};
use crate::cache::{
    clear_cache, get_or_insert_cached_tensor, prompt_text_key, restore_cached_tensor, CachedTensor,
    LruCache, DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{
    check_memory_budget, fmt_gb, free_vram_bytes, memory_status_string, preflight_memory_check,
};
use crate::encoders;
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image};
use crate::progress::{ProgressCallback, ProgressReporter};

// ---------------------------------------------------------------------------
// Loaded state
// ---------------------------------------------------------------------------

/// Loaded Flux.2 model components, ready for inference.
struct LoadedFlux2 {
    /// None after being dropped for VAE decode VRAM; reloaded on next generate.
    transformer: Option<Flux2TransformerWrapper>,
    text_encoder: encoders::qwen3::Qwen3Encoder,
    vae: Flux2AutoEncoder,
    /// GPU device for transformer + VAE
    device: Device,
    dtype: DType,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Flux.2 Klein-4B inference engine backed by candle.
pub struct Flux2Engine {
    base: EngineBase<LoadedFlux2>,
    /// Qwen3 variant preference: None/"auto" = VRAM-based, "bf16" = force BF16, "q8"/etc = specific.
    qwen3_variant: Option<String>,
    prompt_cache: Mutex<LruCache<String, CachedTensor>>,
}

impl Flux2Engine {
    /// Create a new Flux2Engine. Does not load models until `load()` is called.
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        qwen3_variant: Option<String>,
        load_strategy: LoadStrategy,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy),
            qwen3_variant,
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
        }
    }

    /// Select the appropriate transformer config based on the model name.
    /// Klein-9B uses a larger architecture than Klein-4B.
    fn resolve_config(&self) -> Flux2Config {
        let name = self.base.model_name.to_lowercase();
        if name.contains("9b") {
            Flux2Config::klein_9b()
        } else {
            Flux2Config::klein()
        }
    }

    /// Validate that all required paths exist.
    fn validate_paths(&self) -> Result<std::path::PathBuf> {
        let text_tokenizer_path = self
            .base
            .paths
            .text_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("text tokenizer path required for Flux.2 models"))?;
        if !text_tokenizer_path.exists() {
            bail!(
                "text tokenizer file not found: {}",
                text_tokenizer_path.display()
            );
        }

        let encoder_paths = self.text_encoder_paths();
        if encoder_paths.is_empty() {
            bail!("text encoder paths required for Flux.2 models");
        }
        for path in &encoder_paths {
            if !path.exists() {
                bail!("text encoder file not found: {}", path.display());
            }
        }

        if !self.base.paths.transformer.exists() {
            bail!(
                "transformer file not found: {}",
                self.base.paths.transformer.display()
            );
        }
        if !self.base.paths.vae.exists() {
            bail!("VAE file not found: {}", self.base.paths.vae.display());
        }

        Ok(text_tokenizer_path.clone())
    }

    /// Check if the transformer file is a GGUF (quantized) file.
    fn is_gguf_transformer(&self) -> bool {
        self.base
            .paths
            .transformer
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    }

    /// Load the transformer from either GGUF or BF16 safetensors.
    fn load_transformer(
        &self,
        cfg: &Flux2Config,
        gpu_dtype: DType,
        device: &Device,
    ) -> Result<(Flux2TransformerWrapper, &'static str)> {
        if self.is_gguf_transformer() {
            let gguf_vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &self.base.paths.transformer,
                device,
            )?;
            Ok((
                Flux2TransformerWrapper::Quantized(
                    super::quantized_transformer::QuantizedFlux2Transformer::new(
                        cfg, gguf_vb, gpu_dtype, device,
                    )?,
                ),
                "Loading Flux.2 transformer (GPU, GGUF)",
            ))
        } else {
            let xformer_paths = if !self.base.paths.transformer_shards.is_empty() {
                self.base.paths.transformer_shards.clone()
            } else {
                vec![self.base.paths.transformer.clone()]
            };
            let flux_vb = crate::weight_loader::load_safetensors_with_progress(
                &xformer_paths,
                gpu_dtype,
                device,
                "Flux.2 transformer",
                &self.base.progress,
            )?;
            Ok((
                Flux2TransformerWrapper::BF16(super::transformer::Flux2Transformer::new(
                    cfg, flux_vb,
                )?),
                "Loading Flux.2 transformer (GPU, BF16)",
            ))
        }
    }

    /// Reload transformer using `&mut self` — called before the main `loaded` borrow
    /// to avoid borrow conflicts.
    fn reload_transformer_if_needed(&mut self) -> Result<()> {
        let needs_reload = self
            .base
            .loaded
            .as_ref()
            .map(|l| l.transformer.is_none())
            .unwrap_or(false);

        if needs_reload {
            let cfg = self.resolve_config();
            self.base
                .progress
                .stage_start("Reloading Flux.2 transformer");
            let reload_start = Instant::now();
            let (transformer, _label) = self.load_transformer(
                &cfg,
                self.base.loaded.as_ref().unwrap().dtype,
                &self.base.loaded.as_ref().unwrap().device.clone(),
            )?;
            self.base.loaded.as_mut().unwrap().transformer = Some(transformer);
            self.base
                .progress
                .stage_done("Reloading Flux.2 transformer", reload_start.elapsed());
        }
        Ok(())
    }

    /// Get text encoder file paths (shards or single file).
    fn text_encoder_paths(&self) -> Vec<std::path::PathBuf> {
        if !self.base.paths.text_encoder_files.is_empty() {
            self.base.paths.text_encoder_files.clone()
        } else {
            // Fallback: t5_encoder field is reused as the generic text encoder path
            self.base
                .paths
                .t5_encoder
                .as_ref()
                .map(|p| vec![p.clone()])
                .unwrap_or_default()
        }
    }

    /// Encode a prompt with the Qwen3 text encoder, extracting hidden states from
    /// layers 9, 18, 27 and stacking them to produce joint_attention_dim=7680.
    ///
    /// Klein's Qwen3 has hidden_size=2560. The transformer expects context_in_dim=7680,
    /// which is 2560 * 3. The HuggingFace pipeline stacks outputs from intermediate
    /// layers [9, 18, 27] to form the text conditioning.
    ///
    /// Layers 9, 18, 27 correspond to roughly 1/4, 1/2, 3/4 depth of the 36-layer Qwen3.
    const QWEN3_HIDDEN_LAYERS: [usize; 3] = [9, 18, 27];

    fn encode_and_stack(
        encoder: &mut encoders::qwen3::Qwen3Encoder,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<Tensor> {
        // Extract hidden states from layers 9, 18, 27 and stack to (B, seq, 7680)
        let (stacked, _token_count) = encoder.encode_with_layers(
            prompt,
            target_device,
            target_dtype,
            &Self::QWEN3_HIDDEN_LAYERS,
        )?;
        Ok(stacked)
    }

    fn encode_prompt_cached(
        progress: &ProgressReporter,
        prompt_cache: &Mutex<LruCache<String, CachedTensor>>,
        encoder: &mut encoders::qwen3::Qwen3Encoder,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<Tensor> {
        let cache_key = prompt_text_key(prompt);
        let (txt_emb, cache_hit) = get_or_insert_cached_tensor(
            prompt_cache,
            cache_key,
            target_device,
            target_dtype,
            || {
                progress.stage_start("Encoding prompt (Qwen3)");
                let encode_start = Instant::now();
                let txt_emb = Self::encode_and_stack(encoder, prompt, target_device, target_dtype)?;
                progress.stage_done("Encoding prompt (Qwen3)", encode_start.elapsed());
                Ok(txt_emb)
            },
        )?;
        if cache_hit {
            progress.cache_hit("prompt conditioning");
        }
        Ok(txt_emb)
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

        tracing::info!(model = %self.base.model_name, "loading Flux.2 Klein model components...");

        let text_tokenizer_path = self.validate_paths()?;

        let cpu = Device::Cpu;
        let device = crate::device::create_device(&self.base.progress)?;
        let gpu_dtype = crate::engine::gpu_dtype(&device);

        tracing::info!("GPU device: {:?}, GPU dtype: {:?}", device, gpu_dtype);

        // --- Load transformer on GPU first ---
        let flux2_cfg = self.resolve_config();
        let xformer_stage = Instant::now();
        let (transformer, xformer_label) = self.load_transformer(&flux2_cfg, gpu_dtype, &device)?;
        self.base
            .progress
            .stage_done(xformer_label, xformer_stage.elapsed());

        // --- Load VAE on GPU ---
        self.base.progress.stage_start("Loading VAE (GPU)");
        let vae_stage = Instant::now();
        tracing::info!(path = %self.base.paths.vae.display(), "loading VAE on GPU...");
        let vae_cfg = Flux2VaeConfig::klein();
        let vae_vb = crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(&self.base.paths.vae),
            gpu_dtype,
            &device,
            "VAE",
            &self.base.progress,
        )?;
        let vae = Flux2AutoEncoder::new(&vae_cfg, vae_vb)?;
        self.base
            .progress
            .stage_done("Loading VAE (GPU)", vae_stage.elapsed());
        tracing::info!("VAE loaded on GPU");

        // --- Resolve and load Qwen3 text encoder ---
        let free = free_vram_bytes().unwrap_or(0);
        if free > 0 {
            self.base.progress.info(&format!(
                "Free VRAM after transformer+VAE: {}",
                fmt_gb(free)
            ));
        }

        self.base.progress.stage_start("Selecting Qwen3 encoder");
        let resolve_start = Instant::now();
        let (encoder_paths, is_gguf, on_gpu, device_label) = {
            let bf16_paths = self.text_encoder_paths();
            let have_bf16 = !bf16_paths.is_empty() && bf16_paths.iter().all(|p| p.exists());
            crate::encoders::variant_resolution::resolve_qwen3_variant(
                &self.base.progress,
                self.qwen3_variant.as_deref(),
                &device,
                free,
                &bf16_paths,
                have_bf16,
                true,
            )?
        };
        self.base
            .progress
            .stage_done("Selecting Qwen3 encoder", resolve_start.elapsed());

        let enc_device = if on_gpu { &device } else { &cpu };
        let enc_dtype = if on_gpu { gpu_dtype } else { DType::F32 };

        let enc_stage_label = format!("Loading Qwen3 encoder ({device_label})");
        self.base.progress.stage_start(&enc_stage_label);
        let enc_stage = Instant::now();

        let text_encoder = if is_gguf {
            encoders::qwen3::Qwen3Encoder::load_gguf(
                &encoder_paths[0],
                &text_tokenizer_path,
                enc_device,
            )?
        } else {
            encoders::qwen3::Qwen3Encoder::load_bf16(
                &encoder_paths,
                &text_tokenizer_path,
                enc_device,
                enc_dtype,
                &self.base.progress,
            )?
        };
        self.base
            .progress
            .stage_done(&enc_stage_label, enc_stage.elapsed());
        tracing::info!(device = %device_label, "Qwen3 encoder loaded");

        self.base.loaded = Some(LoadedFlux2 {
            transformer: Some(transformer),
            text_encoder,
            vae,
            device,
            dtype: gpu_dtype,
        });

        tracing::info!(model = %self.base.model_name, "all Flux.2 model components loaded successfully");
        Ok(())
    }

    /// Generate an image using sequential loading strategy (load-use-drop).
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let text_tokenizer_path = self.validate_paths()?;

        if let Some(warning) = check_memory_budget(&self.base.paths, LoadStrategy::Sequential) {
            self.base.progress.info(&warning);
        }

        let device = crate::device::create_device(&self.base.progress)?;
        let gpu_dtype = crate::engine::gpu_dtype(&device);

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting sequential Flux.2 generation"
        );

        self.base
            .progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: Qwen3 text encoding ---
        let free = free_vram_bytes().unwrap_or(0);
        self.base.progress.stage_start("Selecting Qwen3 encoder");
        let resolve_start = Instant::now();
        let (encoder_paths, is_gguf, on_gpu, device_label) = {
            let bf16_paths = self.text_encoder_paths();
            let have_bf16 = !bf16_paths.is_empty() && bf16_paths.iter().all(|p| p.exists());
            crate::encoders::variant_resolution::resolve_qwen3_variant(
                &self.base.progress,
                self.qwen3_variant.as_deref(),
                &device,
                free,
                &bf16_paths,
                have_bf16,
                true,
            )?
        };
        self.base
            .progress
            .stage_done("Selecting Qwen3 encoder", resolve_start.elapsed());

        let enc_device = if on_gpu { &device } else { &Device::Cpu };
        let enc_dtype = if on_gpu { gpu_dtype } else { DType::F32 };

        // Pre-flight memory check
        let enc_size: u64 = encoder_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
            .sum();
        preflight_memory_check("Qwen3 encoder", enc_size)?;
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        let enc_stage_label = format!("Loading Qwen3 encoder ({device_label})");
        self.base.progress.stage_start(&enc_stage_label);
        let enc_stage = Instant::now();

        let mut text_encoder = if is_gguf {
            encoders::qwen3::Qwen3Encoder::load_gguf(
                &encoder_paths[0],
                &text_tokenizer_path,
                enc_device,
            )?
        } else {
            encoders::qwen3::Qwen3Encoder::load_bf16(
                &encoder_paths,
                &text_tokenizer_path,
                enc_device,
                enc_dtype,
                &self.base.progress,
            )?
        };
        self.base
            .progress
            .stage_done(&enc_stage_label, enc_stage.elapsed());

        let txt_emb = Self::encode_prompt_cached(
            &self.base.progress,
            &self.prompt_cache,
            &mut text_encoder,
            &req.prompt,
            &device,
            gpu_dtype,
        )?;

        // Drop text encoder to free memory
        drop(text_encoder);
        self.base.progress.info("Freed Qwen3 encoder");
        tracing::info!("Qwen3 encoder dropped (sequential mode)");

        // --- Phase 2: Load transformer + VAE, denoise ---
        let xformer_size = std::fs::metadata(&self.base.paths.transformer)
            .map(|m| m.len())
            .unwrap_or(0);
        let vae_file_size = std::fs::metadata(&self.base.paths.vae)
            .map(|m| m.len())
            .unwrap_or(0);
        preflight_memory_check("Flux.2 transformer + VAE", xformer_size + vae_file_size)?;
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        let flux2_cfg = self.resolve_config();
        let xformer_stage = Instant::now();
        let (transformer, xformer_label) = self.load_transformer(&flux2_cfg, gpu_dtype, &device)?;
        self.base
            .progress
            .stage_done(xformer_label, xformer_stage.elapsed());

        // Load VAE
        self.base.progress.stage_start("Loading VAE (GPU)");
        let vae_stage = Instant::now();
        let vae_cfg = Flux2VaeConfig::klein();
        let vae_vb = crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(&self.base.paths.vae),
            gpu_dtype,
            &device,
            "VAE",
            &self.base.progress,
        )?;
        let vae = Flux2AutoEncoder::new(&vae_cfg, vae_vb)?;
        self.base
            .progress
            .stage_done("Loading VAE (GPU)", vae_stage.elapsed());

        // Generate noise with seed for reproducibility
        let latent_h = height.div_ceil(8);
        let latent_w = width.div_ceil(8);
        let img =
            crate::engine::seeded_randn(seed, &[1, 32, latent_h, latent_w], &device, gpu_dtype)?;
        let state = Flux2State::new(&txt_emb, &img)?;

        // Flux.2 empirical mu schedule (resolution + step-count dependent)
        let image_seq_len = (height / 16) * (width / 16);
        let timesteps = sampling::get_schedule(req.steps as usize, image_seq_len);

        let denoise_label = format!("Denoising ({} steps)", timesteps.len() - 1);
        self.base.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        let img = transformer.denoise(
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            req.guidance,
            &self.base.progress,
        )?;

        let img = sampling::unpack(&img, height, width)?;

        self.base
            .progress
            .stage_done(&denoise_label, denoise_start.elapsed());

        // Drop transformer + state to free memory for VAE decode
        drop(transformer);
        self.base.progress.info("Freed Flux.2 transformer");
        drop(state);
        drop(txt_emb);
        device.synchronize()?;
        tracing::info!("Transformer dropped (sequential mode), decoding VAE...");

        // --- Phase 3: VAE decode ---
        self.base.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();
        let img = vae.decode(&img.to_dtype(gpu_dtype)?)?;

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

// ---------------------------------------------------------------------------
// InferenceEngine implementation
// ---------------------------------------------------------------------------

impl InferenceEngine for Flux2Engine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if req.scheduler.is_some() {
            tracing::warn!(
                "scheduler selection not supported for Flux.2 (flow-matching), ignoring"
            );
        }
        if req.guidance != 0.0 {
            tracing::debug!(
                guidance = req.guidance,
                "Flux.2 Klein is distilled — guidance value is ignored (no guidance embedding)"
            );
        }
        if req.source_image.is_some() {
            tracing::warn!("img2img not yet supported for Flux.2 — generating from text only");
        }
        if req.mask_image.is_some() {
            tracing::warn!("inpainting not yet supported for Flux.2 -- ignoring mask");
        }

        // Sequential mode: load-use-drop each component
        if self.base.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components
        // Reload transformer first (before taking the main `loaded` borrow)
        // if it was dropped after a previous VAE decode.
        self.reload_transformer_if_needed()?;

        let progress = &self.base.progress;

        let loaded = self
            .base
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting Flux.2 generation"
        );

        // 1. Encode prompt with Qwen3 (check cache first to avoid unnecessary reload)
        let cache_key = prompt_text_key(&req.prompt);
        let txt_emb = if let Some(tensor) =
            restore_cached_tensor(&self.prompt_cache, &cache_key, &loaded.device, loaded.dtype)?
        {
            progress.cache_hit("prompt conditioning");
            tensor
        } else {
            // Cache miss — reload encoder if it was dropped after a previous generation
            if loaded.text_encoder.model.is_none() {
                progress.stage_start("Reloading Qwen3 encoder");
                let reload_start = Instant::now();
                loaded.text_encoder.reload(progress)?;
                progress.stage_done("Reloading Qwen3 encoder", reload_start.elapsed());
            }

            let txt_emb = Self::encode_prompt_cached(
                progress,
                &self.prompt_cache,
                &mut loaded.text_encoder,
                &req.prompt,
                &loaded.device,
                loaded.dtype,
            )?;
            tracing::info!("Qwen3 encoding complete");

            // Drop Qwen3 to free memory for denoising.
            // Always drop on GPU. On Metal (unified memory), also drop CPU-loaded
            // weights since they share the same physical RAM as GPU allocations.
            // On CUDA, keep CPU-loaded weights resident to avoid expensive reloads.
            if loaded.text_encoder.on_gpu || loaded.device.is_metal() {
                loaded.text_encoder.drop_weights();
                tracing::info!(
                    on_gpu = loaded.text_encoder.on_gpu,
                    "Qwen3 encoder dropped to free memory for denoising"
                );
            }

            txt_emb
        };

        // 2. Generate initial noise with seed for reproducibility
        let latent_h = height.div_ceil(8);
        let latent_w = width.div_ceil(8);
        let img = crate::engine::seeded_randn(
            seed,
            &[1, 32, latent_h, latent_w],
            &loaded.device,
            loaded.dtype,
        )?;

        // 3. Build sampling state
        let state = Flux2State::new(&txt_emb, &img)?;

        // 4. Flux.2 empirical mu schedule (resolution + step-count dependent)
        let image_seq_len = (height / 16) * (width / 16);
        let timesteps = sampling::get_schedule(req.steps as usize, image_seq_len);

        let denoise_label = format!("Denoising ({} steps)", timesteps.len() - 1);
        progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();
        tracing::info!(steps = timesteps.len() - 1, "running denoising loop...");

        // 5. Denoise
        let transformer = loaded
            .transformer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("transformer not loaded"))?;
        let img = transformer.denoise(
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            req.guidance,
            progress,
        )?;

        // 6. Unpack latent to spatial
        let img = sampling::unpack(&img, height, width)?;
        progress.stage_done(&denoise_label, denoise_start.elapsed());
        tracing::info!("denoising complete, decoding VAE...");

        // Free denoising intermediates and transformer before VAE decode.
        // The transformer consumes most of VRAM — VAE decode needs that
        // memory for conv2d intermediates. Transformer is reloaded next generate.
        drop(state);
        drop(txt_emb);
        loaded.transformer = None;
        // Force CUDA to complete pending operations and release freed memory.
        // Without this, cuMemFree is asynchronous and the freed VRAM may not
        // be available when VAE decode allocates its conv2d intermediates.
        loaded.device.synchronize()?;
        tracing::info!("Transformer dropped to free VRAM for VAE decode");

        // 7. Decode with VAE
        progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();
        let img = loaded.vae.decode(&img.to_dtype(loaded.dtype)?)?;

        // 8. Convert to u8 image: clamp to [-1, 1], map to [0, 255]
        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?; // remove batch dim: [3, H, W]

        progress.stage_done("VAE decode", vae_decode_start.elapsed());
        tracing::info!("VAE decode complete, encoding output image...");

        // 9. Convert candle tensor to image bytes
        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &img,
            req.output_format,
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

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
        self.base.model_name()
    }

    fn is_loaded(&self) -> bool {
        self.base.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        Flux2Engine::load(self)
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
