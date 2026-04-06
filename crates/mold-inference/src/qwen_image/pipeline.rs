//! Qwen-Image-2512 inference engine.
//!
//! Pipeline: Qwen2.5-VL text encoder -> QwenImageTransformer2DModel -> QwenImage VAE
//!
//! Architecture follows Z-Image closely (both from Alibaba/Tongyi):
//! - Dual-stream transformer with joint attention and 3D RoPE
//! - Flow-matching Euler discrete scheduler with dynamic shifting
//! - Drop-and-reload for text encoder to manage VRAM
//! - Both Eager and Sequential loading modes
//!
//! Key differences from Z-Image:
//! - 60 identical dual-stream blocks (no noise_refiner/context_refiner)
//! - Qwen2.5-VL text encoder (hidden_size=3584) instead of Qwen3 (2560)
//! - Custom VAE with per-channel latent normalization
//! - ComfyUI-style SNR time shift scheduling (shift=3.1)

use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::z_image::postprocess_image;
use candle_transformers::quantized_var_builder;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use super::quantized_transformer::QuantizedQwenImageTransformer2DModel;
use super::sampling::{QwenImageScheduler, DEFAULT_SHIFT};
use super::transformer::{QwenImageConfig, QwenImageTransformer2DModel};
use super::vae::QwenImageVae;
use crate::cache::{
    clear_cache, prompt_text_key, CachedTensor, LruCache, DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{
    fmt_gb, free_vram_bytes, memory_status_string, preflight_memory_check, should_use_gpu,
};
use crate::encoders;
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image};
use crate::progress::{ProgressCallback, ProgressEvent, ProgressReporter};

/// Minimum free VRAM (bytes) required to place Qwen-Image VAE on GPU.
/// The VAE weights are ~300MB; decode workspace at 1024x1024 needs ~1-2GB.
const VAE_DECODE_VRAM_THRESHOLD: u64 = 2_500_000_000;

/// Minimum free VRAM for BF16 Qwen2.5-VL 7B text encoder on GPU.
/// ~14GB model + 2GB headroom.
const QWEN2_FP16_VRAM_THRESHOLD: u64 = 16_000_000_000;

/// Check if a Qwen-Image safetensors checkpoint stores weights in FP8 (F8_E4M3).
/// Uses filename pattern first, then dtype probing as fallback.
fn safetensors_is_fp8(path: &Path) -> bool {
    // Filename-based detection
    if path
        .to_str()
        .map(|s| s.contains("fp8"))
        .unwrap_or(false)
    {
        return true;
    }
    // Dtype probing — try both ComfyUI and diffusers key names
    let Ok(tensors) = (unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path]) })
    else {
        return false;
    };
    for key in ["x_embedder.weight", "img_in.weight"] {
        if let Ok(t) = tensors.load(key, &Device::Cpu) {
            return t.dtype() == DType::F8E4M3;
        }
    }
    false
}

/// Check if text encoder safetensors contain FP8 weights.
/// Uses filename pattern first (reliable for known ComfyUI FP8 models),
/// then falls back to dtype probing.
fn text_encoder_is_fp8(paths: &[std::path::PathBuf]) -> bool {
    // Filename-based detection (ComfyUI FP8 models have "fp8" in name)
    if paths.iter().any(|p| {
        p.to_str()
            .map(|s| s.contains("fp8"))
            .unwrap_or(false)
    }) {
        return true;
    }
    // Dtype probing fallback — try common key names
    let Some(first) = paths.first() else {
        return false;
    };
    let Ok(tensors) = (unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[first]) })
    else {
        return false;
    };
    for key in [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ] {
        if let Ok(t) = tensors.load(key, &Device::Cpu) {
            return t.dtype() == DType::F8E4M3;
        }
    }
    false
}

/// Loaded Qwen-Image model components, ready for inference.
struct LoadedQwenImage {
    /// Transformer wrapped in Option for drop-and-reload pattern.
    transformer: Option<QwenImageTransformer>,
    text_encoder: encoders::qwen2_text::Qwen2TextEncoder,
    vae: QwenImageVae,
    transformer_cfg: QwenImageConfig,
    /// GPU device for transformer + denoising
    device: Device,
    /// Device where the VAE lives (may be CPU if VRAM is tight)
    vae_device: Device,
    dtype: DType,
}

#[allow(clippy::large_enum_variant)]
enum QwenImageTransformer {
    BF16(QwenImageTransformer2DModel),
    Quantized(QuantizedQwenImageTransformer2DModel),
    Offloaded(super::offload::OffloadedQwenImageTransformer),
}

#[derive(Clone)]
struct CachedPromptConditioning {
    hidden_states: CachedTensor,
    valid_len: usize,
}

impl CachedPromptConditioning {
    fn from_parts(hidden_states: &Tensor, valid_len: usize) -> Result<Self> {
        Ok(Self {
            hidden_states: CachedTensor::from_tensor(hidden_states)?,
            valid_len,
        })
    }

    fn restore(&self, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        let hidden_states = self.hidden_states.restore(device, dtype)?;
        let mut mask = vec![0u8; hidden_states.dim(1)?];
        for value in &mut mask[..self.valid_len] {
            *value = 1;
        }
        let attention_mask = Tensor::from_vec(mask, (1, hidden_states.dim(1)?), device)?;
        Ok((hidden_states, attention_mask))
    }
}

impl QwenImageTransformer {
    fn forward(
        &self,
        latents: &Tensor,
        t: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Self::BF16(model) => {
                Ok(model.forward(latents, t, encoder_hidden_states, encoder_attention_mask)?)
            }
            Self::Quantized(model) => {
                Ok(model.forward(latents, t, encoder_hidden_states, encoder_attention_mask)?)
            }
            Self::Offloaded(model) => {
                model.forward(latents, t, encoder_hidden_states, encoder_attention_mask)
            }
        }
    }
}

/// Qwen-Image-2512 inference engine.
pub struct QwenImageEngine {
    base: EngineBase<LoadedQwenImage>,
    prompt_cache: Mutex<LruCache<String, CachedPromptConditioning>>,
    offload: bool,
}

impl QwenImageEngine {
    fn debug_tensor_stats(name: &str, tensor: &Tensor) {
        if std::env::var_os("MOLD_QWEN_DEBUG").is_none() {
            return;
        }
        let stats = || -> Result<String> {
            let t = tensor.to_dtype(DType::F32)?;
            let min = t.min_all()?.to_scalar::<f32>()?;
            let max = t.max_all()?.to_scalar::<f32>()?;
            let mean = t.mean_all()?.to_scalar::<f32>()?;
            // NaN detection: x != x is true for NaN (IEEE 754)
            let nan_count = t
                .ne(&t)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()? as u64;
            let total = t.elem_count();
            if nan_count > 0 {
                Ok(format!(
                    "[qwen-debug] {name}: min={min:.4} max={max:.4} mean={mean:.4} NaN={nan_count}/{total} ({:.1}%)",
                    nan_count as f64 / total as f64 * 100.0
                ))
            } else {
                Ok(format!(
                    "[qwen-debug] {name}: min={min:.4} max={max:.4} mean={mean:.4}"
                ))
            }
        };
        match stats() {
            Ok(msg) => eprintln!("{msg}"),
            Err(err) => eprintln!("[qwen-debug] {name}: <failed: {err}>"),
        }
    }

    pub fn new(
        model_name: String,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        offload: bool,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy),
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            offload,
        }
    }

    fn encode_prompt_cached(
        progress: &ProgressReporter,
        prompt_cache: &Mutex<LruCache<String, CachedPromptConditioning>>,
        text_encoder: &mut encoders::qwen2_text::Qwen2TextEncoder,
        prompt: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let cache_key = prompt_text_key(prompt);
        if let Some(cached) = prompt_cache
            .lock()
            .expect("cache poisoned")
            .get_cloned(&cache_key)
        {
            progress.cache_hit("prompt conditioning");
            return cached.restore(device, dtype);
        }

        progress.stage_start("Encoding prompt (Qwen2.5)");
        let encode_start = Instant::now();
        let (hidden_states, _attention_mask, valid_len) =
            text_encoder.encode(prompt, device, dtype)?;
        progress.stage_done("Encoding prompt (Qwen2.5)", encode_start.elapsed());

        prompt_cache.lock().expect("cache poisoned").insert(
            cache_key,
            CachedPromptConditioning::from_parts(&hidden_states, valid_len)?,
        );

        let mut mask = vec![0u8; hidden_states.dim(1)?];
        for value in &mut mask[..valid_len] {
            *value = 1;
        }
        let attention_mask = Tensor::from_vec(mask, (1, hidden_states.dim(1)?), device)?;
        Ok((hidden_states, attention_mask))
    }

    /// Resolve transformer shard paths.
    fn transformer_paths(&self) -> Vec<std::path::PathBuf> {
        if !self.base.paths.transformer_shards.is_empty() {
            self.base.paths.transformer_shards.clone()
        } else {
            vec![self.base.paths.transformer.clone()]
        }
    }

    fn detect_is_quantized(&self) -> bool {
        self.base
            .paths
            .transformer
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    }

    /// Validate required paths exist.
    fn validate_paths(&self) -> Result<std::path::PathBuf> {
        let text_tokenizer_path =
            self.base.paths.text_tokenizer.as_ref().ok_or_else(|| {
                anyhow::anyhow!("text tokenizer path required for Qwen-Image models")
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
        cfg: &QwenImageConfig,
    ) -> Result<QwenImageTransformer> {
        if self.detect_is_quantized() {
            let vb =
                quantized_var_builder::VarBuilder::from_gguf(&self.base.paths.transformer, device)?;
            Ok(QwenImageTransformer::Quantized(
                QuantizedQwenImageTransformer2DModel::new(cfg, vb, device)?,
            ))
        } else {
            let xformer_paths = self.transformer_paths();
            let is_fp8 = xformer_paths
                .first()
                .map(|p| safetensors_is_fp8(p))
                .unwrap_or(false);

            // For offload detection, use the in-memory size (BF16 = 2× FP8 on disk)
            let disk_size: u64 = xformer_paths
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .sum();
            let mem_size = if is_fp8 { disk_size * 2 } else { disk_size };
            let free = free_vram_bytes().unwrap_or(0);
            let use_offload =
                self.offload || crate::device::should_offload(mem_size, free);

            if is_fp8 {
                self.base.progress.info("Detected FP8 safetensors — loading as BF16");
            }

            if use_offload {
                // Load on CPU as BF16 for block-level streaming. For FP8, candle
                // handles F8E4M3→BF16 cast during tensor loading (no matmul on CPU
                // during loading — blocks are just stored, not computed).
                let cpu_dtype = DType::BF16;
                let label = "Qwen-Image transformer (offload)";
                let cpu_vb = crate::weight_loader::load_safetensors_with_progress(
                    &xformer_paths,
                    cpu_dtype,
                    &Device::Cpu,
                    label,
                    &self.base.progress,
                )?;
                Ok(QwenImageTransformer::Offloaded(
                    super::offload::OffloadedQwenImageTransformer::load(
                        cpu_vb,
                        cfg,
                        device,
                        &self.base.progress,
                    )?,
                ))
            } else {
                let xformer_vb = crate::weight_loader::load_safetensors_with_progress(
                    &xformer_paths,
                    dtype,
                    device,
                    "Qwen-Image transformer",
                    &self.base.progress,
                )?;
                Ok(QwenImageTransformer::BF16(
                    QwenImageTransformer2DModel::new(cfg, xformer_vb)?,
                ))
            }
        }
    }

    /// Load VAE from disk.
    fn load_vae(&self, device: &Device, dtype: DType) -> Result<QwenImageVae> {
        Ok(QwenImageVae::load(
            &self.base.paths.vae,
            device,
            dtype,
            &self.base.progress,
        )?)
    }

    /// Load text encoder from disk.
    ///
    /// FP8 text encoders are loaded on GPU with BF16 dtype — candle's CUDA cast
    /// kernel handles F8E4M3→BF16 conversion during tensor loading.
    fn load_text_encoder(
        &self,
        tokenizer_path: &std::path::PathBuf,
        device: &Device,
        dtype: DType,
    ) -> Result<encoders::qwen2_text::Qwen2TextEncoder> {
        let encoder_paths: Vec<std::path::PathBuf> = self.base.paths.text_encoder_files.clone();
        let is_fp8 = text_encoder_is_fp8(&encoder_paths);
        if is_fp8 {
            self.base
                .progress
                .info("Detected FP8 text encoder — loading as BF16 on GPU");
        }
        // FP8 and BF16 both load on the target device with the target dtype.
        // The VarBuilder handles F8E4M3→BF16 casting during tensor loading.
        encoders::qwen2_text::Qwen2TextEncoder::load_bf16(
            &encoder_paths,
            tokenizer_path,
            device,
            dtype,
            &self.base.progress,
        )
    }

    /// Resolve text encoder device placement.
    fn resolve_text_encoder_device(&self, gpu_device: &Device, free_vram: u64) -> (bool, String) {
        let is_cuda = gpu_device.is_cuda();
        let is_metal = gpu_device.is_metal();
        let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, QWEN2_FP16_VRAM_THRESHOLD);
        let label = if on_gpu { "GPU" } else { "CPU" };
        if !on_gpu && (is_cuda || is_metal) {
            self.base.progress.info(&format!(
                "Qwen2.5 text encoder on CPU ({} free < {} threshold)",
                fmt_gb(free_vram),
                fmt_gb(QWEN2_FP16_VRAM_THRESHOLD),
            ));
        }
        (on_gpu, label.to_string())
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

        tracing::info!(model = %self.base.model_name, "loading Qwen-Image model components...");

        let text_tokenizer_path = self.validate_paths()?;
        let device = crate::device::create_device(&self.base.progress)?;
        let transformer_cfg = QwenImageConfig::qwen_image_2512();
        let transformer_is_quantized = self.detect_is_quantized();
        // FP8 safetensors are loaded as BF16 via CPU (candle CUDA kernel bug
        // prevents direct F8E4M3→BF16 on GPU; CPU cast works fine). All paths
        // use BF16 as runtime dtype since the model trains and computes in BF16.
        let dtype = crate::engine::gpu_dtype(&device);

        // Load transformer
        let xformer_paths = self.transformer_paths();
        let xformer_label = if transformer_is_quantized {
            "Loading Qwen-Image transformer (quantized)".to_string()
        } else {
            format!(
                "Loading Qwen-Image transformer ({} shards)",
                xformer_paths.len()
            )
        };
        self.base.progress.stage_start(&xformer_label);
        let xformer_start = Instant::now();
        let transformer = self.load_transformer(&device, dtype, &transformer_cfg)?;
        self.base
            .progress
            .stage_done(&xformer_label, xformer_start.elapsed());
        tracing::info!("Qwen-Image transformer loaded");

        // Decide device placement for VAE and text encoder
        let free = free_vram_bytes().unwrap_or(0);
        let is_cuda = device.is_cuda();
        let is_metal = device.is_metal();
        if free > 0 {
            self.base
                .progress
                .info(&format!("Free VRAM after transformer: {}", fmt_gb(free)));
        }

        let vae_on_gpu = should_use_gpu(is_cuda, is_metal, free, VAE_DECODE_VRAM_THRESHOLD);
        let vae_device = if vae_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        // Always decode in F32 — BF16 convolutions accumulate quantization noise across
        // the 4 upsampling blocks, producing visible grain. Matches diffusers' force_upcast.
        let vae_dtype = DType::F32;
        let vae_device_label = if vae_on_gpu { "GPU" } else { "CPU" };

        // Load VAE
        let vae_label = format!("Loading Qwen-Image VAE ({}, F32)", vae_device_label);
        self.base.progress.stage_start(&vae_label);
        let vae_start = Instant::now();
        let vae = self.load_vae(&vae_device, vae_dtype)?;
        self.base
            .progress
            .stage_done(&vae_label, vae_start.elapsed());

        // Load text encoder
        let (te_on_gpu, te_device_label) = self.resolve_text_encoder_device(&device, free);
        let te_device = if te_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let te_dtype = if te_on_gpu { dtype } else { DType::F32 };

        let te_label = format!(
            "Loading Qwen2.5 text encoder ({} shards, {})",
            self.base.paths.text_encoder_files.len(),
            te_device_label,
        );
        self.base.progress.stage_start(&te_label);
        let te_start = Instant::now();
        let text_encoder = self.load_text_encoder(&text_tokenizer_path, &te_device, te_dtype)?;
        self.base.progress.stage_done(&te_label, te_start.elapsed());
        tracing::info!(device = %te_device_label, "Qwen2.5 text encoder loaded");

        self.base.loaded = Some(LoadedQwenImage {
            transformer: Some(transformer),
            text_encoder,
            vae,
            transformer_cfg,
            device,
            vae_device,
            dtype,
        });

        tracing::info!(model = %self.base.model_name, "all Qwen-Image components loaded");
        Ok(())
    }

    /// Reload the transformer from disk.
    fn reload_transformer(&self, loaded: &mut LoadedQwenImage) -> Result<()> {
        let transformer =
            self.load_transformer(&loaded.device, loaded.dtype, &loaded.transformer_cfg)?;
        loaded.transformer = Some(transformer);
        Ok(())
    }

    /// Generate using sequential loading strategy (load-use-drop each component).
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let text_tokenizer_path = self.validate_paths()?;
        let transformer_cfg = QwenImageConfig::qwen_image_2512();

        let device = crate::device::create_device(&self.base.progress)?;
        let dtype = crate::engine::gpu_dtype(&device);
        let transformer_is_quantized = self.detect_is_quantized();

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting sequential Qwen-Image generation"
        );

        self.base
            .progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: Text encoding (check cache first to skip encoder load) ---
        let use_cfg = req.guidance > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let prompt_key = prompt_text_key(&req.prompt);
        let uncond_key = prompt_text_key(negative);
        let prompt_cached = self
            .prompt_cache
            .lock()
            .expect("cache poisoned")
            .get_cloned(&prompt_key);
        let uncond_cached = if use_cfg {
            self.prompt_cache
                .lock()
                .expect("cache poisoned")
                .get_cloned(&uncond_key)
        } else {
            None
        };
        let both_cached = prompt_cached.is_some() && (!use_cfg || uncond_cached.is_some());

        let (encoder_hidden_states, encoder_attention_mask, uncond_hs, uncond_mask) = if both_cached
        {
            self.base.progress.cache_hit("prompt conditioning");
            let cached = prompt_cached.unwrap();
            let (hs, mask) = cached.restore(&device, dtype)?;
            let (u_hs, u_mask) = if use_cfg {
                let ucached = uncond_cached.unwrap();
                let (u_hs, u_mask) = ucached.restore(&device, dtype)?;
                (Some(u_hs), Some(u_mask))
            } else {
                (None, None)
            };
            (hs, mask, u_hs, u_mask)
        } else {
            let free = free_vram_bytes().unwrap_or(0);
            let (te_on_gpu, te_device_label) = self.resolve_text_encoder_device(&device, free);
            let te_device = if te_on_gpu {
                device.clone()
            } else {
                Device::Cpu
            };
            let te_dtype = if te_on_gpu { dtype } else { DType::F32 };

            let te_label = format!(
                "Loading Qwen2.5 text encoder ({} shards, {})",
                self.base.paths.text_encoder_files.len(),
                te_device_label,
            );
            let te_size: u64 = self
                .base
                .paths
                .text_encoder_files
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .sum();
            preflight_memory_check("Qwen2.5 text encoder", te_size)?;

            if let Some(status) = memory_status_string() {
                self.base.progress.info(&status);
            }

            self.base.progress.stage_start(&te_label);
            let te_start = Instant::now();
            let mut text_encoder =
                self.load_text_encoder(&text_tokenizer_path, &te_device, te_dtype)?;
            self.base.progress.stage_done(&te_label, te_start.elapsed());

            let (hs, mask) = Self::encode_prompt_cached(
                &self.base.progress,
                &self.prompt_cache,
                &mut text_encoder,
                &req.prompt,
                &device,
                dtype,
            )?;

            let (u_hs, u_mask) = if use_cfg {
                let (hs, mask) = Self::encode_prompt_cached(
                    &self.base.progress,
                    &self.prompt_cache,
                    &mut text_encoder,
                    negative,
                    &device,
                    dtype,
                )?;
                (Some(hs), Some(mask))
            } else {
                (None, None)
            };

            drop(text_encoder);
            self.base.progress.info("Freed Qwen2.5 text encoder");

            (hs, mask, u_hs, u_mask)
        };

        // --- Phase 2: Load transformer and denoise ---
        let xformer_paths = self.transformer_paths();
        let xformer_size: u64 = xformer_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        preflight_memory_check("Qwen-Image transformer", xformer_size)?;

        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        let xformer_label = if transformer_is_quantized {
            "Loading Qwen-Image transformer (quantized)".to_string()
        } else {
            format!(
                "Loading Qwen-Image transformer ({} shards)",
                xformer_paths.len()
            )
        };
        self.base.progress.stage_start(&xformer_label);
        let xformer_start = Instant::now();
        let transformer = self.load_transformer(&device, dtype, &transformer_cfg)?;
        self.base
            .progress
            .stage_done(&xformer_label, xformer_start.elapsed());

        // Calculate latent dimensions: image_size / 8 (VAE downsample factor)
        let vae_downsample = 8;
        let latent_h = height / vae_downsample;
        let latent_w = width / vae_downsample;

        let mut scheduler = QwenImageScheduler::new(req.steps as usize, DEFAULT_SHIFT);

        // Initial noise scaled by sigma[0] — matches ComfyUI's CONST noise scaling.
        // For txt2img: initial_sample = sigma[0] * randn()
        let mut latents =
            crate::engine::seeded_randn(seed, &[1, 16, latent_h, latent_w], &device, dtype)?;
        latents = (latents * scheduler.initial_sigma())?;

        let num_steps = req.steps as usize;
        let denoise_label = format!("Denoising ({} steps)", num_steps);
        self.base.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        if std::env::var_os("MOLD_QWEN_DEBUG").is_some() {
            eprintln!(
                "[qwen-debug] cfg={} guidance={:.1} shift={:.2} sigmas[0]={:.4} sigmas[last]={:.4}",
                use_cfg,
                req.guidance,
                DEFAULT_SHIFT,
                scheduler.sigmas[0],
                scheduler.sigmas[num_steps],
            );
        }

        // Pre-batch CFG inputs to halve block transfers for GGUF streaming.
        // Each forward pass streams 60 blocks CPU→GPU; batching cond+uncond avoids
        // doing that twice per step.
        let (batched_hs, batched_mask) = if use_cfg {
            let hs = Tensor::cat(&[&encoder_hidden_states, uncond_hs.as_ref().unwrap()], 0)?;
            let mask = Tensor::cat(&[&encoder_attention_mask, uncond_mask.as_ref().unwrap()], 0)?;
            (hs, mask)
        } else {
            (
                encoder_hidden_states.clone(),
                encoder_attention_mask.clone(),
            )
        };

        for step in 0..num_steps {
            let step_start = Instant::now();
            let t = scheduler.current_timestep();
            let noise_pred = if use_cfg {
                // For offloaded transformer: separate cond/uncond passes to halve peak VRAM
                let is_offloaded = matches!(transformer, QwenImageTransformer::Offloaded(_));
                let (cond_pred, uncond_pred) = if is_offloaded {
                    let t_tensor =
                        Tensor::from_vec(vec![t as f32], (1,), &device)?.to_dtype(dtype)?;
                    let cond_pred = transformer.forward(
                        &latents,
                        &t_tensor,
                        &encoder_hidden_states,
                        &encoder_attention_mask,
                    )?;
                    let uncond_pred = transformer.forward(
                        &latents,
                        &t_tensor,
                        uncond_hs.as_ref().unwrap(),
                        uncond_mask.as_ref().unwrap(),
                    )?;
                    (cond_pred, uncond_pred)
                } else {
                    let t_tensor =
                        Tensor::from_vec(vec![t as f32; 2], (2,), &device)?.to_dtype(dtype)?;
                    let batched_latents = Tensor::cat(&[&latents, &latents], 0)?;
                    let batched_pred = transformer.forward(
                        &batched_latents,
                        &t_tensor,
                        &batched_hs,
                        &batched_mask,
                    )?;
                    let cond_pred = batched_pred.narrow(0, 0, 1)?;
                    let uncond_pred = batched_pred.narrow(0, 1, 1)?;
                    (cond_pred, uncond_pred)
                };
                if step == 0 {
                    Self::debug_tensor_stats("cond_pred[0]", &cond_pred);
                    Self::debug_tensor_stats("uncond_pred[0]", &uncond_pred);
                }
                // CFG in F32 to avoid BF16 cancellation error
                let cond_f32 = cond_pred.to_dtype(DType::F32)?;
                let uncond_f32 = uncond_pred.to_dtype(DType::F32)?;
                let noise_pred = (&uncond_f32 + ((&cond_f32 - &uncond_f32)? * req.guidance)?)?;
                noise_pred.to_dtype(dtype)?
            } else {
                let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &device)?.to_dtype(dtype)?;
                transformer.forward(
                    &latents,
                    &t_tensor,
                    &encoder_hidden_states,
                    &encoder_attention_mask,
                )?
            };
            if step == 0 || step == num_steps / 2 || step == num_steps - 1 {
                Self::debug_tensor_stats(&format!("noise_pred[{step}]"), &noise_pred);
                Self::debug_tensor_stats(&format!("latents[{step}]"), &latents);
            }
            latents = scheduler.step(&noise_pred, &latents)?;
            if std::env::var_os("MOLD_QWEN_DEBUG").is_some() {
                let n = latents
                    .ne(&latents)?
                    .to_dtype(candle_core::DType::U32)?
                    .sum_all()?
                    .to_scalar::<u32>()?;
                if n > 0 {
                    eprintln!(
                        "[qwen-nan] NaN in latents AFTER step {step}: {n}/{}",
                        latents.elem_count()
                    );
                }
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

        // Drop transformer and embeddings
        drop(transformer);
        drop(encoder_hidden_states);
        drop(encoder_attention_mask);
        drop(uncond_hs);
        drop(uncond_mask);
        device.synchronize()?;
        self.base.progress.info("Freed Qwen-Image transformer");

        // --- Phase 3: Load VAE and decode ---
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        let free_for_vae = free_vram_bytes().unwrap_or(0);
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
        // Always decode in F32 — BF16 convolutions accumulate quantization noise across
        // the 4 upsampling blocks, producing visible grain. Matches diffusers' force_upcast.
        let vae_dtype = DType::F32;
        let vae_device_label = if vae_on_gpu { "GPU" } else { "CPU" };

        let vae_label = format!("Loading Qwen-Image VAE ({}, F32)", vae_device_label);
        self.base.progress.stage_start(&vae_label);
        let vae_start = Instant::now();
        let vae = self.load_vae(&vae_device, vae_dtype)?;
        self.base
            .progress
            .stage_done(&vae_label, vae_start.elapsed());

        self.base.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();

        let latents = latents.to_device(&vae_device)?.to_dtype(vae_dtype)?;
        Self::debug_tensor_stats("latents_pre_vae", &latents);
        let image = vae.decode(&latents)?;
        Self::debug_tensor_stats("image_pre_postprocess", &image);
        let image = postprocess_image(&image)?;
        Self::debug_tensor_stats("image_postprocess", &image);
        let image = image.i(0)?;

        self.base
            .progress
            .stage_done("VAE decode", vae_decode_start.elapsed());

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
            "sequential Qwen-Image generation complete"
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

impl InferenceEngine for QwenImageEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if req.scheduler.is_some() {
            tracing::warn!(
                "scheduler selection not supported for Qwen-Image (flow-matching), ignoring"
            );
        }
        if req.source_image.is_some() {
            tracing::warn!("img2img not yet supported for Qwen-Image — generating from text only");
        }
        if req.mask_image.is_some() {
            tracing::warn!("inpainting not yet supported for Qwen-Image -- ignoring mask");
        }

        // Sequential mode: load-use-drop each component
        if self.base.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components
        if self.base.loaded.is_none() {
            bail!("model not loaded -- call load() first");
        }

        let progress = &self.base.progress;
        let start = Instant::now();

        // Reload transformer if it was dropped after previous VAE decode
        let loaded_ref = self
            .base
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let needs_reload = loaded_ref.transformer.is_none();
        if needs_reload {
            let mut loaded_mut = self
                .base
                .loaded
                .take()
                .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
            progress.stage_start("Reloading Qwen-Image transformer");
            let reload_start = Instant::now();
            self.reload_transformer(&mut loaded_mut)?;
            progress.stage_done("Reloading Qwen-Image transformer", reload_start.elapsed());
            self.base.loaded = Some(loaded_mut);
        }

        let loaded = self
            .base
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting Qwen-Image generation"
        );

        // 1. Reload text encoder if weights were dropped
        if loaded.text_encoder.model.is_none() {
            progress.stage_start("Reloading Qwen2.5 encoder");
            let reload_start = Instant::now();
            loaded.text_encoder.reload(progress)?;
            progress.stage_done("Reloading Qwen2.5 encoder", reload_start.elapsed());
        }

        // 2. Encode prompt
        let (encoder_hidden_states, encoder_attention_mask) = Self::encode_prompt_cached(
            progress,
            &self.prompt_cache,
            &mut loaded.text_encoder,
            &req.prompt,
            &loaded.device,
            loaded.dtype,
        )?;

        // Encode unconditional (negative) prompt for classifier-free guidance
        let use_cfg = req.guidance > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let (uncond_hs, uncond_mask) = if use_cfg {
            let (hs, mask) = Self::encode_prompt_cached(
                progress,
                &self.prompt_cache,
                &mut loaded.text_encoder,
                negative,
                &loaded.device,
                loaded.dtype,
            )?;
            (Some(hs), Some(mask))
        } else {
            (None, None)
        };

        // Drop text encoder from GPU to free VRAM for denoising
        if loaded.text_encoder.on_gpu {
            loaded.text_encoder.drop_weights();
            tracing::info!("Qwen2.5 text encoder dropped from GPU");
        }

        // 3. Calculate latent dimensions
        let vae_downsample = 8;
        let latent_h = height / vae_downsample;
        let latent_w = width / vae_downsample;

        // 4. Initialize scheduler (ComfyUI-style SNR shift)
        let mut scheduler = QwenImageScheduler::new(req.steps as usize, DEFAULT_SHIFT);

        // 5. Generate initial noise scaled by sigma[0] (ComfyUI CONST noise scaling)
        let mut latents = crate::engine::seeded_randn(
            seed,
            &[1, 16, latent_h, latent_w],
            &loaded.device,
            loaded.dtype,
        )?;
        latents = (latents * scheduler.initial_sigma())?;

        // 7. Denoising loop
        let num_steps = req.steps as usize;
        let denoise_label = format!("Denoising ({} steps)", num_steps);
        progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        {
            let transformer = loaded
                .transformer
                .as_ref()
                .expect("transformer must be loaded for denoising");

            // Pre-batch CFG inputs to halve block transfers for GGUF streaming.
            let (batched_hs, batched_mask) = if use_cfg {
                let hs = Tensor::cat(&[&encoder_hidden_states, uncond_hs.as_ref().unwrap()], 0)?;
                let mask =
                    Tensor::cat(&[&encoder_attention_mask, uncond_mask.as_ref().unwrap()], 0)?;
                (hs, mask)
            } else {
                (
                    encoder_hidden_states.clone(),
                    encoder_attention_mask.clone(),
                )
            };

            for step in 0..num_steps {
                let step_start = Instant::now();
                let t = scheduler.current_timestep();
                let noise_pred = if use_cfg {
                    // For offloaded transformer: separate cond/uncond passes to halve peak VRAM
                    let is_offloaded = matches!(transformer, QwenImageTransformer::Offloaded(_));
                    let (cond_pred, uncond_pred) = if is_offloaded {
                        let t_tensor =
                            Tensor::from_vec(vec![t as f32], (1,), &loaded.device)?
                                .to_dtype(loaded.dtype)?;
                        let cond_pred = transformer.forward(
                            &latents,
                            &t_tensor,
                            &encoder_hidden_states,
                            &encoder_attention_mask,
                        )?;
                        let uncond_pred = transformer.forward(
                            &latents,
                            &t_tensor,
                            uncond_hs.as_ref().unwrap(),
                            uncond_mask.as_ref().unwrap(),
                        )?;
                        (cond_pred, uncond_pred)
                    } else {
                        let t_tensor =
                            Tensor::from_vec(vec![t as f32; 2], (2,), &loaded.device)?
                                .to_dtype(loaded.dtype)?;
                        let batched_latents = Tensor::cat(&[&latents, &latents], 0)?;
                        let batched_pred = transformer.forward(
                            &batched_latents,
                            &t_tensor,
                            &batched_hs,
                            &batched_mask,
                        )?;
                        let cond_pred = batched_pred.narrow(0, 0, 1)?;
                        let uncond_pred = batched_pred.narrow(0, 1, 1)?;
                        (cond_pred, uncond_pred)
                    };
                    // CFG in F32 to avoid BF16 cancellation error
                    let cond_f32 = cond_pred.to_dtype(DType::F32)?;
                    let uncond_f32 = uncond_pred.to_dtype(DType::F32)?;
                    let noise_pred = (&uncond_f32 + ((&cond_f32 - &uncond_f32)? * req.guidance)?)?;
                    noise_pred.to_dtype(loaded.dtype)?
                } else {
                    let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &loaded.device)?
                        .to_dtype(loaded.dtype)?;
                    transformer.forward(
                        &latents,
                        &t_tensor,
                        &encoder_hidden_states,
                        &encoder_attention_mask,
                    )?
                };
                if step == 0 {
                    Self::debug_tensor_stats("noise_pred", &noise_pred);
                }
                latents = scheduler.step(&noise_pred, &latents)?;
                progress.emit(ProgressEvent::DenoiseStep {
                    step: step + 1,
                    total: num_steps,
                    elapsed: step_start.elapsed(),
                });
            }
        }

        progress.stage_done(&denoise_label, denoise_start.elapsed());

        // Free text embeddings and transformer
        drop(encoder_hidden_states);
        drop(encoder_attention_mask);
        drop(uncond_hs);
        drop(uncond_mask);
        loaded.transformer = None;
        // Synchronize to ensure CUDA's caching allocator reclaims the freed memory
        // before VAE decode allocates workspace buffers.
        loaded.device.synchronize()?;
        tracing::info!("Qwen-Image transformer dropped to free VRAM for VAE decode");

        // 8. VAE decode
        progress.stage_start("VAE decode");
        let vae_start = Instant::now();

        // Always decode in F32 — matches sequential path and diffusers' force_upcast.
        let latents = latents
            .to_device(&loaded.vae_device)?
            .to_dtype(DType::F32)?;
        Self::debug_tensor_stats("latents_pre_vae", &latents);
        let image = loaded.vae.decode(&latents)?;
        Self::debug_tensor_stats("image_pre_postprocess", &image);
        let image = postprocess_image(&image)?;
        Self::debug_tensor_stats("image_postprocess", &image);
        let image = image.i(0)?;

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
        tracing::info!(generation_time_ms, seed, "Qwen-Image generation complete");

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
        QwenImageEngine::load(self)
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
    use std::path::PathBuf;

    #[test]
    fn qwen_image_detects_gguf_transformer() {
        let engine = QwenImageEngine::new(
            "qwen-image:q4".to_string(),
            ModelPaths {
                transformer: PathBuf::from("/tmp/qwen-image-Q4_K_S.gguf"),
                transformer_shards: vec![],
                vae: PathBuf::from("/tmp/vae.safetensors"),
                t5_encoder: None,
                clip_encoder: None,
                t5_tokenizer: None,
                clip_tokenizer: None,
                clip_encoder_2: None,
                clip_tokenizer_2: None,
                text_encoder_files: vec![],
                text_tokenizer: Some(PathBuf::from("/tmp/tokenizer.json")),
                decoder: None,
            },
            LoadStrategy::Sequential,
            false,
        );

        assert!(engine.detect_is_quantized());
    }
}
