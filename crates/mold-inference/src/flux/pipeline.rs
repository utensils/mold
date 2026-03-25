use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Shape};
use candle_nn::VarBuilder;
use candle_transformers::models::flux;
use candle_transformers::quantized_var_builder;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use crate::cache::{
    clear_cache, prompt_text_key, restore_cached_tensor_pair, store_cached_tensor_pair,
    CachedTensorPair, LruCache, DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{
    check_memory_budget, fmt_gb, free_vram_bytes, memory_status_string, preflight_memory_check,
    should_use_gpu, CLIP_VRAM_THRESHOLD,
};
use crate::encoders;
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy, OptionRestoreGuard};
use crate::image::{build_output_metadata, encode_image};
use crate::progress::{ProgressCallback, ProgressReporter};

use super::transformer::FluxTransformer;

/// Some FLUX safetensors checkpoints store transformer tensors at the root
/// while others nest them under `model.diffusion_model`.
fn flux_transformer_var_builder<'a>(vb: VarBuilder<'a>) -> VarBuilder<'a> {
    if vb.contains_tensor("img_in.weight") {
        vb
    } else if vb.contains_tensor("model.diffusion_model.img_in.weight") {
        vb.pp("model.diffusion_model")
    } else if vb.contains_tensor("diffusion_model.img_in.weight") {
        vb.pp("diffusion_model")
    } else {
        vb
    }
}

fn flux_safetensors_transformer_is_fp8(path: &std::path::Path) -> Result<bool> {
    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path])? };
    for key in [
        "img_in.weight",
        "model.diffusion_model.img_in.weight",
        "diffusion_model.img_in.weight",
    ] {
        if let Ok(view) = tensors.get(key) {
            return Ok(format!("{:?}", view.dtype()) == "F8_E4M3");
        }
    }
    Ok(false)
}

fn flux_runtime_dtype(is_cuda: bool, is_quantized: bool, transformer_is_fp8: bool) -> DType {
    if is_quantized {
        DType::BF16
    } else if is_cuda && transformer_is_fp8 {
        // FP8 safetensors must go through F16 on CUDA (candle has a kernel naming
        // bug that prevents direct CUDA FP8→BF16 casts). Loading uses
        // CpuStagedSafetensors which does FP8→F16 on CPU then transfers to GPU.
        DType::F16
    } else if is_cuda {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Path for the Q8 GGUF cache of an FP8 safetensors file.
fn fp8_gguf_cache_path(path: &Path) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("transformer");
    let cache_root = mold_core::Config::mold_dir()
        .unwrap_or_else(|| PathBuf::from(".mold"))
        .join("cache")
        .join("flux-q8");
    cache_root.join(format!("{stem}.q8_0.gguf"))
}

/// Convert an FP8 safetensors checkpoint to Q8_0 GGUF (one-time).
///
/// FP8 safetensors cannot run directly through candle on a 24 GB card because
/// expanding to F16/BF16 doubles the VRAM requirement. Q8_0 GGUF keeps the
/// model at ~12 GB and uses candle's efficient quantized matmul path.
fn ensure_fp8_gguf_cache(path: &Path, progress: &ProgressReporter) -> Result<PathBuf> {
    let cache_path = fp8_gguf_cache_path(path);
    if cache_path.exists() {
        progress.info(&format!("Using cached Q8 GGUF: {}", cache_path.display()));
        return Ok(cache_path);
    }

    let parent = cache_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("invalid cache path: {}", cache_path.display()))?;
    std::fs::create_dir_all(parent)?;

    progress.info("Converting FP8 checkpoint to Q8 GGUF cache (one-time, may take a few minutes)");
    tracing::info!(
        source = %path.display(),
        cache = %cache_path.display(),
        "converting FP8 safetensors to Q8_0 GGUF cache"
    );

    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path])? };

    // Detect and strip the common prefix used in some checkpoints
    let prefix = if tensors.get("img_in.weight").is_ok() {
        ""
    } else if tensors.get("model.diffusion_model.img_in.weight").is_ok() {
        "model.diffusion_model."
    } else if tensors.get("diffusion_model.img_in.weight").is_ok() {
        "diffusion_model."
    } else {
        ""
    };

    // Enumerate all tensor names via MmapedSafetensors::tensors()
    let all_names: Vec<String> = tensors
        .tensors()
        .into_iter()
        .map(|(name, _)| name)
        .collect();

    let block_size = candle_core::quantized::GgmlDType::Q8_0.block_size();
    let mut qtensors: Vec<(String, candle_core::quantized::QTensor)> = Vec::new();

    let total = all_names.len();
    for (i, name) in all_names.iter().enumerate() {
        if (i + 1) % 50 == 0 || i + 1 == total {
            progress.info(&format!("Quantizing tensor {}/{total}", i + 1));
        }

        let tensor = tensors.load(name, &Device::Cpu)?;
        // Strip prefix for GGUF (quantized model expects unprefixed names)
        let out_name = if !prefix.is_empty() && name.starts_with(prefix) {
            name[prefix.len()..].to_string()
        } else {
            name.clone()
        };

        let elem_count = tensor.elem_count();
        let can_quantize = elem_count >= block_size && elem_count % block_size == 0;

        let qt = if can_quantize {
            candle_core::quantized::QTensor::quantize(
                &tensor,
                candle_core::quantized::GgmlDType::Q8_0,
            )?
        } else {
            // Small/odd-shaped tensors (norms, biases): store as F32
            candle_core::quantized::QTensor::quantize(
                &tensor,
                candle_core::quantized::GgmlDType::F32,
            )?
        };
        qtensors.push((out_name, qt));
    }

    // Write GGUF cache (clean up temp file on error)
    let tmp_path = cache_path.with_extension("tmp");
    let write_result = (|| -> Result<()> {
        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = std::io::BufWriter::new(file);
        let tensor_refs: Vec<(&str, &candle_core::quantized::QTensor)> =
            qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
        candle_core::quantized::gguf_file::write(&mut writer, &[], &tensor_refs)?;
        Ok(())
    })();
    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    std::fs::rename(&tmp_path, &cache_path)?;

    progress.info(&format!("Q8 GGUF cache created: {}", cache_path.display()));
    tracing::info!(cache = %cache_path.display(), "FP8→Q8_0 GGUF cache created");
    Ok(cache_path)
}

/// VarBuilder backend that loads tensors on CPU first, converts dtype, then
/// moves to GPU. Works around candle's broken CUDA FP8 cast kernels (the PTX
/// symbols use underscored names `cast_f8_e4m3_f16` but the Rust side constructs
/// `cast_f8e4m3_f16`).  Only used for FP8 safetensors models.
struct CpuStagedSafetensors {
    inner: candle_core::safetensors::MmapedSafetensors,
}

impl candle_nn::var_builder::SimpleBackend for CpuStagedSafetensors {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        let tensor = self
            .inner
            .load(name, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        self.inner
            .load(name, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.get(name).is_ok()
    }
}

fn flux_safetensors_var_builder<'a>(
    path: &std::path::Path,
    dtype: DType,
    device: &Device,
    fp8: bool,
) -> Result<VarBuilder<'a>> {
    if fp8 && device.is_cuda() {
        // FP8→target dtype conversion must happen on CPU because candle's CUDA
        // backend has a kernel naming mismatch for F8_E4M3 casts.
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path])? };
        Ok(VarBuilder::from_backend(
            Box::new(CpuStagedSafetensors { inner: tensors }),
            dtype,
            device.clone(),
        ))
    } else {
        Ok(unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&path), dtype, device)?
        })
    }
}

/// Loaded FLUX model components, ready for inference.
/// FLUX transformer and VAE always run on GPU. T5 and CLIP run on GPU or CPU
/// depending on available VRAM (checked at load time after the transformer is loaded).
/// When T5/CLIP are loaded on GPU, they are dropped after encoding to free VRAM
/// for the denoising pass (their weights are only needed for prompt encoding).
struct LoadedFlux {
    /// None after being dropped for VAE decode VRAM; reloaded on next generate.
    flux_model: Option<FluxTransformer>,
    t5: encoders::t5::T5Encoder,
    clip: encoders::clip::ClipEncoder,
    vae: flux::autoencoder::AutoEncoder,
    /// GPU device for FLUX transformer + VAE
    device: Device,
    dtype: DType,
    is_schnell: bool,
    /// True if using quantized GGUF model (state tensors must be F32)
    is_quantized: bool,
    /// Resolved transformer path (may be a GGUF cache for FP8 models).
    transformer_path: PathBuf,
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
    prompt_cache: Mutex<LruCache<String, CachedTensorPair>>,
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
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
        }
    }

    fn restore_prompt_cache(
        progress: &ProgressReporter,
        prompt_cache: &Mutex<LruCache<String, CachedTensorPair>>,
        prompt: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Option<(candle_core::Tensor, candle_core::Tensor)>> {
        let restored =
            restore_cached_tensor_pair(prompt_cache, &prompt_text_key(prompt), device, dtype)?;
        let Some(restored) = restored else {
            return Ok(None);
        };
        progress.cache_hit("prompt conditioning");
        Ok(Some(restored))
    }

    fn store_prompt_cache(
        prompt_cache: &Mutex<LruCache<String, CachedTensorPair>>,
        prompt: &str,
        t5_emb: &candle_core::Tensor,
        clip_emb: &candle_core::Tensor,
    ) -> Result<()> {
        store_cached_tensor_pair(prompt_cache, prompt_text_key(prompt), t5_emb, clip_emb)
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
    ///
    /// On error, `self.loaded` remains `None` — all components are assembled into
    /// local variables and only stored in `self.loaded` on success, so partial loads
    /// cannot leave the engine in an inconsistent state.
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
        let mut is_quantized = self.detect_is_quantized();
        let transformer_is_fp8 = !is_quantized
            && flux_safetensors_transformer_is_fp8(&self.paths.transformer).unwrap_or(false);

        // FP8 safetensors → Q8 GGUF cache: candle lacks native FP8 compute and
        // expanding to F16 doubles VRAM (OOM on 24 GB). Q8 GGUF keeps the model
        // compact (~12 GB) and uses candle's efficient quantized matmul.
        let transformer_path = if transformer_is_fp8 {
            let p = ensure_fp8_gguf_cache(&self.paths.transformer, &self.progress)?;
            is_quantized = true;
            p
        } else {
            self.paths.transformer.clone()
        };

        let gpu_dtype = flux_runtime_dtype(device.is_cuda(), is_quantized, false);

        tracing::info!("GPU device: {:?}, GPU dtype: {:?}", device, gpu_dtype);

        // --- Load FLUX transformer + VAE on GPU first (variable size) ---
        // This must happen before T5/CLIP so we can measure remaining VRAM.

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
            path = %transformer_path.display(),
            quantized = is_quantized,
            "loading FLUX transformer on GPU..."
        );

        let flux_model = if is_quantized {
            let vb = quantized_var_builder::VarBuilder::from_gguf(&transformer_path, &device)?;
            FluxTransformer::Quantized(flux::quantized_model::Flux::new(&flux_cfg, vb)?)
        } else {
            let flux_vb = flux_transformer_var_builder(flux_safetensors_var_builder(
                &transformer_path,
                gpu_dtype,
                &device,
                false,
            )?);
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
            crate::encoders::variant_resolution::resolve_t5_variant(
                &self.progress,
                t5_preference,
                &device,
                free,
                &t5_encoder_path,
            )?;
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
            flux_model: Some(flux_model),
            t5,
            clip,
            vae,
            device,
            dtype: gpu_dtype,
            is_schnell,
            is_quantized,
            transformer_path,
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
        let mut is_quantized = self.detect_is_quantized();

        let (t5_encoder_path, t5_tokenizer_path, clip_encoder_path, clip_tokenizer_path) =
            self.validate_paths()?;

        // Check memory budget
        if let Some(warning) = check_memory_budget(&self.paths, LoadStrategy::Sequential) {
            self.progress.info(&warning);
        }

        let device = crate::device::create_device(&self.progress)?;
        let transformer_is_fp8 = !is_quantized
            && flux_safetensors_transformer_is_fp8(&self.paths.transformer).unwrap_or(false);

        // FP8 safetensors → Q8 GGUF cache (same as eager path)
        let transformer_path = if transformer_is_fp8 {
            let p = ensure_fp8_gguf_cache(&self.paths.transformer, &self.progress)?;
            is_quantized = true;
            p
        } else {
            self.paths.transformer.clone()
        };

        let gpu_dtype = flux_runtime_dtype(device.is_cuda(), is_quantized, false);

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

        let (t5_emb, clip_emb) = if let Some((t5_emb, clip_emb)) = Self::restore_prompt_cache(
            &self.progress,
            &self.prompt_cache,
            &req.prompt,
            &device,
            gpu_dtype,
        )? {
            (t5_emb, clip_emb)
        } else {
            // --- Phase 1: T5 encoding ---
            let free = free_vram_bytes().unwrap_or(0);
            self.progress.stage_start("Selecting T5 encoder");
            let t5_resolve_start = Instant::now();
            let t5_preference = self.t5_variant.as_deref();
            let (resolved_t5_path, t5_on_gpu, t5_device_label) =
                crate::encoders::variant_resolution::resolve_t5_variant(
                    &self.progress,
                    t5_preference,
                    &device,
                    free,
                    &t5_encoder_path,
                )?;
            self.progress
                .stage_done("Selecting T5 encoder", t5_resolve_start.elapsed());

            let t5_device = if t5_on_gpu { &device } else { &Device::Cpu };
            let t5_dtype = if t5_on_gpu { gpu_dtype } else { DType::F32 };

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

            self.progress.info("Freed CLIP encoder");
            tracing::info!("CLIP encoder dropped (sequential mode)");

            Self::store_prompt_cache(&self.prompt_cache, &req.prompt, &t5_emb, &clip_emb)?;
            (t5_emb, clip_emb)
        };

        // Synchronize to ensure freed T5/CLIP VRAM is reclaimed before
        // loading the transformer (critical for FP8 models that expand to F16).
        device.synchronize()?;

        // --- Phase 3: Load transformer, denoise ---
        let xformer_size = std::fs::metadata(&transformer_path)
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
            let vb = quantized_var_builder::VarBuilder::from_gguf(&transformer_path, &device)?;
            FluxTransformer::Quantized(flux::quantized_model::Flux::new(&flux_cfg, vb)?)
        } else {
            let flux_vb = flux_transformer_var_builder(flux_safetensors_var_builder(
                &transformer_path,
                gpu_dtype,
                &device,
                false,
            )?);
            FluxTransformer::BF16(flux::model::Flux::new(&flux_cfg, flux_vb)?)
        };
        self.progress
            .stage_done(xformer_label, xformer_stage.elapsed());

        // Generate noise and build state
        let noise_dtype = if is_quantized { DType::F32 } else { gpu_dtype };
        let latent_h = height / 16 * 2;
        let latent_w = width / 16 * 2;
        // Pre-compute timestep schedule (needed before mixing for img2img).
        // For non-schnell models the schedule depends on image_seq_len which
        // we can derive from latent dimensions without the actual tensor.
        let image_seq_len = (latent_h / 2) * (latent_w / 2);
        let mut timesteps = if is_schnell {
            flux::sampling::get_schedule(req.steps as usize, None)
        } else {
            flux::sampling::get_schedule(req.steps as usize, Some((image_seq_len, 0.5, 1.15)))
        };

        // For img2img, build a schedule starting at exactly `strength`.
        // Insert strength as the first timestep, then keep all original schedule
        // points below it. This ensures noise level matches the denoising start
        // and the user gets the exact strength they requested.
        if req.source_image.is_some() {
            let strength = req.strength;
            // Keep only schedule points strictly below strength
            let tail: Vec<f64> = timesteps.into_iter().filter(|&t| t < strength).collect();
            timesteps = std::iter::once(strength).chain(tail).collect();
            tracing::info!(
                strength,
                schedule = ?timesteps,
                remaining_steps = timesteps.len().saturating_sub(1),
                "img2img: built schedule from strength"
            );
        }

        // For img2img we need the VAE before denoising (to encode the source image).
        // For txt2img we defer VAE loading until after denoising to maximize VRAM
        // available for the transformer — critical for FP8 models expanded to F16.
        let vae_cfg = if is_schnell {
            flux::autoencoder::Config::schnell()
        } else {
            flux::autoencoder::Config::dev()
        };

        let (img, inpaint_ctx, early_vae) = if let Some(ref source_bytes) = req.source_image {
            let start_t = req.strength;

            // Load VAE early for source image encoding
            self.progress.stage_start("Loading VAE (GPU)");
            let vae_stage = Instant::now();
            let vae_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&self.paths.vae),
                    gpu_dtype,
                    &device,
                )?
            };
            let vae = flux::autoencoder::AutoEncoder::new(&vae_cfg, vae_vb)?;
            self.progress
                .stage_done("Loading VAE (GPU)", vae_stage.elapsed());

            self.progress.stage_start("Encoding source image (VAE)");
            let encode_start = Instant::now();
            let source_tensor = crate::img_utils::decode_source_image(
                source_bytes,
                req.width,
                req.height,
                crate::img_utils::NormalizeRange::MinusOneToOne,
                &device,
                gpu_dtype,
            )?;
            // FLUX VAE expects pixels in [-1, 1]; encode applies shift/scale internally
            let encoded = vae.encode(&source_tensor)?;
            self.progress
                .stage_done("Encoding source image (VAE)", encode_start.elapsed());

            // Flow-matching img2img: interpolate between encoded latents and noise
            // at the exact noise level matching the first timestep in the schedule
            let noise = crate::engine::seeded_randn(
                seed,
                &[1, 16, latent_h, latent_w],
                &device,
                noise_dtype,
            )?;
            let encoded = encoded.to_dtype(noise_dtype)?;

            // Build inpaint context if mask provided
            let inpaint_ctx = if let Some(ref mask_bytes) = req.mask_image {
                let mask = crate::img_utils::decode_mask_image(
                    mask_bytes,
                    latent_h,
                    latent_w,
                    &device,
                    noise_dtype,
                )?;
                Some(crate::img_utils::InpaintContext {
                    original_latents: encoded.clone(),
                    mask,
                    noise: noise.clone(),
                })
            } else {
                None
            };

            // latent = (1 - t) * encoded + t * noise
            // t matches the first schedule timestep, so denoising starts at the correct level
            let img = ((&encoded * (1.0 - start_t))? + (&noise * start_t)?)?;
            (img, inpaint_ctx, Some(vae))
        } else {
            let img = crate::engine::seeded_randn(
                seed,
                &[1, 16, latent_h, latent_w],
                &device,
                noise_dtype,
            )?;
            (img, None, None)
        };

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

        let denoise_label = format!("Denoising ({} steps)", timesteps.len().saturating_sub(1));
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
            inpaint_ctx.as_ref(),
        )?;

        let img = flux::sampling::unpack(&img, height, width)?;
        self.progress
            .stage_done(&denoise_label, denoise_start.elapsed());

        // Drop transformer + state to free memory for VAE decode
        drop(inpaint_ctx);
        drop(flux_model);
        self.progress.info("Freed FLUX transformer");
        drop(state);
        drop(t5_emb_state);
        drop(clip_emb_state);
        drop(img_state);
        // Synchronize to ensure CUDA frees dropped memory before VAE allocates
        device.synchronize()?;
        tracing::info!("Transformer dropped (sequential mode), decoding VAE...");

        // --- Phase 4: VAE decode ---
        // Use VAE from img2img path if already loaded, otherwise load now
        // (deferred loading saves ~300MB VRAM during denoising for FP8 models).
        let vae = if let Some(vae) = early_vae {
            vae
        } else {
            self.progress.stage_start("Loading VAE (GPU)");
            let vae_stage = Instant::now();
            let vae_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&self.paths.vae),
                    gpu_dtype,
                    &device,
                )?
            };
            let vae = flux::autoencoder::AutoEncoder::new(&vae_cfg, vae_vb)?;
            self.progress
                .stage_done("Loading VAE (GPU)", vae_stage.elapsed());
            vae
        };
        self.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();
        let img = vae.decode(&img.to_dtype(gpu_dtype)?)?;

        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?;

        self.progress
            .stage_done("VAE decode", vae_decode_start.elapsed());
        // VAE dropped here

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

impl InferenceEngine for FluxEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if req.scheduler.is_some() {
            tracing::warn!("scheduler selection not supported for FLUX (flow-matching), ignoring");
        }

        // Sequential mode: load-use-drop each component
        if self.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components
        // Borrow progress reporter separately from loaded state.
        let progress = &self.progress;
        let prompt_cache = &self.prompt_cache;

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
        let transformer_path = self
            .loaded
            .as_ref()
            .map(|l| l.transformer_path.clone())
            .unwrap_or_else(|| self.paths.transformer.clone());

        let mut loaded = OptionRestoreGuard::take(&mut self.loaded)
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;
        let loaded_dtype = loaded.dtype;
        let loaded_device = loaded.device.clone();

        tracing::info!(
            prompt = %req.prompt,
            seed,
            width,
            height,
            steps = req.steps,
            "starting generation"
        );

        (|| -> Result<GenerateResponse> {
            if loaded.flux_model.is_none() {
                let xformer_label = if loaded.is_quantized {
                    "Reloading FLUX transformer (GPU, quantized)"
                } else if loaded.dtype == DType::F16 {
                    "Reloading FLUX transformer (GPU, FP16)"
                } else {
                    "Reloading FLUX transformer (GPU, BF16)"
                };
                progress.stage_start(xformer_label);
                let reload_start = Instant::now();
                let flux_cfg = if loaded.is_schnell {
                    flux::model::Config::schnell()
                } else {
                    flux::model::Config::dev()
                };
                loaded.flux_model = Some(if loaded.is_quantized {
                    let vb = quantized_var_builder::VarBuilder::from_gguf(
                        &transformer_path,
                        &loaded.device,
                    )?;
                    FluxTransformer::Quantized(flux::quantized_model::Flux::new(&flux_cfg, vb)?)
                } else {
                    let flux_vb = flux_transformer_var_builder(flux_safetensors_var_builder(
                        &transformer_path,
                        loaded.dtype,
                        &loaded.device,
                        false, // FP8 models are cached as GGUF, never reloaded as safetensors
                    )?);
                    FluxTransformer::BF16(flux::model::Flux::new(&flux_cfg, flux_vb)?)
                });
                progress.stage_done(xformer_label, reload_start.elapsed());
            }

            if let Some((t5_emb, clip_emb)) = Self::restore_prompt_cache(
                progress,
                prompt_cache,
                &req.prompt,
                &loaded_device,
                loaded_dtype,
            )? {
                return Self::generate_with_embeddings(
                    progress,
                    req,
                    &mut loaded,
                    t5_emb,
                    clip_emb,
                    seed,
                    width,
                    height,
                    start,
                );
            }

            if loaded.t5.model.is_none() {
                progress.stage_start("Reloading T5 encoder (GPU)");
                let reload_start = Instant::now();
                loaded.t5.reload(&t5_encoder_path, loaded_dtype)?;
                progress.stage_done("Reloading T5 encoder (GPU)", reload_start.elapsed());
            }
            if loaded.clip.model.is_none() {
                progress.stage_start("Reloading CLIP encoder (GPU)");
                let reload_start = Instant::now();
                loaded.clip.reload(&clip_encoder_path, loaded_dtype)?;
                progress.stage_done("Reloading CLIP encoder (GPU)", reload_start.elapsed());
            }

            progress.stage_start("Encoding prompt (T5)");
            let encode_t5 = Instant::now();
            let t5_emb = loaded
                .t5
                .encode(&req.prompt, &loaded_device, loaded_dtype)?;
            progress.stage_done("Encoding prompt (T5)", encode_t5.elapsed());
            tracing::info!("T5 encoding complete");

            progress.stage_start("Encoding prompt (CLIP)");
            let encode_clip = Instant::now();
            let clip_emb = loaded
                .clip
                .encode(&req.prompt, &loaded_device, loaded_dtype)?;
            progress.stage_done("Encoding prompt (CLIP)", encode_clip.elapsed());
            tracing::info!("CLIP encoding complete");
            Self::store_prompt_cache(prompt_cache, &req.prompt, &t5_emb, &clip_emb)?;

            if loaded.t5.on_gpu {
                loaded.t5.drop_weights();
                tracing::info!("T5 encoder dropped from GPU to free VRAM for denoising");
            }
            if loaded.clip.on_gpu {
                loaded.clip.drop_weights();
                tracing::info!("CLIP encoder dropped from GPU to free VRAM for denoising");
            }

            Self::generate_with_embeddings(
                progress,
                req,
                &mut loaded,
                t5_emb,
                clip_emb,
                seed,
                width,
                height,
                start,
            )
        })()
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
        clear_cache(&self.prompt_cache);
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }

    fn clear_on_progress(&mut self) {
        self.progress.clear_callback();
    }
}

impl FluxEngine {
    #[allow(clippy::too_many_arguments)]
    fn generate_with_embeddings(
        progress: &ProgressReporter,
        req: &GenerateRequest,
        loaded: &mut LoadedFlux,
        t5_emb: candle_core::Tensor,
        clip_emb: candle_core::Tensor,
        seed: u64,
        width: usize,
        height: usize,
        start: Instant,
    ) -> Result<GenerateResponse> {
        // 3. Generate initial noise (F32 for quantized, gpu_dtype for BF16)
        let noise_dtype = if loaded.is_quantized {
            DType::F32
        } else {
            loaded.dtype
        };
        let latent_h = height / 16 * 2;
        let latent_w = width / 16 * 2;

        // Pre-compute timestep schedule (needed before mixing for img2img).
        let image_seq_len = (latent_h / 2) * (latent_w / 2);
        let mut timesteps = if loaded.is_schnell {
            flux::sampling::get_schedule(req.steps as usize, None)
        } else {
            flux::sampling::get_schedule(req.steps as usize, Some((image_seq_len, 0.5, 1.15)))
        };

        // For img2img, build a schedule starting at exactly `strength`.
        if req.source_image.is_some() {
            let strength = req.strength;
            let tail: Vec<f64> = timesteps.into_iter().filter(|&t| t < strength).collect();
            timesteps = std::iter::once(strength).chain(tail).collect();
            tracing::info!(
                strength,
                schedule = ?timesteps,
                remaining_steps = timesteps.len().saturating_sub(1),
                "img2img: built schedule from strength"
            );
        }

        let (img, inpaint_ctx) = if let Some(ref source_bytes) = req.source_image {
            let start_t = req.strength;

            progress.stage_start("Encoding source image (VAE)");
            let encode_start = Instant::now();
            let source_tensor = crate::img_utils::decode_source_image(
                source_bytes,
                req.width,
                req.height,
                crate::img_utils::NormalizeRange::MinusOneToOne,
                &loaded.device,
                loaded.dtype,
            )?;
            let encoded = loaded.vae.encode(&source_tensor)?;
            progress.stage_done("Encoding source image (VAE)", encode_start.elapsed());

            let noise = crate::engine::seeded_randn(
                seed,
                &[1, 16, latent_h, latent_w],
                &loaded.device,
                noise_dtype,
            )?;
            let encoded = encoded.to_dtype(noise_dtype)?;

            let inpaint_ctx = if let Some(ref mask_bytes) = req.mask_image {
                let mask = crate::img_utils::decode_mask_image(
                    mask_bytes,
                    latent_h,
                    latent_w,
                    &loaded.device,
                    noise_dtype,
                )?;
                Some(crate::img_utils::InpaintContext {
                    original_latents: encoded.clone(),
                    mask,
                    noise: noise.clone(),
                })
            } else {
                None
            };

            // latent = (1 - t) * encoded + t * noise
            let img = ((&encoded * (1.0 - start_t))? + (&noise * start_t)?)?;
            (img, inpaint_ctx)
        } else {
            let img = crate::engine::seeded_randn(
                seed,
                &[1, 16, latent_h, latent_w],
                &loaded.device,
                noise_dtype,
            )?;
            (img, None)
        };

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

        // Build sampling state
        let state = flux::sampling::State::new(&t5_emb_state, &clip_emb_state, &img_state)?;

        let denoise_label = format!("Denoising ({} steps)", timesteps.len().saturating_sub(1));
        progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();
        tracing::info!(
            steps = timesteps.len().saturating_sub(1),
            quantized = loaded.is_quantized,
            "running denoising loop..."
        );

        // Denoise — guidance from request (0.0 for schnell, 3.5+ for dev/finetuned)
        let img = loaded
            .flux_model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("transformer not loaded"))?
            .denoise(
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &timesteps,
                req.guidance,
                progress,
                inpaint_ctx.as_ref(),
            )?;

        // 7. Unpack latent to spatial
        let img = flux::sampling::unpack(&img, height, width)?;
        progress.stage_done(&denoise_label, denoise_start.elapsed());
        tracing::info!("denoising complete, decoding VAE...");

        // Free denoising intermediates and transformer before VAE decode.
        // On discrete GPUs (CUDA), the Q8 transformer alone is ~13GB — VAE decode
        // needs that VRAM for conv2d intermediates. Transformer is reloaded next generate.
        drop(state);
        drop(t5_emb_state);
        drop(clip_emb_state);
        drop(img_state);
        loaded.flux_model = None;
        // Force CUDA to complete pending operations and release freed memory.
        // Without this, cuMemFree is asynchronous and the freed VRAM from the
        // transformer (~13GB) may not be available when VAE decode allocates
        // its conv2d intermediates, causing OOM on subsequent generations.
        loaded.device.synchronize()?;
        tracing::info!("Transformer dropped to free VRAM for VAE decode");

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
}

#[cfg(test)]
mod tests {
    use super::{flux_runtime_dtype, flux_transformer_var_builder};
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    #[test]
    fn flux_var_builder_uses_root_tensors_when_present() -> Result<()> {
        let tensors = HashMap::from([(
            "img_in.weight".to_string(),
            Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
        )]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let resolved = flux_transformer_var_builder(vb);

        assert!(resolved.contains_tensor("img_in.weight"));
        assert_eq!(resolved.prefix(), "");
        Ok(())
    }

    #[test]
    fn flux_var_builder_uses_model_diffusion_model_prefix_when_present() -> Result<()> {
        let tensors = HashMap::from([(
            "model.diffusion_model.img_in.weight".to_string(),
            Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
        )]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let resolved = flux_transformer_var_builder(vb);

        assert!(resolved.contains_tensor("img_in.weight"));
        assert_eq!(resolved.prefix(), "model.diffusion_model");
        Ok(())
    }

    #[test]
    fn flux_runtime_dtype_prefers_f16_for_cuda_fp8_safetensors() {
        assert_eq!(flux_runtime_dtype(true, false, true), DType::F16);
        assert_eq!(flux_runtime_dtype(true, false, false), DType::BF16);
        assert_eq!(flux_runtime_dtype(false, false, true), DType::F32);
    }
}
