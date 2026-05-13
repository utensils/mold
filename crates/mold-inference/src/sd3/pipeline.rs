use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp};
use candle_transformers::models::mmdit::model::{Config as MMDiTConfig, MMDiT};
use candle_transformers::quantized_var_builder;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, LoraWeight, ModelPaths};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::cache::{
    cfg_prompt_cache_key, clear_cache, get_or_insert_cached_tensor_pair,
    restore_cached_tensor_pair, CachedTensorPair, CfgPromptCacheKey, LruCache,
    DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{
    check_memory_budget, fmt_gb, free_vram_bytes, memory_status_string, preflight_memory_check,
    usable_free_vram_bytes,
};
use crate::encoders;
use crate::engine::{
    rand_seed, resolve_cfg_plus, InferenceEngine, LoadStrategy, OptionRestoreGuard,
};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image};
use crate::img_utils;
use crate::progress::{ProgressCallback, ProgressReporter};

use super::lora as sd3_lora;
use super::quantized_mmdit::QuantizedMMDiT;
use super::sampling::{self, SkipLayerGuidanceConfig};
use super::transformer::SD3Transformer;
use super::vae::{build_sd3_vae_autoencoder, sd3_vae_vb_rename};

/// Smallest LoRA scale the engine treats as non-zero. Sliders pinned to
/// 0.0 are dropped from the effective stack — forcing a transformer
/// rebuild for a no-op patch is pure overhead. The threshold matches
/// the precision of an f64 scrubbed by a UI slider.
const ZERO_SCALE_EPS: f64 = 1e-8;

/// Collapse a `GenerateRequest`'s `lora` (legacy single) and `loras`
/// (plural) fields into an ordered, zero-pruned working list. Plural
/// wins when both are set — the singular form is for backward compat.
pub(crate) fn effective_loras(req: &GenerateRequest) -> Vec<LoraWeight> {
    let raw: Vec<LoraWeight> = if let Some(plural) = &req.loras {
        if !plural.is_empty() {
            plural.clone()
        } else {
            req.lora.iter().cloned().collect()
        }
    } else {
        req.lora.iter().cloned().collect()
    };
    raw.into_iter()
        .filter(|w| {
            let keep = w.scale.abs() > ZERO_SCALE_EPS;
            if !keep {
                tracing::debug!(
                    path = w.path.as_str(),
                    scale = w.scale,
                    "dropping zero-scale LoRA from SD3 effective stack"
                );
            }
            keep
        })
        .collect()
}

/// Build a LoRA-aware SD3 `VarBuilder` from BF16 safetensors. Caller
/// is responsible for short-circuiting to the non-LoRA mmap path when
/// `loras` is empty — this function errors on an empty stack so a stray
/// caller doesn't silently get the wrong builder shape.
fn sd3_lora_var_builder<'a>(
    transformer_path: &Path,
    loras: &[LoraWeight],
    dtype: DType,
    device: &Device,
    progress: &ProgressReporter,
    delta_cache: Option<Arc<Mutex<sd3_lora::LoraDeltaCache>>>,
) -> Result<candle_nn::VarBuilder<'a>> {
    let adapters: Vec<Arc<sd3_lora::LoraAdapter>> = loras
        .iter()
        .map(|w| {
            progress.info("Loading SD3 LoRA adapter");
            let adapter = sd3_lora::get_or_load_adapter(Path::new(&w.path))?;
            progress.info(&format!(
                "SD3 LoRA: {} layers, rank {}, scale {:.2}",
                adapter.layers.len(),
                adapter.rank,
                w.scale,
            ));
            anyhow::Ok(adapter)
        })
        .collect::<Result<_>>()?;

    let specs: Vec<sd3_lora::LoraSpec<'_>> = adapters
        .iter()
        .zip(loras.iter())
        .map(|(adapter, w)| sd3_lora::LoraSpec {
            adapter: adapter.as_ref(),
            scale: w.scale,
            path_hash: sd3_lora::lora_path_hash(&w.path),
        })
        .collect();

    sd3_lora::lora_var_builder(
        transformer_path,
        &specs,
        dtype,
        device,
        progress,
        delta_cache,
    )
}

/// GGUF counterpart to `sd3_lora_var_builder` — selectively dequantizes
/// patched tensors, merges, and re-quantizes back to the original GGML
/// dtype on the target device.
fn sd3_gguf_lora_var_builder(
    transformer_path: &Path,
    loras: &[LoraWeight],
    device: &Device,
    progress: &ProgressReporter,
    delta_cache: Option<Arc<Mutex<sd3_lora::LoraDeltaCache>>>,
) -> Result<quantized_var_builder::VarBuilder> {
    let adapters: Vec<Arc<sd3_lora::LoraAdapter>> = loras
        .iter()
        .map(|w| {
            progress.info("Loading SD3 LoRA adapter");
            let adapter = sd3_lora::get_or_load_adapter(Path::new(&w.path))?;
            progress.info(&format!(
                "SD3 LoRA: {} layers, rank {}, scale {:.2}",
                adapter.layers.len(),
                adapter.rank,
                w.scale,
            ));
            anyhow::Ok(adapter)
        })
        .collect::<Result<_>>()?;

    let specs: Vec<sd3_lora::LoraSpec<'_>> = adapters
        .iter()
        .zip(loras.iter())
        .map(|(adapter, w)| sd3_lora::LoraSpec {
            adapter: adapter.as_ref(),
            scale: w.scale,
            path_hash: sd3_lora::lora_path_hash(&w.path),
        })
        .collect();

    sd3_lora::gguf_lora_var_builder(transformer_path, &specs, device, progress, delta_cache)
}

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
    prompt_cache: Mutex<LruCache<CfgPromptCacheKey, CachedTensorPair>>,
    pending_placement: Option<mold_core::types::DevicePlacement>,
    /// CPU-resident cache of pre-computed LoRA deltas, shared across
    /// transformer rebuilds. Saves the `B @ A * scale` matmul when the
    /// same LoRA stack reappears on a later generate.
    lora_delta_cache: Arc<Mutex<sd3_lora::LoraDeltaCache>>,
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
            lora_delta_cache: Arc::new(Mutex::new(sd3_lora::LoraDeltaCache::new())),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_conditioning(
        progress: &ProgressReporter,
        prompt_cache: &Mutex<LruCache<CfgPromptCacheKey, CachedTensorPair>>,
        triple_encoder: &mut encoders::sd3_clip::SD3TripleEncoder,
        prompt: &str,
        negative_prompt: &str,
        guidance: f64,
        device: &Device,
        dtype: DType,
        is_quantized: bool,
    ) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
        // SD3 always concatenates `(cond, uncond)` for CFG, so the cache key
        // must include the negative prompt and guidance — keying only on the
        // positive prompt returned a stale `(cond, uncond_old)` pair when the
        // user changed only the negative.
        let cache_key = cfg_prompt_cache_key(prompt, negative_prompt, guidance);
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
        // Log the raw driver reading; pass the reserve-adjusted budget to
        // variant resolution below.
        let free_raw = free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        if free_raw > 0 {
            self.base.progress.info(&format!(
                "Free VRAM after transformer: {}",
                fmt_gb(free_raw)
            ));
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
        let auto_encoder_device = if t5_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let encoder_device_owned =
            crate::device::resolve_device(Some(tier1), || Ok(auto_encoder_device.clone()))?;
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
        let cache_key = cfg_prompt_cache_key(&req.prompt, neg, req.guidance);
        let (context, y) = if let Some((context, y)) =
            restore_cached_tensor_pair(&self.prompt_cache, &cache_key, &device, gpu_dtype)?
        {
            self.base.progress.cache_hit("prompt conditioning");
            (context, y)
        } else {
            // Reserve-adjusted reading drives the T5 variant selection.
            let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
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
            let auto_encoder_device = if t5_on_gpu {
                device.clone()
            } else {
                Device::Cpu
            };
            let encoder_device_owned =
                crate::device::resolve_device(Some(tier1), || Ok(auto_encoder_device.clone()))?;
            let encoder_device = &encoder_device_owned;
            let t5_on_gpu = !encoder_device.is_cpu();
            let t5_device_label = if t5_on_gpu { "GPU" } else { "CPU" };
            let encoder_dtype = if t5_on_gpu { gpu_dtype } else { DType::F32 };

            let t5_size = std::fs::metadata(&resolved_t5_path)
                .map(|m| m.len())
                .unwrap_or(0);
            let te_activation_budget = crate::device::activation_bytes(
                req.width,
                req.height,
                1,
                crate::device::dtype_bytes(encoder_dtype),
                crate::device::ActivationFamily::SmallTransformer,
            );
            preflight_memory_check("SD3 triple encoder", t5_size, te_activation_budget)?;
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
                req.guidance,
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
            let vae_dtype = crate::device::resolve_vae_dtype(gpu_dtype);
            let vae_vb = crate::weight_loader::load_safetensors_with_progress(
                std::slice::from_ref(&self.base.paths.vae),
                vae_dtype,
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
                vae_dtype,
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
        // SD3 runs CFG by default → batch=2 if guidance > 1, else batch=1.
        let xformer_batch = if req.guidance > 1.0 { 2 } else { 1 };
        let xformer_activation_budget = crate::device::activation_bytes(
            req.width,
            req.height,
            xformer_batch,
            crate::device::dtype_bytes(gpu_dtype),
            crate::device::ActivationFamily::Sd3Mmdit,
        );
        preflight_memory_check(
            "SD3 MMDiT transformer",
            xformer_size,
            xformer_activation_budget,
        )?;
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        let active_loras = effective_loras(req);
        let lora_delta_cache = self.lora_delta_cache.clone();
        let xformer_label = match (is_quantized, active_loras.is_empty()) {
            (true, true) => "Loading SD3 MMDiT transformer (GPU, quantized)",
            (true, false) => "Loading SD3 MMDiT transformer (GPU, quantized, with LoRA)",
            (false, true) => "Loading SD3 MMDiT transformer (GPU, FP16)",
            (false, false) => "Loading SD3 MMDiT transformer (GPU, FP16, with LoRA)",
        };
        self.base.progress.stage_start(xformer_label);
        let xformer_stage = Instant::now();

        let transformer = if is_quantized {
            let vb = if active_loras.is_empty() {
                quantized_var_builder::VarBuilder::from_gguf(&self.base.paths.transformer, &device)?
            } else {
                sd3_gguf_lora_var_builder(
                    &self.base.paths.transformer,
                    &active_loras,
                    &device,
                    &self.base.progress,
                    Some(lora_delta_cache.clone()),
                )?
            };
            SD3Transformer::Quantized(QuantizedMMDiT::new(&mmdit_config, vb)?)
        } else if active_loras.is_empty() {
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
        } else {
            // LoRA path: the `LoraBackend` strips the on-disk prefix
            // automatically, so the builder is positioned at the prefix
            // root — feed it straight to `MMDiT::new`.
            let vb = sd3_lora_var_builder(
                &self.base.paths.transformer,
                &active_loras,
                gpu_dtype,
                &device,
                &self.base.progress,
                Some(lora_delta_cache.clone()),
            )?;
            SD3Transformer::BF16(MMDiT::new(&mmdit_config, false, vb)?)
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
            resolve_cfg_plus(req),
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
        let vae_dtype = crate::device::resolve_vae_dtype(gpu_dtype);
        let vae_vb = crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(&self.base.paths.vae),
            vae_dtype,
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
        // Cast to VAE dtype (quantized path outputs F32, VAE is F16/BF16/F32
        // depending on MOLD_VAE_DTYPE).
        let x = ((x / 1.5305)? + 0.0609)?.to_dtype(vae_dtype)?;
        let device_for_sync = device.clone();
        let img = crate::vae_tiling::decode_with_oom_fallback(
            &x,
            |t| autoencoder.decode(t).map_err(Into::into),
            || {
                if let Err(e) = device_for_sync.synchronize() {
                    tracing::warn!(
                        "SD3 (sequential) device.synchronize() after VAE OOM failed: {e}"
                    );
                }
            },
        )?;

        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?;

        self.base
            .progress
            .stage_done("VAE decode", vae_decode_start.elapsed());

        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &img,
            req.resolved_output_format(),
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
                format: req.resolved_output_format(),
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
        let active_loras = effective_loras(req);
        let lora_delta_cache = self.lora_delta_cache.clone();

        let mut loaded = OptionRestoreGuard::take(&mut self.base.loaded)
            .ok_or_else(|| anyhow::anyhow!("model not loaded -- call load() first"))?;
        let loaded_dtype = loaded.dtype;
        let loaded_device = loaded.device.clone();
        let is_quantized = loaded._is_quantized;

        // LoRA: force a transformer rebuild via the LoRA path. The
        // eager-loaded transformer is the unpatched base; even if the
        // same stack was applied on a previous generate, the in-memory
        // tensors no longer carry that delta after the post-decode
        // drop. The `LoraDeltaCache` keeps the matmul work cheap on
        // every rebuild.
        if !active_loras.is_empty() && loaded.transformer.is_some() {
            loaded.transformer = None;
            loaded_device.synchronize()?;
            progress.info("SD3 LoRA: dropping base transformer for LoRA merge");
        }

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
                let label = if loaded.triple_encoder.is_parked() {
                    "Unparking SD3 triple encoder (CPU→GPU)"
                } else {
                    "Reloading SD3 triple encoder"
                };
                progress.stage_start(label);
                let reload_start = Instant::now();
                if loaded.triple_encoder.is_parked() {
                    loaded
                        .triple_encoder
                        .unpark_to_gpu(loaded_dtype, progress)?;
                } else {
                    loaded.triple_encoder.reload(loaded_dtype, progress)?;
                }
                progress.stage_done(label, reload_start.elapsed());
            }

            let neg = req.negative_prompt.as_deref().unwrap_or("");
            let (context, y) = Self::encode_conditioning(
                progress,
                prompt_cache,
                &mut loaded.triple_encoder,
                &req.prompt,
                neg,
                req.guidance,
                &loaded_device,
                loaded_dtype,
                is_quantized,
            )?;

            if loaded.triple_encoder.on_gpu {
                // Park mode keeps the FP16 encoders alive on host RAM (~9 GB
                // T5 + ~1.6 GB CLIPs). Disabled on Metal (unified memory).
                let park_mode = crate::device::keep_te_in_ram() && !loaded_device.is_metal();
                if park_mode {
                    loaded.triple_encoder.park_to_cpu()?;
                    tracing::info!("SD3 triple encoder parked to CPU host RAM");
                } else {
                    loaded.triple_encoder.drop_weights();
                    tracing::info!(
                        "SD3 triple encoder dropped from GPU to free VRAM for denoising"
                    );
                }
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
                    let vae_dtype = crate::device::resolve_vae_dtype(loaded_dtype);
                    let vae_vb = crate::weight_loader::load_safetensors_with_progress(
                        std::slice::from_ref(&loaded.vae_vb_path),
                        vae_dtype,
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
                        vae_dtype,
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
                let reload_label = if active_loras.is_empty() {
                    "Reloading SD3 transformer"
                } else {
                    "Reloading SD3 transformer (with LoRA)"
                };
                progress.stage_start(reload_label);
                let reload_start = Instant::now();
                let transformer = if is_quantized {
                    let vb = if active_loras.is_empty() {
                        quantized_var_builder::VarBuilder::from_gguf(
                            &transformer_path,
                            &loaded_device,
                        )?
                    } else {
                        sd3_gguf_lora_var_builder(
                            &transformer_path,
                            &active_loras,
                            &loaded_device,
                            progress,
                            Some(lora_delta_cache.clone()),
                        )?
                    };
                    SD3Transformer::Quantized(QuantizedMMDiT::new(&mmdit_config, vb)?)
                } else if active_loras.is_empty() {
                    let vb = crate::weight_loader::load_safetensors_with_progress(
                        std::slice::from_ref(&transformer_path),
                        loaded_dtype,
                        &loaded_device,
                        "SD3 transformer",
                        progress,
                    )?;
                    let vb = vb.pp("model.diffusion_model");
                    SD3Transformer::BF16(MMDiT::new(&mmdit_config, false, vb)?)
                } else {
                    // LoRA path: the `LoraBackend` already absorbs the
                    // on-disk `model.diffusion_model.` prefix, so the
                    // builder is positioned at the prefix root.
                    let vb = sd3_lora_var_builder(
                        &transformer_path,
                        &active_loras,
                        loaded_dtype,
                        &loaded_device,
                        progress,
                        Some(lora_delta_cache.clone()),
                    )?;
                    SD3Transformer::BF16(MMDiT::new(&mmdit_config, false, vb)?)
                };
                loaded.transformer = Some(transformer);
                progress.stage_done(reload_label, reload_start.elapsed());
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
                resolve_cfg_plus(req),
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

            let vae_dtype = crate::device::resolve_vae_dtype(loaded.dtype);
            let vae_vb = crate::weight_loader::load_safetensors_with_progress(
                std::slice::from_ref(&loaded.vae_vb_path),
                vae_dtype,
                &loaded.device,
                "VAE",
                progress,
            )?;
            let vae_vb = vae_vb.rename_f(sd3_vae_vb_rename).pp("first_stage_model");
            let autoencoder = build_sd3_vae_autoencoder(vae_vb)?;

            let x = ((x / 1.5305)? + 0.0609)?.to_dtype(vae_dtype)?;
            let device_for_sync = loaded.device.clone();
            let img = crate::vae_tiling::decode_with_oom_fallback(
                &x,
                |t| autoencoder.decode(t).map_err(Into::into),
                || {
                    if let Err(e) = device_for_sync.synchronize() {
                        tracing::warn!(
                            "SD3 (parallel) device.synchronize() after VAE OOM failed: {e}"
                        );
                    }
                },
            )?;

            let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
            let img = img.i(0)?;

            progress.stage_done("VAE decode", vae_decode_start.elapsed());

            let output_metadata = build_output_metadata(req, seed, None);
            let image_bytes = encode_image(
                &img,
                req.resolved_output_format(),
                req.width,
                req.height,
                output_metadata.as_ref(),
            )?;

            let generation_time_ms = start.elapsed().as_millis() as u64;
            tracing::info!(generation_time_ms, seed, "SD3 generation complete");

            Ok(GenerateResponse {
                images: vec![ImageData {
                    data: image_bytes,
                    format: req.resolved_output_format(),
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

    /// Regression test for the SD3 prompt-cache key bug: when the cache is
    /// keyed only on the positive prompt, a follow-up request that changes
    /// just the negative prompt returns the previous `(cond, uncond_old)`
    /// pair — silently producing wrong output.
    ///
    /// Drives the cache through `restore_cached_tensor_pair` directly because
    /// the surrounding `encode_conditioning` requires loaded T5/CLIP weights;
    /// the contract under test is the keying, not the encoder forward pass.
    #[test]
    fn sd3_prompt_cache_distinguishes_negative_prompt_changes() {
        use crate::cache::{cfg_prompt_cache_key, store_cached_tensor_pair};

        let cache: Mutex<LruCache<CfgPromptCacheKey, CachedTensorPair>> =
            Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY));
        let device = Device::Cpu;
        let dtype = DType::F32;
        let context = candle_core::Tensor::zeros((1, 4), dtype, &device).unwrap();
        let y = candle_core::Tensor::zeros((1, 4), dtype, &device).unwrap();

        let key_a = cfg_prompt_cache_key("a cat", "blurry", 7.0);
        store_cached_tensor_pair(&cache, key_a.clone(), &context, &y).unwrap();

        // Same positive + same guidance, different negative → MUST miss.
        let key_b = cfg_prompt_cache_key("a cat", "low quality", 7.0);
        let restored = restore_cached_tensor_pair(&cache, &key_b, &device, dtype).unwrap();
        assert!(
            restored.is_none(),
            "different negative prompt must miss the cache (was the silent-wrong-output bug)"
        );

        // Same key as the insert → MUST hit.
        let restored = restore_cached_tensor_pair(&cache, &key_a, &device, dtype).unwrap();
        assert!(
            restored.is_some(),
            "identical (pos, neg, guidance) must still hit"
        );
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

    // -----------------------------------------------------------------------
    // resolve_cfg_plus precedence: explicit request field beats env var,
    // env var beats default-off. MOLD_CFG_PLUS is process-global so these
    // tests serialize via a static mutex.
    // -----------------------------------------------------------------------

    fn cfg_env_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::{Mutex, OnceLock};
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    fn req_with_cfg_plus(cfg_plus: Option<bool>) -> GenerateRequest {
        // Build a minimal SD3-shaped request via JSON to avoid maintaining a
        // by-hand list of every GenerateRequest field across schema changes.
        let mut req: GenerateRequest = serde_json::from_str(
            r#"{
                "prompt":"x",
                "model":"sd3.5-large:fp16",
                "width":1024,
                "height":1024,
                "steps":28,
                "guidance":4.5
            }"#,
        )
        .unwrap();
        req.cfg_plus = cfg_plus;
        req
    }

    #[test]
    fn resolve_cfg_plus_defaults_off() {
        let _guard = cfg_env_lock();
        // SAFETY: serialized via cfg_env_lock to avoid racing parallel tests.
        unsafe { std::env::remove_var("MOLD_CFG_PLUS") };
        assert!(!resolve_cfg_plus(&req_with_cfg_plus(None)));
    }

    #[test]
    fn resolve_cfg_plus_env_enables() {
        let _guard = cfg_env_lock();
        unsafe { std::env::set_var("MOLD_CFG_PLUS", "1") };
        let on = resolve_cfg_plus(&req_with_cfg_plus(None));
        unsafe { std::env::remove_var("MOLD_CFG_PLUS") };
        assert!(on, "MOLD_CFG_PLUS=1 must enable cfg++");
    }

    #[test]
    fn resolve_cfg_plus_request_field_wins_over_env() {
        let _guard = cfg_env_lock();
        // Env says on, request explicitly says off → request wins. Without
        // this precedence a server with a global env default could not be
        // overridden per-request, which is the whole point of having a
        // request field.
        unsafe { std::env::set_var("MOLD_CFG_PLUS", "1") };
        let off = resolve_cfg_plus(&req_with_cfg_plus(Some(false)));
        unsafe { std::env::remove_var("MOLD_CFG_PLUS") };
        assert!(!off, "explicit Some(false) must override env=on");
    }

    #[test]
    fn resolve_cfg_plus_request_true_without_env() {
        let _guard = cfg_env_lock();
        unsafe { std::env::remove_var("MOLD_CFG_PLUS") };
        assert!(resolve_cfg_plus(&req_with_cfg_plus(Some(true))));
    }
}
