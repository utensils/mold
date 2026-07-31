//! Flux.2 Klein inference engine (4B and 9B variants).
//!
//! Follows the same Eager + Sequential loading pattern as FluxEngine and ZImageEngine.
//!
//! Key differences from FLUX.1:
//! - Uses Qwen3 text encoder (not T5 + CLIP)
//!   - Klein-4B: Qwen3-4B (hidden=2560), stacked layers → 7680-dim context
//!   - Klein-9B: Qwen3-8B (hidden=4096), stacked layers → 12288-dim context
//! - VAE has latent_channels=32 (not 16)
//! - Transformer has 128 input channels (not 64)
//! - 4D RoPE (not 3D)
//! - Klein is distilled (no guidance embedding)
//! - No pooled text vector input
//! - Linear timestep schedule (distilled, no time-shifting)

use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, LoraWeight, ModelPaths};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;

use super::sampling::{self, Flux2State};
use super::transformer::{Flux2Config, Flux2TransformerWrapper};
use super::vae::{Flux2AutoEncoder, Flux2VaeConfig};

use crate::cache::{
    clear_cache, get_or_insert_cached_tensor, prompt_text_key, restore_cached_tensor, CachedTensor,
    LruCache, DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{
    check_memory_budget, effective_device_ref, fmt_gb, free_vram_bytes, memory_status_string,
    preflight_memory_check, usable_free_vram_bytes,
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
    /// Effective VAE dtype after `MOLD_VAE_DTYPE` resolution. May differ from
    /// `dtype` when fp32 VAE decode is forced to suppress banding artifacts.
    /// Captured at load time; sequential reloads re-resolve per request.
    vae_dtype: DType,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Flux.2 Klein inference engine (4B and 9B variants) backed by candle.
pub struct Flux2Engine {
    base: EngineBase<LoadedFlux2>,
    /// Qwen3 variant preference: None/"auto" = VRAM-based, "bf16" = force BF16, "q8"/etc = specific.
    qwen3_variant: Option<String>,
    /// Force adaptive block-level transformer offload.
    offload: bool,
    prompt_cache: Mutex<LruCache<String, CachedTensor>>,
    /// Per-request placement override. Set at the start of `generate()`,
    /// cleared on exit. `None` preserves the existing VRAM-aware auto logic.
    pending_placement: Option<mold_core::types::DevicePlacement>,
    /// Per-request LoRA stack (effective: zero-scale entries already filtered).
    /// Set at the start of `generate()`, cleared on exit. Read by
    /// `load_transformer` / `reload_transformer_if_needed` to decide whether
    /// to wrap the transformer's `VarBuilder` with a `Flux2LoraBackend`.
    pending_loras: Vec<LoraWeight>,
    shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
}

/// Resolve the effective LoRA list for a request. Mirrors the FLUX helper of
/// the same shape: `loras` (plural) wins over `lora` (singular) when both are
/// set, and zero-scale entries are filtered out so they don't trigger a
/// transformer rebuild for nothing.
pub(crate) fn effective_flux2_loras(req: &GenerateRequest) -> Vec<LoraWeight> {
    /// Threshold below which a LoRA scale is treated as off (matches FLUX).
    const ZERO_SCALE_EPS: f64 = 1e-8;

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
                    "dropping zero-scale Flux.2 LoRA"
                );
            }
            keep
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
enum Flux2OffloadDecision {
    Disabled,
    Selected,
    Unsupported(&'static str),
}

fn flux2_offload_decision(
    forced_offload: bool,
    is_gguf: bool,
    is_nvfp4: bool,
    has_lora: bool,
) -> Flux2OffloadDecision {
    if !forced_offload {
        return Flux2OffloadDecision::Disabled;
    }
    if is_nvfp4 {
        return Flux2OffloadDecision::Disabled;
    }
    if is_gguf {
        return Flux2OffloadDecision::Unsupported(
            "Flux.2 block-level offload is only planned for BF16/FP transformers; \
             GGUF variants already use quantized transformer paths",
        );
    }
    if has_lora {
        return Flux2OffloadDecision::Unsupported(
            "Flux.2 block-level offload with LoRA is not wired yet; \
             LoRA merge/bypass semantics need a dedicated offload design",
        );
    }
    Flux2OffloadDecision::Selected
}

impl Flux2Engine {
    /// Create a new Flux2Engine. Does not load models until `load()` is called.
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        qwen3_variant: Option<String>,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        offload: bool,
        shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy, gpu_ordinal),
            qwen3_variant,
            offload,
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            pending_placement: None,
            pending_loras: Vec::new(),
            shared_pool,
        }
    }

    /// Construct a Flux.2 engine from a Civitai / ComfyUI single-file
    /// safetensors checkpoint (BFL-native naming, every key prefixed
    /// `model.diffusion_model.`).
    ///
    /// The transformer is the single-file checkpoint itself; the VAE,
    /// Qwen3 text encoder, and tokenizer arrive via companion paths
    /// resolved by the catalog bridge before the engine is constructed.
    /// The header is not peeked here — `load_transformer` re-detects the
    /// format at load time so a per-engine error surfaces in the same
    /// place as every other transformer load failure.
    #[allow(clippy::too_many_arguments)]
    pub fn from_single_file(
        model_name: String,
        transformer_path: PathBuf,
        vae_path: PathBuf,
        text_encoder_files: Vec<PathBuf>,
        text_tokenizer: Option<PathBuf>,
        qwen3_variant: Option<String>,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        offload: bool,
        shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
    ) -> Result<Self> {
        if !transformer_path.exists() {
            bail!(
                "single-file Flux.2 checkpoint not found: {}",
                transformer_path.display()
            );
        }

        let paths = ModelPaths {
            transformer: transformer_path,
            transformer_shards: Vec::new(),
            vae: vae_path,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files,
            text_tokenizer,
            decoder: None,
        };

        Ok(Self {
            base: EngineBase::new(model_name, paths, load_strategy, gpu_ordinal),
            qwen3_variant,
            offload,
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            pending_placement: None,
            pending_loras: Vec::new(),
            shared_pool,
        })
    }

    /// Select the appropriate transformer config. Header-peeks the
    /// checkpoint when it's a single-file `.safetensors` to determine
    /// hidden_size (3072 → Klein-4B, 4096 → Klein-9B). Falls back to
    /// the model-name heuristic for sharded HF diffusers layouts and
    /// when header-peek can't find an `img_in.weight` marker (e.g. some
    /// community FP8 conversions). This is necessary for opaque names
    /// like `cv:2759597` whose mapping to a Klein variant is only
    /// recoverable from the file itself.
    fn resolve_config(&self) -> Flux2Config {
        if let Some(cfg) = self.detect_config_from_checkpoint() {
            return cfg;
        }
        if self.base.model_name.to_lowercase().contains("9b") {
            Flux2Config::klein_9b()
        } else {
            Flux2Config::klein()
        }
    }

    /// Header-peek the transformer file (if it's a single `.safetensors`)
    /// and pick the config matching its `hidden_size`. Returns `None` for
    /// sharded loads or when no `img_in.weight` marker is present.
    fn detect_config_from_checkpoint(&self) -> Option<Flux2Config> {
        if !self.base.paths.transformer_shards.is_empty() {
            return None;
        }
        let path = &self.base.paths.transformer;
        let is_safetensors = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
        if !is_safetensors {
            return None;
        }
        match super::single_file::detect_hidden_size(path) {
            Ok(Some(4096)) => Some(Flux2Config::klein_9b()),
            Ok(Some(3072)) => Some(Flux2Config::klein()),
            // Anything else: unknown variant, defer to name heuristic.
            _ => None,
        }
    }

    /// Whether this is a Klein-9B model (uses Qwen3-8B text encoder).
    /// Mirrors `resolve_config` — peek the checkpoint first, fall back
    /// to the model-name heuristic.
    fn is_9b(&self) -> bool {
        if let Some(cfg) = self.detect_config_from_checkpoint() {
            return cfg.hidden_size == 4096;
        }
        self.base.model_name.to_lowercase().contains("9b")
    }

    /// Return the Qwen3 encoder size enum for this model.
    fn qwen3_size(&self) -> crate::encoders::variant_resolution::Qwen3Size {
        if self.is_9b() {
            crate::encoders::variant_resolution::Qwen3Size::B8
        } else {
            crate::encoders::variant_resolution::Qwen3Size::B4
        }
    }

    /// Return the BF16 config for the Qwen3 encoder used by this model.
    fn qwen3_bf16_config(&self) -> encoders::qwen3_bf16::Qwen3BF16Config {
        if self.is_9b() {
            encoders::qwen3_bf16::Qwen3BF16Config::qwen3_8b()
        } else {
            encoders::qwen3_bf16::Qwen3BF16Config::qwen3_4b()
        }
    }

    fn load_text_tokenizer(&self, tokenizer_path: &Path) -> Result<Arc<Tokenizer>> {
        if let Some(shared_pool) = &self.shared_pool {
            return shared_pool.lock().unwrap().load_tokenizer(tokenizer_path);
        }
        Tokenizer::from_file(tokenizer_path)
            .map(Arc::new)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))
    }

    fn load_vae_cpu_tensors(&self) -> Result<Option<Arc<HashMap<String, Tensor>>>> {
        let Some(shared_pool) = &self.shared_pool else {
            return Ok(None);
        };
        shared_pool
            .lock()
            .unwrap()
            .load_safetensors_cpu_tensors(std::slice::from_ref(&self.base.paths.vae))
    }

    fn load_vae_var_builder<'a>(
        &self,
        dtype: DType,
        device: &Device,
        component: &str,
    ) -> Result<VarBuilder<'a>> {
        if let Some(tensors) = self.load_vae_cpu_tensors()? {
            return Ok(crate::encoders::park::varbuilder_from_parked(
                tensors.as_ref(),
                dtype,
                device,
            ));
        }

        crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(&self.base.paths.vae),
            dtype,
            device,
            component,
            &self.base.progress,
        )
    }

    fn img2img_source_normalize_range() -> crate::img_utils::NormalizeRange {
        crate::img_utils::NormalizeRange::MinusOneToOne
    }

    #[cfg(test)]
    fn sequential_img2img_preencodes_source() -> bool {
        true
    }

    fn uses_sequential_generate_path(&self, req: &GenerateRequest) -> bool {
        self.base.load_strategy == LoadStrategy::Sequential
            || self.offload
            || !self.pending_loras.is_empty()
            || req.source_image.is_some()
    }

    fn load_sequential_vae(
        &self,
        device: &Device,
        gpu_dtype: DType,
    ) -> Result<(Flux2AutoEncoder, DType)> {
        let vae_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.vae.clone()),
            false,
        );
        let vae_device = crate::device::resolve_device(Some(vae_ref), || Ok(device.clone()))?;
        self.base.progress.stage_start("Loading VAE (GPU)");
        let vae_stage = Instant::now();
        let vae_cfg = Flux2VaeConfig::klein();
        // Sequential path resolves MOLD_VAE_DTYPE per request — env changes
        // take effect on the next generate() without an engine reload.
        let vae_dtype = crate::device::resolve_vae_dtype(gpu_dtype);
        let vae_vb = self.load_vae_var_builder(vae_dtype, &vae_device, "VAE")?;
        let vae = Flux2AutoEncoder::new(&vae_cfg, vae_vb)?;
        self.base
            .progress
            .stage_done("Loading VAE (GPU)", vae_stage.elapsed());
        Ok((vae, vae_dtype))
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
    ///
    /// When `self.pending_loras` is non-empty, every BF16 / GGUF branch wraps
    /// the underlying tensor source with a Flux.2 LoRA backend so the
    /// constructed transformer carries `W' = W + scale·B@A` for every
    /// LoRA-targeted layer (additive across multiple adapters). See
    /// `super::lora` for the key-mapping rules.
    fn load_transformer(
        &self,
        cfg: &Flux2Config,
        gpu_dtype: DType,
        device: &Device,
        activation_budget: u64,
    ) -> Result<(Flux2TransformerWrapper, &'static str)> {
        let has_lora = !self.pending_loras.is_empty();
        if self.is_gguf_transformer() {
            if has_lora {
                // Dequant→merge→requant on every LoRA-affected GGUF tensor.
                // Non-LoRA tensors stay quantised, untouched.
                let adapters =
                    super::lora::load_lora_adapters(&self.pending_loras, &self.base.progress)?;
                let specs: Vec<super::lora::Flux2LoraSpec<'_>> = adapters
                    .iter()
                    .zip(self.pending_loras.iter())
                    .map(|(adapter, w)| super::lora::Flux2LoraSpec {
                        adapter: adapter.as_ref(),
                        scale: w.scale,
                        path_hash: super::lora::lora_path_hash(&w.path),
                    })
                    .collect();
                let gguf_vb = super::lora::gguf_lora_var_builder_flux2(
                    &self.base.paths.transformer,
                    &specs,
                    device,
                    &self.base.progress,
                    None,
                )?;
                return Ok((
                    Flux2TransformerWrapper::Quantized(
                        super::quantized_transformer::QuantizedFlux2Transformer::new(
                            cfg, gguf_vb, device,
                        )?,
                    ),
                    "Loading Flux.2 transformer (GPU, GGUF + LoRA)",
                ));
            }
            // Weights stay quantized in VRAM via QMatMul — no dequantization at load
            // time. A Q4 Klein-9B uses ~6GB VRAM instead of ~18GB with full dequant.
            let gguf_vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &self.base.paths.transformer,
                device,
            )?;
            Ok((
                Flux2TransformerWrapper::Quantized(
                    super::quantized_transformer::QuantizedFlux2Transformer::new(
                        cfg, gguf_vb, device,
                    )?,
                ),
                "Loading Flux.2 transformer (GPU, GGUF)",
            ))
        } else if self.is_bfl_native_single_file() {
            let is_nvfp4 = self.is_nvfp4_single_file();
            // Civitai / ComfyUI single-file checkpoints carry BFL-native
            // tensor names (`model.diffusion_model.*`); the diffusers
            // `Flux2Transformer::new` consumer is wrapped over a
            // `SingleFileBackend` that translates those keys on the fly.
            tracing::info!(
                path = %self.base.paths.transformer.display(),
                "loading Flux.2 transformer from BFL-native single-file checkpoint"
            );
            let backend =
                crate::loader::single_file_backend::SingleFileBackend::from_flux2_singlefile(
                    &self.base.paths.transformer,
                    cfg,
                )?;
            let backend: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(backend);
            if self.offload && !has_lora && !is_nvfp4 {
                let flux_vb = candle_nn::VarBuilder::from_backend(backend, gpu_dtype, Device::Cpu);
                return Ok((
                    Flux2TransformerWrapper::Offloaded(
                        super::transformer::OffloadedFlux2Transformer::new(
                            cfg,
                            flux_vb,
                            device,
                            self.base.gpu_ordinal,
                            activation_budget,
                            &self.base.progress,
                        )?,
                    ),
                    "Loading Flux.2 transformer (offload, BF16, single-file remap)",
                ));
            }
            let backend = if has_lora {
                let adapters =
                    super::lora::load_lora_adapters(&self.pending_loras, &self.base.progress)?;
                let specs: Vec<super::lora::Flux2LoraSpec<'_>> = adapters
                    .iter()
                    .zip(self.pending_loras.iter())
                    .map(|(adapter, w)| super::lora::Flux2LoraSpec {
                        adapter: adapter.as_ref(),
                        scale: w.scale,
                        path_hash: super::lora::lora_path_hash(&w.path),
                    })
                    .collect();
                super::lora::wrap_backend_with_lora(
                    backend,
                    &specs,
                    super::lora::Flux2KeySpace::Diffusers,
                    &self.base.progress,
                    None,
                )?
            } else {
                backend
            };
            let flux_vb = candle_nn::VarBuilder::from_backend(backend, gpu_dtype, device.clone());
            let label = if has_lora {
                "Loading Flux.2 transformer (GPU, BF16, single-file remap + LoRA)"
            } else {
                "Loading Flux.2 transformer (GPU, BF16, single-file remap)"
            };
            Ok((
                Flux2TransformerWrapper::BF16(super::transformer::Flux2Transformer::new(
                    cfg, flux_vb,
                )?),
                label,
            ))
        } else {
            let xformer_paths = if !self.base.paths.transformer_shards.is_empty() {
                self.base.paths.transformer_shards.clone()
            } else {
                vec![self.base.paths.transformer.clone()]
            };
            let (flux_vb, offloaded_label) = if has_lora {
                // Build our own mmap-backed SimpleBackend so we can wrap with
                // `Flux2LoraBackend`. The progress reporting drops to a single
                // info line — the legacy progress bar is keyed to candle's
                // internal mmap path.
                use candle_core::safetensors::MmapedSafetensors;
                let path_refs: Vec<&std::path::Path> =
                    xformer_paths.iter().map(|p| p.as_path()).collect();
                let st = unsafe { MmapedSafetensors::multi(&path_refs)? };
                struct MmapBackend {
                    st: MmapedSafetensors,
                }
                impl candle_nn::var_builder::SimpleBackend for MmapBackend {
                    fn get(
                        &self,
                        _s: candle_core::Shape,
                        name: &str,
                        _h: candle_nn::Init,
                        dtype: DType,
                        dev: &Device,
                    ) -> candle_core::Result<Tensor> {
                        let t = self.st.load(name, dev)?;
                        if t.dtype() != dtype {
                            t.to_dtype(dtype)
                        } else {
                            Ok(t)
                        }
                    }
                    fn get_unchecked(
                        &self,
                        name: &str,
                        dtype: DType,
                        dev: &Device,
                    ) -> candle_core::Result<Tensor> {
                        let t = self.st.load(name, dev)?;
                        if t.dtype() != dtype {
                            t.to_dtype(dtype)
                        } else {
                            Ok(t)
                        }
                    }
                    fn contains_tensor(&self, name: &str) -> bool {
                        self.st.get(name).is_ok()
                    }
                }
                let inner: Box<dyn candle_nn::var_builder::SimpleBackend> =
                    Box::new(MmapBackend { st });
                let adapters =
                    super::lora::load_lora_adapters(&self.pending_loras, &self.base.progress)?;
                let specs: Vec<super::lora::Flux2LoraSpec<'_>> = adapters
                    .iter()
                    .zip(self.pending_loras.iter())
                    .map(|(adapter, w)| super::lora::Flux2LoraSpec {
                        adapter: adapter.as_ref(),
                        scale: w.scale,
                        path_hash: super::lora::lora_path_hash(&w.path),
                    })
                    .collect();
                let wrapped = super::lora::wrap_backend_with_lora(
                    inner,
                    &specs,
                    super::lora::Flux2KeySpace::Diffusers,
                    &self.base.progress,
                    None,
                )?;
                (
                    candle_nn::VarBuilder::from_backend(wrapped, gpu_dtype, device.clone()),
                    None,
                )
            } else if self.offload {
                (
                    crate::weight_loader::load_safetensors_with_progress(
                        &xformer_paths,
                        gpu_dtype,
                        &Device::Cpu,
                        "Flux.2 transformer (offload blocks)",
                        &self.base.progress,
                    )?,
                    Some("Loading Flux.2 transformer (offload, BF16)"),
                )
            } else {
                (
                    crate::weight_loader::load_safetensors_with_progress(
                        &xformer_paths,
                        gpu_dtype,
                        device,
                        "Flux.2 transformer",
                        &self.base.progress,
                    )?,
                    None,
                )
            };
            if let Some(label) = offloaded_label {
                return Ok((
                    Flux2TransformerWrapper::Offloaded(
                        super::transformer::OffloadedFlux2Transformer::new(
                            cfg,
                            flux_vb,
                            device,
                            self.base.gpu_ordinal,
                            activation_budget,
                            &self.base.progress,
                        )?,
                    ),
                    label,
                ));
            }
            let label = if has_lora {
                "Loading Flux.2 transformer (GPU, BF16 + LoRA)"
            } else {
                "Loading Flux.2 transformer (GPU, BF16)"
            };
            Ok((
                Flux2TransformerWrapper::BF16(super::transformer::Flux2Transformer::new(
                    cfg, flux_vb,
                )?),
                label,
            ))
        }
    }

    /// `true` when the transformer is a single `.safetensors` file whose
    /// tensor keys are BFL-native (`model.diffusion_model.*`). Returns
    /// `true` for `BflNative`, `BflNativeRoot`, and `Nvfp4` — all route
    /// through `SingleFileBackend` (the NVFP4 path exposes packed FP4,
    /// FP8 block scales, and tensor scales as streaming subkeys; the BFL
    /// variants pass tensors through directly). Sharded loads (HF diffusers layout) and any
    /// non-safetensors path skip this detection. Header-peeks the file
    /// once per load — a few KB read.
    fn is_bfl_native_single_file(&self) -> bool {
        if !self.base.paths.transformer_shards.is_empty() {
            return false;
        }
        let path = &self.base.paths.transformer;
        let is_safetensors = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
        if !is_safetensors {
            return false;
        }
        matches!(
            super::single_file::detect_format(path),
            Ok(super::single_file::Flux2SingleFileFormat::BflNative)
                | Ok(super::single_file::Flux2SingleFileFormat::BflNativeRoot)
                | Ok(super::single_file::Flux2SingleFileFormat::Nvfp4)
        )
    }

    fn is_nvfp4_single_file(&self) -> bool {
        if !self.base.paths.transformer_shards.is_empty() {
            return false;
        }
        let path = &self.base.paths.transformer;
        let is_safetensors = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
        if !is_safetensors {
            return false;
        }
        matches!(
            super::single_file::detect_format(path),
            Ok(super::single_file::Flux2SingleFileFormat::Nvfp4)
        )
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
                0,
            )?;
            self.base.loaded.as_mut().unwrap().transformer = Some(transformer);
            self.base
                .progress
                .stage_done("Reloading Flux.2 transformer", reload_start.elapsed());
        }
        Ok(())
    }

    fn should_delay_transformer_reload_for_prompt_encode(
        load_strategy: LoadStrategy,
        transformer_loaded: bool,
    ) -> bool {
        load_strategy == LoadStrategy::Eager && !transformer_loaded
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
    /// layers 9, 18, 27 and stacking them to produce the context embedding.
    ///
    /// - Klein-4B: Qwen3-4B (hidden=2560) → stacked dim = 2560 * 3 = 7680
    /// - Klein-9B: Qwen3-8B (hidden=4096) → stacked dim = 4096 * 3 = 12288
    ///
    /// Both use the same 36-layer Qwen3 architecture. Layers 9, 18, 27 correspond
    /// to roughly 1/4, 1/2, 3/4 depth.
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
                progress.phase_done(
                    crate::ProgressPhase::PromptEncode,
                    "Encoding prompt (Qwen3)",
                    encode_start.elapsed(),
                );
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
    ///
    /// GGUF variants keep weights quantized in VRAM via QMatMul (~6GB for Q4 9B),
    /// so both Klein-4B and Klein-9B fit comfortably in eager mode on 24GB GPUs.
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
        let transformer_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.transformer.clone()),
            false,
        );
        let device = crate::device::resolve_device(Some(transformer_ref), || {
            crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)
        })?;
        let gpu_dtype = crate::engine::gpu_dtype(&device);

        tracing::info!("GPU device: {:?}, GPU dtype: {:?}", device, gpu_dtype);

        // --- Load transformer on GPU first ---
        let flux2_cfg = self.resolve_config();
        let xformer_stage = Instant::now();
        let (transformer, xformer_label) =
            self.load_transformer(&flux2_cfg, gpu_dtype, &device, 0)?;
        self.base
            .progress
            .stage_done(xformer_label, xformer_stage.elapsed());

        // --- Load VAE on GPU ---
        let vae_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.vae.clone()),
            false,
        );
        let vae_device = crate::device::resolve_device(Some(vae_ref), || Ok(device.clone()))?;
        self.base.progress.stage_start("Loading VAE (GPU)");
        let vae_stage = Instant::now();
        tracing::info!(path = %self.base.paths.vae.display(), "loading VAE on GPU...");
        let vae_cfg = Flux2VaeConfig::klein();
        // Resolve VAE precision once at load — see LoadedFlux2::vae_dtype.
        let vae_dtype = crate::device::resolve_vae_dtype(gpu_dtype);
        let vae_vb = self.load_vae_var_builder(vae_dtype, &vae_device, "VAE")?;
        let vae = Flux2AutoEncoder::new(&vae_cfg, vae_vb)?;
        self.base
            .progress
            .stage_done("Loading VAE (GPU)", vae_stage.elapsed());
        tracing::info!("VAE loaded on GPU");

        // --- Resolve and load Qwen3 text encoder ---
        // Log the raw reading (matches `nvidia-smi`); budget the variant
        // selection against the reserve-adjusted value.
        let free_raw = free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        if free_raw > 0 {
            self.base.progress.info(&format!(
                "Free VRAM after transformer+VAE: {}",
                fmt_gb(free_raw)
            ));
        }

        self.base.progress.stage_start("Selecting Qwen3 encoder");
        let resolve_start = Instant::now();
        let qwen3_size = self.qwen3_size();
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
                qwen3_size,
            )?
        };
        self.base
            .progress
            .stage_done("Selecting Qwen3 encoder", resolve_start.elapsed());

        let qwen3_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| adv.qwen.clone(),
            true,
        );
        let auto_enc_device = if on_gpu { device.clone() } else { cpu.clone() };
        let enc_device_owned =
            crate::device::resolve_device(Some(qwen3_ref), || Ok(auto_enc_device.clone()))?;
        let enc_device = &enc_device_owned;
        let on_gpu = !enc_device.is_cpu();
        let enc_dtype = if on_gpu { gpu_dtype } else { DType::F32 };
        let bf16_cfg = self.qwen3_bf16_config();

        let enc_stage_label = format!("Loading Qwen3 encoder ({device_label})");
        self.base.progress.stage_start(&enc_stage_label);
        let enc_stage = Instant::now();
        let text_tokenizer = self.load_text_tokenizer(&text_tokenizer_path)?;

        let text_encoder = if is_gguf {
            encoders::qwen3::Qwen3Encoder::load_gguf_with_tokenizer(
                &encoder_paths[0],
                &text_tokenizer_path,
                Some(text_tokenizer),
                enc_device,
                &bf16_cfg,
            )?
        } else {
            encoders::qwen3::Qwen3Encoder::load_bf16_with_tokenizer(
                &encoder_paths,
                &text_tokenizer_path,
                Some(text_tokenizer),
                enc_device,
                enc_dtype,
                &bf16_cfg,
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
            vae_dtype,
        });

        tracing::info!(model = %self.base.model_name, "all Flux.2 model components loaded successfully");
        Ok(())
    }

    /// Generate an image using sequential loading strategy (load-use-drop).
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let text_tokenizer_path = self.validate_paths()?;
        let is_gguf = self.is_gguf_transformer();

        match flux2_offload_decision(
            self.offload,
            is_gguf,
            self.is_nvfp4_single_file(),
            !self.pending_loras.is_empty(),
        ) {
            Flux2OffloadDecision::Disabled => {}
            Flux2OffloadDecision::Unsupported(reason) => bail!("{reason}"),
            Flux2OffloadDecision::Selected => {}
        }

        if let Some(warning) = check_memory_budget(&self.base.paths, LoadStrategy::Sequential) {
            self.base.progress.info(&warning);
        }

        let transformer_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.transformer.clone()),
            false,
        );
        let device = crate::device::resolve_device(Some(transformer_ref), || {
            crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)
        })?;
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
        // Check prompt cache first — skip encoder load entirely on cache hit.
        // This saves ~1-5s per batch image (encoder load + VRAM allocation).
        let cache_key = prompt_text_key(&req.prompt);
        let txt_emb = if let Some(tensor) =
            restore_cached_tensor(&self.prompt_cache, &cache_key, &device, gpu_dtype)?
        {
            self.base.progress.cache_hit("prompt conditioning");
            tensor
        } else {
            // Reserve-adjusted reading drives the Qwen3 variant selection.
            let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
            self.base.progress.stage_start("Selecting Qwen3 encoder");
            let resolve_start = Instant::now();
            let qwen3_size = self.qwen3_size();
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
                    qwen3_size,
                )?
            };
            self.base
                .progress
                .stage_done("Selecting Qwen3 encoder", resolve_start.elapsed());

            let qwen3_ref = effective_device_ref(
                self.pending_placement.as_ref(),
                |adv| adv.qwen.clone(),
                true,
            );
            let auto_enc_device = if on_gpu { device.clone() } else { Device::Cpu };
            let enc_device_owned =
                crate::device::resolve_device(Some(qwen3_ref), || Ok(auto_enc_device.clone()))?;
            let enc_device = &enc_device_owned;
            let on_gpu = !enc_device.is_cpu();
            let enc_dtype = if on_gpu { gpu_dtype } else { DType::F32 };
            let bf16_cfg = self.qwen3_bf16_config();

            // Pre-flight memory check
            let enc_size: u64 = encoder_paths
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
                .sum();
            let enc_activation_budget = crate::device::activation_bytes(
                req.width,
                req.height,
                1,
                crate::device::dtype_bytes(enc_dtype),
                crate::device::ActivationFamily::SmallTransformer,
            );
            preflight_memory_check("Qwen3 encoder", enc_size, enc_activation_budget)?;
            if let Some(status) = memory_status_string() {
                self.base.progress.info(&status);
            }

            let enc_stage_label = format!("Loading Qwen3 encoder ({device_label})");
            self.base.progress.stage_start(&enc_stage_label);
            let enc_stage = Instant::now();
            let text_tokenizer = self.load_text_tokenizer(&text_tokenizer_path)?;

            let mut text_encoder = if is_gguf {
                encoders::qwen3::Qwen3Encoder::load_gguf_with_tokenizer(
                    &encoder_paths[0],
                    &text_tokenizer_path,
                    Some(text_tokenizer),
                    enc_device,
                    &bf16_cfg,
                )?
            } else {
                encoders::qwen3::Qwen3Encoder::load_bf16_with_tokenizer(
                    &encoder_paths,
                    &text_tokenizer_path,
                    Some(text_tokenizer),
                    enc_device,
                    enc_dtype,
                    &bf16_cfg,
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

            txt_emb
        };

        let latent_h = height.div_ceil(8);
        let latent_w = width.div_ceil(8);

        // Pre-compute timestep schedule (needed before mixing for img2img)
        let image_seq_len = (height / 16) * (width / 16);
        let mut timesteps = sampling::get_schedule(req.steps as usize, image_seq_len);

        if req.source_image.is_some() {
            let (trimmed, start_index) =
                crate::img2img::trim_schedule_tail(&timesteps, req.steps as usize, req.strength);
            timesteps = trimmed;
            tracing::info!(
                strength = req.strength,
                start_index,
                start_timestep = timesteps[0],
                schedule = ?timesteps,
                remaining_steps = timesteps.len().saturating_sub(1),
                "img2img: truncated schedule from strength"
            );
        }

        // Generate noise / encode source image for img2img. Source-image
        // requests pre-encode in a VAE-only phase, then drop VAE before the
        // transformer load. Klein-9B BF16 cannot keep transformer+VAE
        // co-resident on 24 GB cards.
        let (img, inpaint_ctx) = if let Some(ref source_bytes) = req.source_image {
            let start_t = timesteps[0];
            let (vae, _vae_dtype) = self.load_sequential_vae(&device, gpu_dtype)?;

            self.base
                .progress
                .stage_start("Encoding source image (VAE)");
            let encode_start = Instant::now();
            let source_tensor = crate::img_utils::decode_source_image(
                source_bytes,
                req.width,
                req.height,
                Self::img2img_source_normalize_range(),
                &device,
                gpu_dtype,
            )?;
            let encoded = vae.encode(&source_tensor)?;
            self.base.progress.phase_done(
                crate::ProgressPhase::Vae,
                "Encoding source image (VAE)",
                encode_start.elapsed(),
            );

            let prepared = crate::img2img::prepare_flow_match_img2img(
                &encoded,
                seed,
                &[1, 32, latent_h, latent_w],
                start_t,
                req.mask_image.as_deref(),
                latent_h,
                latent_w,
                &device,
                gpu_dtype,
            )?;
            drop(vae);
            drop(encoded);
            drop(source_tensor);
            device.synchronize()?;
            self.base.progress.info("Freed VAE after source encoding");
            (prepared.initial_latents, prepared.inpaint_ctx)
        } else {
            let img = crate::engine::seeded_randn(
                seed,
                &[1, 32, latent_h, latent_w],
                &device,
                gpu_dtype,
            )?;
            (img, None)
        };

        let state = Flux2State::new(&txt_emb, &img)?;
        let inpaint_ctx = inpaint_ctx
            .as_ref()
            .map(crate::img2img::pack_flux_inpaint_context)
            .transpose()?;

        // --- Phase 2: Load transformer, denoise ---
        let xformer_size = std::fs::metadata(&self.base.paths.transformer)
            .map(|m| m.len())
            .unwrap_or(0);
        let xformer_activation_budget = crate::device::activation_bytes(
            req.width,
            req.height,
            1,
            crate::device::dtype_bytes(gpu_dtype),
            crate::device::ActivationFamily::Flux2Dit,
        );
        preflight_memory_check(
            "Flux.2 transformer",
            xformer_size,
            xformer_activation_budget,
        )?;
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        let flux2_cfg = self.resolve_config();
        let xformer_stage = Instant::now();
        let (transformer, xformer_label) =
            self.load_transformer(&flux2_cfg, gpu_dtype, &device, xformer_activation_budget)?;
        self.base
            .progress
            .stage_done(xformer_label, xformer_stage.elapsed());

        let denoise_label = format!("Denoising ({} steps)", timesteps.len().saturating_sub(1));
        self.base.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        let previewer = crate::latent_preview::LatentPreviewer::flux2(height, width);
        let img = transformer.denoise(
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            req.guidance,
            &self.base.progress,
            inpaint_ctx.as_ref(),
            Some(&previewer),
        )?;

        let img = sampling::unpack(&img, height, width)?;

        self.base
            .progress
            .stage_done(&denoise_label, denoise_start.elapsed());

        // Drop transformer + state to free memory for VAE decode
        drop(inpaint_ctx);
        drop(transformer);
        self.base.progress.info("Freed Flux.2 transformer");
        drop(state);
        drop(txt_emb);
        device.synchronize()?;
        tracing::info!("Transformer dropped (sequential mode), decoding VAE...");

        let (vae, vae_dtype) = self.load_sequential_vae(&device, gpu_dtype)?;

        // --- Phase 3: VAE decode ---
        self.base.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();
        // DEBUG: dump pre-VAE latent (B, 32, H, W) when MOLD_FLUX2_DUMP_LATENT is set.
        if let Ok(dump_path) = std::env::var("MOLD_FLUX2_DUMP_LATENT") {
            let latent_f32 = img
                .to_dtype(DType::F32)?
                .to_device(&candle_core::Device::Cpu)?;
            let dims = latent_f32.dims().to_vec();
            let v: Vec<f32> = latent_f32.flatten_all()?.to_vec1()?;
            let mut bytes = Vec::with_capacity(8 * 4 + v.len() * 4);
            bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in &dims {
                bytes.extend_from_slice(&(*d as u32).to_le_bytes());
            }
            for x in &v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            std::fs::write(&dump_path, &bytes)?;
            tracing::info!(path = %dump_path, dims = ?dims, "dumped pre-VAE latent");
        }
        let img_for_vae = img.to_dtype(vae_dtype)?;
        let device_for_sync = device.clone();
        let img = crate::vae_tiling::decode_with_oom_fallback(
            &img_for_vae,
            |latents| vae.decode(latents).map_err(Into::into),
            || {
                if let Err(e) = device_for_sync.synchronize() {
                    tracing::warn!(
                        "FLUX2 (sequential) device.synchronize() after VAE OOM failed: {e}"
                    );
                }
            },
        )?;

        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?;

        self.base.progress.phase_done(
            crate::ProgressPhase::Vae,
            "VAE decode",
            vae_decode_start.elapsed(),
        );

        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &img,
            req.resolved_output_format(),
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "sequential generation complete");

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

// ---------------------------------------------------------------------------
// InferenceEngine implementation
// ---------------------------------------------------------------------------

impl Flux2Engine {
    fn generate_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
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
        // Sequential mode: load-use-drop each component
        if self.uses_sequential_generate_path(req) {
            // A stale/mismatched eager plan must not leave its transformer,
            // VAE, or text encoder resident while the LoRA path constructs
            // staged replacements. Production planning selects Sequential for
            // Flux.2 LoRA requests, but this defensive release keeps direct
            // callers and pre-existing cached engines from doubling the peak.
            self.base.unload();
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components. After a previous request we
        // intentionally drop the transformer before VAE decode, but the VAE
        // and Qwen3 shell remain resident. In that warm state, reloading the
        // transformer before prompt encoding recreates the highest peak
        // (transformer + Qwen3) and can OOM on 24 GB cards when queued
        // requests arrive back-to-back. Encode/drop Qwen3 first, then reload
        // the transformer for denoising.
        if self.base.loaded.is_none() {
            self.load()?;
        }
        let delay_transformer_reload = self.base.loaded.as_ref().is_some_and(|loaded| {
            Self::should_delay_transformer_reload_for_prompt_encode(
                self.base.load_strategy,
                loaded.transformer.is_some(),
            )
        });
        if delay_transformer_reload {
            tracing::info!(
                "delaying Flux.2 transformer reload until after prompt encode to reduce peak VRAM"
            );
        }

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
        let txt_emb = {
            let progress = &self.base.progress;
            let loaded = self
                .base
                .loaded
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
            let cache_key = prompt_text_key(&req.prompt);
            if let Some(tensor) =
                restore_cached_tensor(&self.prompt_cache, &cache_key, &loaded.device, loaded.dtype)?
            {
                progress.cache_hit("prompt conditioning");
                tensor
            } else {
                // Cache miss — restore encoder if it was dropped or parked after
                // a previous generation.
                if loaded.text_encoder.model.is_none() {
                    let label = if loaded.text_encoder.is_parked() {
                        "Unparking Qwen3 encoder (CPU→GPU)"
                    } else {
                        "Reloading Qwen3 encoder"
                    };
                    progress.stage_start(label);
                    let reload_start = Instant::now();
                    if loaded.text_encoder.is_parked() {
                        loaded.text_encoder.unpark_to_gpu(progress)?;
                    } else {
                        loaded.text_encoder.reload(progress)?;
                    }
                    progress.stage_done(label, reload_start.elapsed());
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

                // Free GPU VRAM for denoising. With `MOLD_KEEP_TE_RAM=1` and the
                // BF16 encoder, parameters move to host RAM instead of being
                // released — saves ~10 s on the next request. GGUF and Metal
                // flow through the original drop path.
                if loaded.text_encoder.on_gpu || loaded.device.is_metal() {
                    let park_mode = crate::device::keep_te_in_ram()
                        && !loaded.device.is_metal()
                        && !loaded.text_encoder.is_quantized;
                    if park_mode {
                        loaded.text_encoder.park_to_cpu()?;
                        tracing::info!(
                            on_gpu = loaded.text_encoder.on_gpu,
                            "Qwen3 encoder parked to CPU host RAM"
                        );
                    } else {
                        loaded.text_encoder.drop_weights();
                        tracing::info!(
                            on_gpu = loaded.text_encoder.on_gpu,
                            "Qwen3 encoder dropped to free memory for denoising"
                        );
                    }
                }

                txt_emb
            }
        };

        self.reload_transformer_if_needed()?;

        let loaded = self
            .base
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
        let progress = &self.base.progress;

        // 2. Prepare latent space dimensions and timestep schedule
        let latent_h = height.div_ceil(8);
        let latent_w = width.div_ceil(8);

        // Pre-compute timestep schedule (needed before mixing for img2img)
        let image_seq_len = (height / 16) * (width / 16);
        let mut timesteps = sampling::get_schedule(req.steps as usize, image_seq_len);

        if req.source_image.is_some() {
            let (trimmed, start_index) =
                crate::img2img::trim_schedule_tail(&timesteps, req.steps as usize, req.strength);
            timesteps = trimmed;
            tracing::info!(
                strength = req.strength,
                start_index,
                start_timestep = timesteps[0],
                schedule = ?timesteps,
                remaining_steps = timesteps.len().saturating_sub(1),
                "img2img: truncated schedule from strength"
            );
        }

        // 3. Generate noise / encode source image for img2img
        let (img, inpaint_ctx) = if let Some(ref source_bytes) = req.source_image {
            let start_t = timesteps[0];

            progress.stage_start("Encoding source image (VAE)");
            let encode_start = Instant::now();
            let source_tensor = crate::img_utils::decode_source_image(
                source_bytes,
                req.width,
                req.height,
                Self::img2img_source_normalize_range(),
                &loaded.device,
                loaded.vae_dtype,
            )?;
            let encoded = loaded.vae.encode(&source_tensor)?;
            progress.phase_done(
                crate::ProgressPhase::Vae,
                "Encoding source image (VAE)",
                encode_start.elapsed(),
            );

            let prepared = crate::img2img::prepare_flow_match_img2img(
                &encoded,
                seed,
                &[1, 32, latent_h, latent_w],
                start_t,
                req.mask_image.as_deref(),
                latent_h,
                latent_w,
                &loaded.device,
                loaded.dtype,
            )?;
            (prepared.initial_latents, prepared.inpaint_ctx)
        } else {
            let img = crate::engine::seeded_randn(
                seed,
                &[1, 32, latent_h, latent_w],
                &loaded.device,
                loaded.dtype,
            )?;
            (img, None)
        };

        // 4. Build sampling state
        let state = Flux2State::new(&txt_emb, &img)?;
        let inpaint_ctx = inpaint_ctx
            .as_ref()
            .map(crate::img2img::pack_flux_inpaint_context)
            .transpose()?;

        let denoise_label = format!("Denoising ({} steps)", timesteps.len().saturating_sub(1));
        progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();
        tracing::info!(
            steps = timesteps.len().saturating_sub(1),
            "running denoising loop..."
        );

        // 5. Denoise
        let transformer = loaded
            .transformer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("transformer not loaded"))?;
        let previewer = crate::latent_preview::LatentPreviewer::flux2(height, width);
        let img = transformer.denoise(
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            req.guidance,
            progress,
            inpaint_ctx.as_ref(),
            Some(&previewer),
        )?;

        // 6. Unpack latent to spatial
        let img = sampling::unpack(&img, height, width)?;
        progress.stage_done(&denoise_label, denoise_start.elapsed());
        tracing::info!("denoising complete, decoding VAE...");

        // Free denoising intermediates and transformer before VAE decode.
        // The transformer consumes most of VRAM — VAE decode needs that
        // memory for conv2d intermediates. Transformer is reloaded next generate.
        drop(inpaint_ctx);
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
        // DEBUG: dump pre-VAE latent when MOLD_FLUX2_DUMP_LATENT is set.
        if let Ok(dump_path) = std::env::var("MOLD_FLUX2_DUMP_LATENT") {
            let latent_f32 = img
                .to_dtype(DType::F32)?
                .to_device(&candle_core::Device::Cpu)?;
            let dims = latent_f32.dims().to_vec();
            let v: Vec<f32> = latent_f32.flatten_all()?.to_vec1()?;
            let mut bytes = Vec::with_capacity(8 * 4 + v.len() * 4);
            bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in &dims {
                bytes.extend_from_slice(&(*d as u32).to_le_bytes());
            }
            for x in &v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            std::fs::write(&dump_path, &bytes)?;
            tracing::info!(path = %dump_path, dims = ?dims, "dumped pre-VAE latent (parallel)");
        }
        let img_for_vae = img.to_dtype(loaded.vae_dtype)?;
        let vae = &loaded.vae;
        let device_for_sync = loaded.device.clone();
        let img = crate::vae_tiling::decode_with_oom_fallback(
            &img_for_vae,
            |latents| vae.decode(latents).map_err(Into::into),
            || {
                if let Err(e) = device_for_sync.synchronize() {
                    tracing::warn!(
                        "FLUX2 (parallel) device.synchronize() after VAE OOM failed: {e}"
                    );
                }
            },
        )?;

        // 8. Convert to u8 image: clamp to [-1, 1], map to [0, 255]
        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?; // remove batch dim: [3, H, W]

        progress.phase_done(
            crate::ProgressPhase::Vae,
            "VAE decode",
            vae_decode_start.elapsed(),
        );
        tracing::info!("VAE decode complete, encoding output image...");

        // 9. Convert candle tensor to image bytes
        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &img,
            req.resolved_output_format(),
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "generation complete");

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

impl InferenceEngine for Flux2Engine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.base.progress.checkpoint()?;
        self.pending_placement = req.placement.clone();
        self.pending_loras = effective_flux2_loras(req);
        let result = self.generate_inner(req);
        self.pending_placement = None;
        self.pending_loras.clear();
        result
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

    fn load_for_request(&mut self, req: &GenerateRequest) -> Result<()> {
        self.pending_placement = req.placement.clone();
        self.pending_loras = effective_flux2_loras(req);
        let result = if !self.pending_loras.is_empty()
            && self.base.load_strategy != LoadStrategy::Sequential
        {
            Err(anyhow::anyhow!(
                "Flux.2 LoRA requests require a sequential engine load plan; \
                 refusing to preload an unadapted transformer"
            ))
        } else {
            Flux2Engine::load(self)
        };
        self.pending_placement = None;
        self.pending_loras.clear();
        result
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

    fn set_cancellation_token(&mut self, token: crate::progress::InferenceCancellationToken) {
        self.base.set_cancellation_token(token);
    }

    fn clear_cancellation_token(&mut self) {
        self.base.clear_cancellation_token();
    }

    fn batch_execution_capability(&self) -> crate::BatchExecutionCapability {
        crate::batch_execution_capability_for_family("flux2")
            .expect("production Flux.2 batch capability must be registered")
    }

    fn model_paths(&self) -> Option<&mold_core::ModelPaths> {
        Some(&self.base.paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::variant_resolution::Qwen3Size;
    use crate::engine::LoadStrategy;
    use crate::shared_pool::SharedPool;
    use mold_core::ModelPaths;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokenizers::models::bpe::BPE;

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

    #[test]
    fn lora_request_refuses_eager_preload_before_touching_model_files() {
        let dir = temp_test_dir("mold-flux2-lora-eager-preload");
        let mut engine = Flux2Engine::new(
            "flux2-klein:bf16".to_string(),
            flux2_model_paths(&dir, "missing-transformer.safetensors", vec![], None),
            None,
            LoadStrategy::Eager,
            0,
            false,
            None,
        );
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "portrait",
            "model": "flux2-klein:bf16",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 1.0,
            "batch_size": 1,
            "loras": [{
                "path": dir.join("adapter.safetensors"),
                "scale": 1.0
            }]
        }))
        .unwrap();

        let error = engine.load_for_request(&request).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("LoRA requests require a sequential engine load plan"),
            "got: {error:#}"
        );
        assert!(engine.pending_loras.is_empty());
        assert!(engine.pending_placement.is_none());
        assert!(
            !engine.is_loaded(),
            "fail-closed request loading must not retain partial eager components"
        );

        fs::remove_dir_all(dir).ok();
    }

    fn flux2_model_paths(
        dir: &Path,
        transformer_name: &str,
        text_encoder_files: Vec<PathBuf>,
        t5_encoder: Option<PathBuf>,
    ) -> ModelPaths {
        ModelPaths {
            transformer: dir.join(transformer_name),
            transformer_shards: vec![],
            vae: dir.join("vae.safetensors"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files,
            text_tokenizer: Some(dir.join("tokenizer.json")),
            decoder: None,
        }
    }

    fn test_generate_request() -> GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "portrait",
            "model": "flux2-klein:bf16",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 1.0,
            "batch_size": 1
        }))
        .unwrap()
    }

    #[test]
    fn flux2_img2img_uses_minus_one_to_one_source_normalization() {
        assert_eq!(
            Flux2Engine::img2img_source_normalize_range(),
            crate::img_utils::NormalizeRange::MinusOneToOne
        );
    }

    #[test]
    fn sequential_img2img_encodes_source_before_transformer_load() {
        assert!(
            Flux2Engine::sequential_img2img_preencodes_source(),
            "sequential Flux.2 img2img must not keep the VAE resident while loading the transformer"
        );
    }

    #[test]
    fn source_image_requires_sequential_generation_even_for_eager_engines() {
        let dir = temp_test_dir("mold-flux2-source-sequential");
        let engine = Flux2Engine::new(
            "flux2-klein:bf16".to_string(),
            flux2_model_paths(&dir, "transformer.safetensors", vec![], None),
            None,
            LoadStrategy::Eager,
            0,
            false,
            None,
        );
        let mut req = test_generate_request();
        req.source_image = Some(vec![0x89, 0x50, 0x4e, 0x47]);

        assert!(engine.uses_sequential_generate_path(&req));
        req.source_image = None;
        assert!(!engine.uses_sequential_generate_path(&req));
    }

    #[test]
    fn eager_warm_request_delays_transformer_reload_until_after_prompt_encode() {
        assert!(
            Flux2Engine::should_delay_transformer_reload_for_prompt_encode(
                LoadStrategy::Eager,
                false
            ),
            "warm eager requests with a dropped transformer must encode/drop Qwen3 before reload"
        );
        assert!(
            !Flux2Engine::should_delay_transformer_reload_for_prompt_encode(
                LoadStrategy::Eager,
                true
            ),
            "fully loaded eager requests should keep the existing hot path"
        );
        assert!(
            !Flux2Engine::should_delay_transformer_reload_for_prompt_encode(
                LoadStrategy::Sequential,
                false
            ),
            "sequential mode already handles load-use-drop ordering"
        );
    }

    #[test]
    fn flux2_model_name_controls_transformer_and_encoder_config() {
        let base_dir = temp_test_dir("mold-flux2-config");
        let standard = Flux2Engine::new(
            "flux2-klein:q8".to_string(),
            flux2_model_paths(&base_dir, "transformer.gguf", vec![], None),
            None,
            LoadStrategy::Sequential,
            0,
            false,
            None,
        );
        let nine_b = Flux2Engine::new(
            "flux2-klein-9b:q8".to_string(),
            flux2_model_paths(&base_dir, "transformer.gguf", vec![], None),
            None,
            LoadStrategy::Sequential,
            0,
            false,
            None,
        );

        let standard_cfg = standard.resolve_config();
        let nine_b_cfg = nine_b.resolve_config();

        assert_eq!(standard_cfg.hidden_size, 3072);
        assert_eq!(standard_cfg.context_in_dim, 7680);
        assert_eq!(standard.qwen3_size(), Qwen3Size::B4);
        assert_eq!(standard.qwen3_bf16_config().hidden_size, 2560);

        assert_eq!(nine_b_cfg.hidden_size, 4096);
        assert_eq!(nine_b_cfg.context_in_dim, 12288);
        assert_eq!(nine_b.qwen3_size(), Qwen3Size::B8);
        assert_eq!(nine_b.qwen3_bf16_config().hidden_size, 4096);

        fs::remove_dir_all(base_dir).ok();
    }

    #[test]
    fn flux2_text_encoder_paths_use_shards_or_t5_fallback() {
        let dir = temp_test_dir("mold-flux2-paths");
        let shard_a = touch(&dir, "encoder-1.safetensors");
        let shard_b = touch(&dir, "encoder-2.safetensors");
        let fallback = touch(&dir, "encoder.safetensors");

        let sharded = Flux2Engine::new(
            "flux2-klein:q8".to_string(),
            flux2_model_paths(
                &dir,
                "transformer.gguf",
                vec![shard_a.clone(), shard_b.clone()],
                Some(fallback.clone()),
            ),
            None,
            LoadStrategy::Sequential,
            0,
            false,
            None,
        );
        assert_eq!(sharded.text_encoder_paths(), vec![shard_a, shard_b]);

        let fallback_engine = Flux2Engine::new(
            "flux2-klein:q8".to_string(),
            flux2_model_paths(&dir, "transformer.gguf", vec![], Some(fallback.clone())),
            None,
            LoadStrategy::Sequential,
            0,
            false,
            None,
        );
        assert_eq!(fallback_engine.text_encoder_paths(), vec![fallback]);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn flux2_loads_qwen3_tokenizer_through_shared_pool() {
        let dir = temp_test_dir("mold-flux2-tokenizer-pool");
        let tokenizer_path = dir.join("tokenizer.json");
        tokenizers::Tokenizer::new(BPE::default())
            .save(&tokenizer_path, false)
            .unwrap();

        let shared_pool = Arc::new(Mutex::new(SharedPool::new()));
        let pooled = shared_pool
            .lock()
            .unwrap()
            .load_tokenizer(&tokenizer_path)
            .unwrap();

        let engine = Flux2Engine::new(
            "flux2-klein:q8".to_string(),
            flux2_model_paths(&dir, "transformer.gguf", vec![], None),
            None,
            LoadStrategy::Sequential,
            0,
            false,
            Some(shared_pool),
        );

        let loaded = engine.load_text_tokenizer(&tokenizer_path).unwrap();

        assert!(Arc::ptr_eq(&pooled, &loaded));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn flux2_forced_offload_uses_sequential_generation_path() {
        let dir = temp_test_dir("mold-flux2-offload-sequential");
        let engine = Flux2Engine::new(
            "flux2-klein:bf16".to_string(),
            flux2_model_paths(&dir, "transformer.safetensors", vec![], None),
            None,
            LoadStrategy::Eager,
            0,
            true,
            None,
        );

        assert!(
            engine.uses_sequential_generate_path(&test_generate_request()),
            "Flux.2 --offload requests must reach the engine and select the \
             staged generation path instead of being silently ignored"
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn flux2_offload_decision_gates_current_unsupported_cases() {
        assert_eq!(
            flux2_offload_decision(false, false, false, false),
            Flux2OffloadDecision::Disabled
        );
        assert_eq!(
            flux2_offload_decision(true, false, false, false),
            Flux2OffloadDecision::Selected
        );
        assert_eq!(
            flux2_offload_decision(true, false, true, false),
            Flux2OffloadDecision::Disabled
        );
        assert!(matches!(
            flux2_offload_decision(true, true, false, false),
            Flux2OffloadDecision::Unsupported(reason)
                if reason.contains("GGUF variants")
        ));
        assert!(matches!(
            flux2_offload_decision(true, false, false, true),
            Flux2OffloadDecision::Unsupported(reason)
                if reason.contains("LoRA")
        ));
    }

    #[test]
    fn flux2_selected_bf16_offload_reaches_runtime_loader() {
        let dir = temp_test_dir("mold-flux2-offload-loader");
        let transformer = touch(&dir, "transformer.safetensors");
        let vae = touch(&dir, "vae.safetensors");
        let encoder = touch(&dir, "encoder.safetensors");
        let tokenizer = touch(&dir, "tokenizer.json");
        let mut engine = Flux2Engine::new(
            "flux2-klein:bf16".to_string(),
            ModelPaths {
                transformer,
                transformer_shards: vec![],
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
                text_encoder_files: vec![encoder],
                text_tokenizer: Some(tokenizer),
                decoder: None,
            },
            None,
            LoadStrategy::Sequential,
            0,
            true,
            None,
        );
        let cfg = engine.resolve_config();
        let txt_emb = Tensor::zeros((1, 1, cfg.context_in_dim), DType::F32, &Device::Cpu).unwrap();
        engine.prompt_cache.lock().unwrap().insert(
            prompt_text_key("a cat"),
            CachedTensor::from_tensor(&txt_emb).unwrap(),
        );
        let req = GenerateRequest {
            prompt: "a cat".to_string(),
            negative_prompt: None,
            model: "flux2-klein:bf16".to_string(),
            width: 64,
            height: 64,
            steps: 1,
            guidance: 0.0,
            seed: Some(1),
            batch_size: 1,
            output_format: None,
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            strength: 1.0,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: Some(mold_core::types::DevicePlacement {
                text_encoders: mold_core::types::DeviceRef::Cpu,
                advanced: Some(mold_core::types::AdvancedPlacement {
                    transformer: mold_core::types::DeviceRef::Cpu,
                    vae: mold_core::types::DeviceRef::Cpu,
                    ..Default::default()
                }),
            }),
        };

        let err = engine.generate_sequential(&req).unwrap_err().to_string();

        assert!(
            !err.contains("streaming is not implemented yet"),
            "selected BF16 offload must reach the runtime loader, got: {err}"
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn flux2_loads_vae_tensors_through_shared_pool() {
        let dir = temp_test_dir("mold-flux2-vae-pool");
        let vae_path = dir.join("vae.safetensors");
        let weight = 1.0f32.to_le_bytes();
        let mut tensors = HashMap::new();
        tensors.insert(
            "encoder.conv_in.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1], &weight).unwrap(),
        );
        serialize_to_file(&tensors, &None, &vae_path).unwrap();

        let shared_pool = Arc::new(Mutex::new(SharedPool::new()));
        let pooled = shared_pool
            .lock()
            .unwrap()
            .load_safetensors_cpu_tensors(std::slice::from_ref(&vae_path))
            .unwrap()
            .unwrap();

        let engine = Flux2Engine::new(
            "flux2-klein:q8".to_string(),
            flux2_model_paths(&dir, "transformer.gguf", vec![], None),
            None,
            LoadStrategy::Sequential,
            0,
            false,
            Some(shared_pool),
        );

        let loaded = engine.load_vae_cpu_tensors().unwrap().unwrap();

        assert!(Arc::ptr_eq(&pooled, &loaded));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn flux2_validate_paths_accepts_existing_files_and_returns_tokenizer() {
        let dir = temp_test_dir("mold-flux2-validate-ok");
        let transformer = touch(&dir, "transformer.gguf");
        let vae = touch(&dir, "vae.safetensors");
        let encoder = touch(&dir, "encoder.safetensors");
        let tokenizer = touch(&dir, "tokenizer.json");

        let engine = Flux2Engine::new(
            "flux2-klein:q8".to_string(),
            ModelPaths {
                transformer,
                transformer_shards: vec![],
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
                text_encoder_files: vec![encoder],
                text_tokenizer: Some(tokenizer.clone()),
                decoder: None,
            },
            None,
            LoadStrategy::Sequential,
            0,
            false,
            None,
        );

        assert_eq!(engine.validate_paths().unwrap(), tokenizer);
        assert!(engine.is_gguf_transformer());

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn flux2_validate_paths_requires_text_encoder_paths() {
        let dir = temp_test_dir("mold-flux2-validate-missing");
        let transformer = touch(&dir, "transformer.safetensors");
        let vae = touch(&dir, "vae.safetensors");
        let tokenizer = touch(&dir, "tokenizer.json");

        let engine = Flux2Engine::new(
            "flux2-klein:bf16".to_string(),
            ModelPaths {
                transformer,
                transformer_shards: vec![],
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
                text_tokenizer: Some(tokenizer),
                decoder: None,
            },
            None,
            LoadStrategy::Sequential,
            0,
            false,
            None,
        );

        let err = engine.validate_paths().unwrap_err();
        assert!(err.to_string().contains("text encoder paths required"));
        assert!(!engine.is_gguf_transformer());

        fs::remove_dir_all(dir).ok();
    }
}
