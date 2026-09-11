//! FLUX.2 inference engine (Klein 4B/9B and full Dev variants).
//!
//! Follows the same Eager + Sequential loading pattern as FluxEngine and ZImageEngine.
//!
//! Key differences from FLUX.1:
//! - Klein uses Qwen3; Dev streams the checkpoint-native Mistral3 encoder
//!   - Klein-4B: Qwen3-4B (hidden=2560), stacked layers → 7680-dim context
//!   - Klein-9B: Qwen3-8B (hidden=4096), stacked layers → 12288-dim context
//! - VAE has latent_channels=32 (not 16)
//! - Transformer has 128 input channels (not 64)
//! - 4D RoPE (not 3D)
//! - Klein has no guidance embedding; Dev uses guidance-distilled conditioning
//! - No pooled text vector input
//! - Linear timestep schedule (distilled, no time-shifting)

use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, LoraWeight, ModelPaths};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
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
// Prompt conditioning cache
// ---------------------------------------------------------------------------

/// The epoch of the FLUX.2 conditioning CONTRACT, folded into the prompt cache
/// key.
///
/// Bump it whenever the shape or the values of a cached conditioning tensor
/// stop being interchangeable with the previous build's. Epoch 2 is the
/// padding fix: a [klein] embedding is now 512 rows wide whatever the prompt
/// says, so a tensor cached under epoch 1 is a different tensor entirely, not
/// a smaller one. The cache is an in-process LRU and cannot outlive the binary
/// that filled it, so this is documentation of the contract rather than a
/// migration — but it is the key's job to say so, and a future on-disk cache
/// inherits the guarantee for free.
const FLUX2_CONDITIONING_EPOCH: u32 = 2;

/// The prompt cache key for one FLUX.2 conditioning tensor.
fn flux2_prompt_cache_key(prompt: &str) -> String {
    format!("v{FLUX2_CONDITIONING_EPOCH}:{}", prompt_text_key(prompt))
}

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

/// How a guided FLUX.2 [klein] base render issues its two predictions.
///
/// Resolved by the caller, which is the only place that knows the config, the
/// checkpoint's size, the card and both prompts' token counts; the engine
/// obeys the answer rather than re-deriving it, so the progress line and the
/// render cannot disagree. Without a readable VRAM total — CPU, Metal, a build
/// without NVML — the answer is the historical two forwards.
fn resolve_cfg_batching(
    flux2_cfg: &super::transformer::Flux2Config,
    positive_txt: &candle_core::Tensor,
    negative_txt: &candle_core::Tensor,
    img_tokens: usize,
    transformer_bytes: u64,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> super::transformer::Flux2CfgBatching {
    use super::transformer::Flux2CfgBatching;
    let ordinal = match device.location() {
        candle_core::DeviceLocation::Cuda { gpu_id } => Some(gpu_id),
        _ => None,
    };
    let Some(total) = ordinal.and_then(crate::device::total_vram_bytes) else {
        return Flux2CfgBatching::Sequential;
    };
    let positive_tokens = positive_txt.dim(1).unwrap_or(0);
    let negative_tokens = negative_txt.dim(1).unwrap_or(usize::MAX);
    let head_dim = flux2_cfg
        .hidden_size
        .checked_div(flux2_cfg.num_heads)
        .unwrap_or(0);
    let backend = crate::attention::effective_backend_under(
        crate::attention::AttentionPolicy::FastStill,
        device,
        dtype,
        head_dim,
    );
    let activation = super::transformer::flux2_activation_bytes_for(
        flux2_cfg,
        positive_tokens + img_tokens,
        2,
        dtype,
        backend,
    );
    super::transformer::flux2_cfg_batching_for(
        positive_tokens,
        negative_tokens,
        transformer_bytes,
        activation,
        total,
    )
}

/// Flux.2 Klein inference engine (4B and 9B variants) backed by candle.
pub struct Flux2Engine {
    base: EngineBase<LoadedFlux2>,
    /// Header-derived architecture is immutable for an engine. Cache it so a
    /// sharded Dev checkpoint's large safetensors header is parsed once, not
    /// repeatedly throughout every request.
    resolved_config: OnceLock<Flux2Config>,
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
    /// FLUX.2 [dev]'s Mistral3 conditioner, outliving the request that built
    /// it.
    ///
    /// The shell itself is cheap — paths, a tokenizer, the resolved key
    /// namespace — and the encoder holds NO device weights between requests
    /// either way, because `encode` builds and drops one decoder layer at a
    /// time. What this slot exists for is the HOST park: when
    /// `decide_text_encoder_residency` says the machine can afford it, the
    /// prefix stays in RAM and the next encode is a host-to-device copy per
    /// layer instead of a page fault, a dtype conversion and a copy.
    dev_text_encoder: Option<encoders::mistral3::Mistral3Encoder>,
    /// A transformer the SEQUENTIAL generate path kept on the card, when the
    /// residency budget allowed it.
    ///
    /// The sequential path exists because some phase of the render cannot
    /// co-reside with the transformer — the VAE encode of a source image, the
    /// 35 GB Mistral3 prefix, a LoRA merge. None of those is a reason to give
    /// the weights back to the DISK: dropping a 33 GB Q8 dev transformer and
    /// re-reading it measured 34 s per render on an idle 46 GB card. The slot
    /// lives on the engine rather than inside `base.loaded` because the
    /// sequential path deliberately never populates that — it is the state
    /// that says "nothing is eagerly resident" — and because the engine is
    /// what the model cache owns, so `unload()` is what releases this.
    retained_transformer: Option<RetainedFlux2Transformer>,
}

/// A GPU-resident FLUX.2 transformer kept between sequential renders, with
/// everything that has to match before it may be reused.
///
/// Reuse is refused on ANY mismatch rather than repaired. A transformer is
/// the render, so serving one request's weights to another's settings is the
/// one failure mode that produces a plausible wrong picture instead of an
/// error — the lesson `QwenImageEngine::active_lora_fingerprint` records.
pub(crate) struct RetainedFlux2Transformer {
    transformer: super::transformer::Flux2TransformerWrapper,
    /// Device bytes these weights occupy.
    ///
    /// Carried on the slot rather than re-derived, because the cache asks for
    /// it through `&self` long after the paths that knew it are out of scope —
    /// and because the figure must be the one the residency budget was
    /// decided against, not a fresh `stat` of a file that may have moved.
    device_bytes: u64,
    /// The GPU this was built on. A multi-GPU host leases whichever device is
    /// free, and candle tensors are bound to their ordinal.
    ordinal: usize,
    /// The working dtype the linears were materialized at.
    dtype: DType,
    /// `(path hash, scale bits)` per adapter, in request order — the FLUX.1
    /// `LoraFingerprint` keying, order-sensitive because the merge is.
    lora_fingerprint: Vec<(u64, u64)>,
    /// The resolved architecture. An engine resolves its config once, so this
    /// cannot normally move; it is here so that a checkpoint swapped under a
    /// live engine cannot be rendered with the previous one's geometry.
    config_hash: u64,
}

impl RetainedFlux2Transformer {
    fn matches(
        &self,
        ordinal: usize,
        dtype: DType,
        loras: &[(u64, u64)],
        config_hash: u64,
    ) -> bool {
        self.ordinal == ordinal
            && self.dtype == dtype
            && self.config_hash == config_hash
            && self.lora_fingerprint == loras
    }
}

/// `(path hash, scale bits)` per adapter, in request order.
///
/// Scales are compared by BITS, not by value: two `f64`s that differ in the
/// last bit merge different weights, and `to_bits` is also what the FLUX.1
/// fingerprint compares.
fn flux2_lora_fingerprint(loras: &[LoraWeight]) -> Vec<(u64, u64)> {
    loras
        .iter()
        .map(|weight| {
            (
                super::lora::lora_path_hash(&weight.path),
                weight.scale.to_bits(),
            )
        })
        .collect()
}

/// On-disk bytes of the transformer, sharded or not.
///
/// This is the resident figure for both arms: a GGUF stays quantized on the
/// card, and a BF16 safetensors is materialized one for one.
fn xformer_component_bytes(paths: &mold_core::ModelPaths) -> u64 {
    let files: &[std::path::PathBuf] = if paths.transformer_shards.is_empty() {
        std::slice::from_ref(&paths.transformer)
    } else {
        paths.transformer_shards.as_slice()
    };
    files
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok().map(|metadata| metadata.len()))
        .sum()
}

/// A hash of the resolved architecture, for the retained slot's guard.
fn flux2_config_hash(cfg: &Flux2Config) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    // `Flux2Config` is plain data with no `Hash`, and adding one would put a
    // derive on a public type for a private guard. Its `Debug` form names
    // every field, which is exactly the property this needs.
    format!("{cfg:?}").hash(&mut hasher);
    hasher.finish()
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

/// The activation dtype a GGUF Flux.2 transformer is built and run at.
///
/// The GGUF path used to pin F32 on the premise that candle's quantized
/// matmul is f32-only. It is not — `fast_mmq::try_fwd` takes BF16/F16/F32 and
/// returns what it was fed (`candle-core/src/quantized/fast_mmq.rs:218-221`,
/// `:349-358`) — but the permission is still narrow, and
/// `crate::quantized_linear::gguf_activation_dtype` is the one place every
/// GGUF family asks. `MOLD_WAN_FORCE_DMMV=1` withdraws it, because the
/// fallback it selects reads activations as f32.
fn gguf_activation_dtype(device: &Device, requested: DType) -> DType {
    crate::quantized_linear::gguf_activation_dtype(
        crate::quantized_linear::LinearDevice::of(device),
        requested,
        crate::quantized_dmmv::force_dmmv_enabled(),
    )
}

fn validate_dev_lora_runtime(config: &Flux2Config, has_lora: bool) -> Result<()> {
    if config.hidden_size == 6144 && has_lora {
        bail!("FLUX.2 [dev] LoRA loading is not implemented")
    }
    Ok(())
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
            resolved_config: OnceLock::new(),
            qwen3_variant,
            offload,
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            pending_placement: None,
            pending_loras: Vec::new(),
            shared_pool,
            dev_text_encoder: None,
            retained_transformer: None,
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
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
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
            resolved_config: OnceLock::new(),
            qwen3_variant,
            offload,
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            pending_placement: None,
            pending_loras: Vec::new(),
            shared_pool,
            dev_text_encoder: None,
            retained_transformer: None,
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
    fn resolve_config(&self) -> Result<Flux2Config> {
        if let Some(config) = self.resolved_config.get() {
            return Ok(config.clone());
        }
        if let Some(cfg) = self.detect_config_from_checkpoint() {
            let _ = self.resolved_config.set(cfg.clone());
            return Ok(cfg);
        }
        let name = self.base.model_name.to_lowercase();
        let config = if name.contains("flux2-dev") || name.contains("flux.2-dev") {
            Flux2Config::dev()
        } else if name.contains("9b") {
            Flux2Config::klein_9b()
        } else if name.contains("klein") || self.is_gguf_transformer() {
            // Opaque catalog IDs are common for Klein GGUF checkpoints. GGUF
            // metadata is not available through the safetensors header probe,
            // so retain the established Klein-4B default for this format.
            Flux2Config::klein()
        } else {
            return Err(anyhow::anyhow!(
                "unsupported FLUX.2 architecture for model '{}': checkpoint metadata did not identify a known 3072, 4096, or 6144-wide transformer",
                self.base.model_name
            ));
        };
        let _ = self.resolved_config.set(config.clone());
        Ok(config)
    }

    /// Header-peek the transformer file (if it's a single `.safetensors`)
    /// and pick the config matching its `hidden_size`. Returns `None` for
    /// sharded loads or when no `img_in.weight` marker is present.
    fn detect_config_from_checkpoint(&self) -> Option<Flux2Config> {
        let path = &self.base.paths.transformer;
        let is_safetensors = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
        if !is_safetensors {
            return None;
        }
        match super::single_file::detect_hidden_size(path) {
            Ok(Some(6144)) => Some(Flux2Config::dev()),
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
        self.resolve_config()
            .is_ok_and(|config| config.hidden_size == 4096)
    }

    fn is_dev(&self) -> bool {
        self.resolve_config()
            .is_ok_and(|config| config.hidden_size == 6144)
    }

    /// The ordered reference images this request conditions on, if any.
    ///
    /// Tier-agnostic on purpose. FLUX.2's reference protocol is a property of
    /// the ARCHITECTURE, not of one checkpoint: diffusers' Klein pipeline
    /// takes `image: list | PIL | None` and prepares its reference ids in a
    /// block marked "Copied from" the [dev] pipeline
    /// (`pipeline_flux2_klein.py:318-366,616,765-782` vs
    /// `pipeline_flux2.py:406`), BFL's first-party `flux2/src/flux2/sampling.py`
    /// numbers references from one `scale = 10` (`:53`, `default_prep` at
    /// `:226`) for every variant, and ComfyUI sets `ref_index_scale = 10.0`
    /// for EVERY flux2 checkpoint (`comfy/model_detection.py:242-256`).
    /// Gating this on `is_dev()` is what kept Klein — the tier people actually
    /// run locally — on plain img2img.
    ///
    /// An empty vector is not a reference request; it must leave a plain
    /// text-to-image render byte-identical.
    fn reference_images(req: &GenerateRequest) -> Option<&[Vec<u8>]> {
        req.edit_images
            .as_deref()
            .filter(|images| !images.is_empty())
    }

    /// The negative prompt this render's unconditional branch encodes, or
    /// The transformer checkpoint's size on disk, across shards.
    ///
    /// Stands in for its resident size because it is measured per tier: a Q8
    /// GGUF is charged its quantized bytes, not what the same parameters would
    /// cost in the working dtype.
    fn transformer_file_bytes(&self) -> u64 {
        let paths = if self.base.paths.transformer_shards.is_empty() {
            std::slice::from_ref(&self.base.paths.transformer)
        } else {
            self.base.paths.transformer_shards.as_slice()
        };
        paths
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok().map(|metadata| metadata.len()))
            .sum()
    }

    /// `None` when the render runs one forward per step.
    ///
    /// `diffusers`' `pipeline_flux2_klein.py:593` is the condition —
    /// `guidance_scale > 1 and not self.config.is_distilled` — and `:748` is
    /// the default: an undistilled base render with no negative prompt uses
    /// `""`, not "no branch". Distilled Klein and guidance-embedded Dev never
    /// reach here, so their loop is unchanged.
    fn cfg_branch_prompt(&self, req: &GenerateRequest) -> Option<String> {
        if !mold_core::validation::is_flux2_base_model(&self.base.model_name) {
            return None;
        }
        if req.guidance <= 1.0 {
            return None;
        }
        Some(req.negative_prompt.clone().unwrap_or_default())
    }

    /// The prompts this render must encode: the positive one, plus the
    /// unconditional branch's when there is one.
    fn prompts_to_encode<'a>(
        &self,
        req: &'a GenerateRequest,
        cfg_prompt: Option<&'a str>,
    ) -> Vec<&'a str> {
        match cfg_prompt {
            Some(negative) => vec![req.prompt.as_str(), negative],
            None => vec![req.prompt.as_str()],
        }
    }

    /// Every prompt's conditioning, or `None` if any of them still needs the
    /// encoder. Checked as a set so a two-prompt CFG render loads the encoder
    /// once for whichever half missed.
    fn restore_cached_prompts(
        prompt_cache: &Mutex<LruCache<String, CachedTensor>>,
        prompts: &[&str],
        device: &Device,
        dtype: DType,
    ) -> Result<Option<Vec<Tensor>>> {
        let mut hits = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            match restore_cached_tensor(
                prompt_cache,
                &flux2_prompt_cache_key(prompt),
                device,
                dtype,
            )? {
                Some(tensor) => hits.push(tensor),
                None => return Ok(None),
            }
        }
        Ok(Some(hits))
    }

    fn block_offload_enabled(&self) -> bool {
        self.offload
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
            .map_err(|e| anyhow::anyhow!("failed to load FLUX.2 text tokenizer: {e}"))
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
        // `is_dev()` is here for the ENCODER PHASE, not for the transformer's
        // size. FLUX.2 [dev] conditions on a streamed Mistral3 whose prefix is
        // ~35 GB if it were resident at once, and the eager path holds the
        // transformer across prompt encoding — the two do not co-reside on any
        // card mold ships against. The transformer itself is a residency
        // question now (`retained_transformer`) and the sequential path keeps
        // it across renders when the budget allows, so taking this path no
        // longer implies a reload.
        self.is_dev()
            || self.base.load_strategy == LoadStrategy::Sequential
            || self.offload
            || !self.pending_loras.is_empty()
            || req.source_image.is_some()
            // References are VAE-encoded in a phase of their own and the VAE
            // is dropped before the transformer loads — the same reason a
            // source image forces this path. An eager plan keeps both
            // resident, which is what OOMs Klein-9B BF16 on a 24 GB card.
            // Routing references through the eager path (VAE already resident,
            // saves a transformer reload on 4B/GGUF) is filed as a follow-up.
            || Self::reference_images(req).is_some()
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
        // This engine-level guard is load-bearing. Validation may only know an
        // opaque catalog ID, while the checkpoint header still identifies a
        // 6144-wide Dev transformer.
        validate_dev_lora_runtime(cfg, has_lora)?;
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
                            cfg,
                            gguf_vb,
                            device,
                            gguf_activation_dtype(device, gpu_dtype),
                        )?,
                    ),
                    "Loading Flux.2 transformer (GPU, GGUF + LoRA)",
                ));
            }
            // Weights stay quantized in VRAM via QMatMul — no dequantization at load
            // time. A Q4 Klein-9B uses ~6GB VRAM instead of ~18GB with full dequant.
            let gguf_vb = crate::weight_loader::load_gguf_var_builder(
                &self.base.paths.transformer,
                device,
                "Flux.2 transformer (GGUF)",
                &self.base.progress,
            )?;
            Ok((
                Flux2TransformerWrapper::Quantized(
                    super::quantized_transformer::QuantizedFlux2Transformer::new(
                        cfg,
                        gguf_vb,
                        device,
                        gguf_activation_dtype(device, gpu_dtype),
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
            // An FP8-scaled checkpoint stores `weight / weight_scale` and the
            // scale beside it, and only `Flux2Linear`'s FP8 arm applies it.
            // A LoRA merge widens the weight it patches to the working dtype,
            // so a patched layer would reach that constructor as BF16, take
            // the arm with no scale, and come out ~500x hot while every
            // untouched layer stayed correct. Refuse, exactly as Wan's
            // fp8-scaled tier does, rather than render the mixture.
            if backend.is_fp8_scaled() && has_lora {
                anyhow::bail!(
                    "{} is fp8-scaled, and mold merges LoRAs into bf16 and GGUF checkpoints \
                     only — an fp8 merge would drop the per-tensor dequantization scale on \
                     every layer the adapter touches. Use a bf16 or GGUF tier for adapters.",
                    self.base.paths.transformer.display()
                );
            }
            let backend: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(backend);
            if self.block_offload_enabled() && !has_lora && !is_nvfp4 {
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
            } else if self.block_offload_enabled() {
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
            let cfg = self.resolve_config()?;
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

    /// Whether a resident transformer has to go before the text encoder runs.
    ///
    /// This replaces a predicate that asked only whether the transformer was
    /// ALREADY absent (`Eager && !loaded`) and used the answer for a log line.
    /// Now that the transformer survives a render, the real question is an
    /// arithmetic one: the encoder's peak and the retained weights are both on
    /// the card at the same moment, and on FLUX.2 [dev] that is a ~35 GB
    /// Mistral3 prefix beside a 33 GB Q8 transformer — 68 GB on a 46 GB card.
    /// Klein's Qwen3 is a few GB and normally co-resides fine.
    ///
    /// `retained_bytes` is zero when nothing is resident, which makes the
    /// answer false: there is nothing to drop.
    ///
    /// Everything else FAILS CLOSED, for the reason the three other probe
    /// sites do. A reading of `Measured(0)` is a card with nothing free — the
    /// most pressured answer there is — and `Unmeasurable` is a probe that
    /// failed on a device that has VRAM. Both used to return "keep", on the
    /// very path whose own comment describes a 35 GB prefix landing beside a
    /// 33 GB transformer on a 46 GB card.
    fn encoder_needs_transformer_dropped(
        retained_bytes: u64,
        encoder_peak_bytes: u64,
        usable_free: crate::device::UsableFreeVram,
    ) -> bool {
        if retained_bytes == 0 {
            return false;
        }
        match usable_free {
            // A CPU render's encoder and transformer share host memory, where
            // this drop frees nothing the encoder can use.
            crate::device::UsableFreeVram::NotApplicable => false,
            crate::device::UsableFreeVram::Unmeasurable => true,
            crate::device::UsableFreeVram::Measured(free) => {
                retained_bytes.saturating_add(encoder_peak_bytes) > free
            }
        }
    }

    /// Whether FLUX.2 [dev]'s Mistral3 prefix should live in host RAM.
    ///
    /// The engine asks the SAME pure function the planner asks
    /// (`text_encoder_residency::decide_text_encoder_residency`), with the
    /// host's own live numbers, so a machine that the planner charged for a
    /// park actually takes one and a machine it did not is not surprised by
    /// 35 GB it never budgeted.
    fn decide_mistral_prefix_residency(
        encoder_device: &Device,
        encoder_dtype: DType,
        transformer_bytes: u64,
        already_parked_bytes: u64,
    ) -> super::text_encoder_residency::TextEncoderResidency {
        use super::text_encoder_residency as residency;
        let device = if encoder_device.is_metal() {
            residency::TextEncoderDevice::Metal
        } else if encoder_device.is_cuda() {
            residency::TextEncoderDevice::Cuda
        } else {
            residency::TextEncoderDevice::Cpu
        };
        residency::decide_text_encoder_residency(&residency::TextEncoderResidencyInputs {
            encoder_bytes: residency::mistral3_prefix_bytes(encoder_dtype),
            transformer_bytes,
            // An unmeasurable host reads as zero, which the decision answers
            // with `StreamFromMmap` — today's behaviour.
            host_total_bytes: crate::flux::pinned::total_system_ram_bytes().unwrap_or(0),
            host_available_bytes: crate::device::available_host_ram_bytes().unwrap_or(0),
            pinned_cap_bytes: crate::flux::pinned::pinned_cap_bytes(),
            keep_te_ram: crate::device::keep_te_ram_mode(),
            device,
            already_parked_bytes,
        })
    }

    /// Peak device bytes this checkpoint's conditioner holds while it runs.
    ///
    /// [dev] streams its Mistral3 prefix, so the figure is the streamed peak
    /// `flux2::text_encoder_residency` is the authority for — never the 36 GB
    /// of shards, which are a reclaimable mapping. Klein materializes its
    /// Qwen3, so there the file length IS the residency.
    fn text_encoder_peak_bytes(&self, gpu_dtype: DType) -> u64 {
        if self.is_dev() {
            return super::text_encoder_residency::mistral3_streamed_device_peak_bytes(
                gpu_dtype,
                super::text_encoder_residency::MISTRAL3_DEFAULT_LOOKAHEAD,
            );
        }
        // The files the encoder was ACTUALLY built from when one is loaded —
        // a resolved Q8 GGUF variant is a tenth of the BF16 shards the
        // manifest lists, and pricing the wrong one would drop a transformer
        // that had room. The manifest list is the fallback for the cold case,
        // where over-estimating only means taking today's drop.
        let paths = self
            .base
            .loaded
            .as_ref()
            .map(|loaded| loaded.text_encoder.encoder_paths().to_vec())
            .unwrap_or_else(|| self.text_encoder_paths());
        paths
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok().map(|metadata| metadata.len()))
            .sum()
    }

    /// The residency budget for this render, shared by both generate paths.
    ///
    /// `transformer_bytes` is the caller's, because the two paths know it from
    /// different places: the sequential path has already summed the checkpoint
    /// files to preflight against them, and the eager path reads the resolved
    /// path off `LoadedFlux2`.
    fn still_transformer_budget(
        &self,
        req: &GenerateRequest,
        gpu_dtype: DType,
        vae_dtype: DType,
        cfg: &Flux2Config,
        transformer_bytes: u64,
    ) -> crate::device::StillTransformerBudget {
        crate::device::StillTransformerBudget {
            transformer_bytes,
            activation_bytes: crate::device::flux_activation_budget_bytes_for(
                req.width,
                req.height,
                1,
                crate::device::dtype_bytes(gpu_dtype),
                crate::device::ActivationFamily::Flux2Dit,
                cfg.num_heads as u64,
                crate::device::flux_effective_attention_backend(),
            ),
            vae_decode_peak_bytes: crate::device::flux_vae_decode_peak_bytes(
                req.width,
                req.height,
                crate::device::dtype_bytes(vae_dtype),
            ),
            runtime_headroom_bytes: crate::device::STILL_RESIDENCY_RUNTIME_HEADROOM_BYTES,
        }
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
        // Extract hidden states from layers 9, 18, 27 and stack to (B, seq, 7680).
        //
        // Every Klein tier — 4B, 9B and the undistilled base — conditions on a
        // FIXED 512 rows: the prompt is truncated and right-padded, the pad
        // keys are masked out of the language model, and all 512 rows reach
        // the transformer (BFL `flux2/text_encoder.py:28,397-419`).
        let (stacked, _token_count) = encoder.encode_with_layers(
            prompt,
            target_device,
            target_dtype,
            &Self::QWEN3_HIDDEN_LAYERS,
            Some(encoders::qwen3::FLUX2_KLEIN_MAX_LENGTH),
        )?;
        Ok(stacked)
    }

    /// Encode one prompt through the cache.
    ///
    /// `phase` is `true` only for the render's PRIMARY prompt. A true-CFG
    /// render encodes two, and `ProgressPhase::PromptEncode` names a phase of
    /// the render rather than a call — `phase_done` is a plain emit, so
    /// reporting it per prompt would publish the phase twice on one stream.
    /// The unconditional branch still reports its own stage line.
    fn encode_prompt_cached(
        progress: &ProgressReporter,
        prompt_cache: &Mutex<LruCache<String, CachedTensor>>,
        encoder: &mut encoders::qwen3::Qwen3Encoder,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
        phase: bool,
    ) -> Result<Tensor> {
        let label = if phase {
            "Encoding prompt (Qwen3)"
        } else {
            "Encoding negative prompt (Qwen3)"
        };
        let cache_key = flux2_prompt_cache_key(prompt);
        let (txt_emb, cache_hit) = get_or_insert_cached_tensor(
            prompt_cache,
            cache_key,
            target_device,
            target_dtype,
            || {
                progress.stage_start(label);
                let encode_start = Instant::now();
                let txt_emb = Self::encode_and_stack(encoder, prompt, target_device, target_dtype)?;
                if phase {
                    progress.phase_done(
                        crate::ProgressPhase::PromptEncode,
                        label,
                        encode_start.elapsed(),
                    );
                } else {
                    progress.stage_done(label, encode_start.elapsed());
                }
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

        // Sequential mode and full Dev defer loading to
        // generate_sequential(). Dev's 64 GB transformer must never enter the
        // eager path even if a stale caller supplied an eager strategy.
        if self.base.load_strategy == LoadStrategy::Sequential || self.is_dev() {
            return Ok(());
        }

        // An eager load populates `base.loaded` with a transformer of its own,
        // so the sequential path's retained slot must go first — holding both
        // is the doubled peak each path exists to bound. `generate_inner`
        // already clears it before taking the eager branch; this covers the
        // admin and direct-caller routes that reach `load()` on their own.
        self.retained_transformer = None;

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
        let flux2_cfg = self.resolve_config()?;
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
            self.block_offload_enabled(),
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

        if self.is_dev() && (req.source_image.is_some() || req.mask_image.is_some()) {
            bail!(
                "FLUX.2 [dev] uses ordered edit_images references; source_image, strength, and mask_image img2img controls are unsupported"
            );
        }

        // Klein renders from a source image OR from references, never both in
        // one pass: upstream has no pipeline that takes the two together
        // (`pipeline_flux2_klein_inpaint.py` is a separate class with its own
        // `image_reference` argument). Admission already refuses the pair
        // through the profile's `Exclusive` source relation; this is the
        // engine's own tripwire so a private or future caller that skips that
        // door fails loudly instead of silently dropping one of them.
        if req.source_image.is_some() && Self::reference_images(req).is_some() {
            bail!(
                "FLUX.2 renders from a source image or from reference images, not both in one pass"
            );
        }

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting sequential Flux.2 generation"
        );

        self.base
            .progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: text encoding ---
        // Check prompt cache first — skip encoder load entirely on cache hit.
        // This saves ~1-5s per batch image (encoder load + VRAM allocation).
        // A true-CFG render encodes two prompts, so both are asked for as a
        // set: one encoder load serves whichever half missed.
        let cfg_prompt = self.cfg_branch_prompt(req);
        let prompts = self.prompts_to_encode(req, cfg_prompt.as_deref());
        let embeddings = if let Some(hits) =
            Self::restore_cached_prompts(&self.prompt_cache, &prompts, &device, gpu_dtype)?
        {
            // A cache hit runs no encoder at all, so a retained transformer
            // has nothing to make room for — and this is exactly the case the
            // retention exists for: a repeated prompt, or the second and later
            // members of a batch, render with no reload and no encode.
            self.base.progress.cache_hit("prompt conditioning");
            hits
        } else {
            // A cache MISS streams the conditioner beside whatever is
            // resident. Ask the budget whether the two fit; on FLUX.2 [dev]
            // they do not, and the slot goes before the encoder allocates
            // rather than after it fails.
            // The slot's OWN recorded bytes, not a fresh `stat` of the
            // checkpoint: that is the figure the residency budget was decided
            // against, and the file may have moved since.
            let retained_bytes = self
                .retained_transformer
                .as_ref()
                .map_or(0, |retained| retained.device_bytes);
            if retained_bytes > 0 {
                let usable_free = crate::device::usable_free_for_residency(
                    &device,
                    self.base.gpu_ordinal,
                    retained_bytes,
                );
                let encoder_peak = self.text_encoder_peak_bytes(gpu_dtype);
                if Self::encoder_needs_transformer_dropped(
                    retained_bytes,
                    encoder_peak,
                    usable_free,
                ) {
                    self.retained_transformer = None;
                    tracing::info!(
                        retained_mb = retained_bytes / 1024 / 1024,
                        encoder_peak_mb = encoder_peak / 1024 / 1024,
                        "released the retained Flux.2 transformer so the text encoder can stream"
                    );
                }
            }
            if self.is_dev() {
                if self
                    .qwen3_variant
                    .as_deref()
                    .is_some_and(|variant| !matches!(variant, "auto" | "bf16"))
                {
                    bail!(
                        "FLUX.2 [dev] uses its checkpoint-native Mistral3 BF16 encoder; Qwen3 variant overrides are unsupported"
                    );
                }
                let encoder_paths = self.text_encoder_paths();
                // Mistral replaces Qwen for Dev, but it occupies the same
                // frozen text-encoder placement slot in the wire contract.
                let mistral_ref = effective_device_ref(
                    self.pending_placement.as_ref(),
                    |advanced| advanced.qwen.clone(),
                    true,
                );
                let encoder_device =
                    crate::device::resolve_device(Some(mistral_ref), || Ok(device.clone()))?;
                let encoder_dtype = if encoder_device.is_cpu() {
                    DType::F32
                } else {
                    gpu_dtype
                };
                let activation_budget = crate::device::activation_bytes(
                    512,
                    1,
                    1,
                    crate::device::dtype_bytes(encoder_dtype),
                    crate::device::ActivationFamily::SmallTransformer,
                );
                preflight_memory_check(
                    "FLUX.2 [dev] streamed Mistral3 encoder",
                    encoders::mistral3::streamed_peak_weight_bytes(encoder_dtype),
                    activation_budget,
                )?;

                // The encoder may already be here, holding its prefix in host
                // RAM from the previous request. `dev_text_encoder` is what
                // makes that possible: the shell is cheap (paths, tokenizer,
                // the resolved namespace) and what it OWNS is the park.
                // A shell built for a different placement is not reusable:
                // a prefix parked at BF16 on the GPU is not the checkpoint a
                // CPU F32 encode wants.
                if self.dev_text_encoder.as_ref().is_some_and(|encoder| {
                    !encoder.matches_placement(&encoder_device, encoder_dtype)
                }) {
                    self.dev_text_encoder = None;
                }
                if self.dev_text_encoder.is_none() {
                    let text_tokenizer = self.load_text_tokenizer(&text_tokenizer_path)?;
                    self.dev_text_encoder = Some(encoders::mistral3::Mistral3Encoder::load(
                        &encoder_paths,
                        text_tokenizer,
                        &encoder_device,
                        encoder_dtype,
                    )?);
                }
                let park = Self::decide_mistral_prefix_residency(
                    &encoder_device,
                    encoder_dtype,
                    xformer_component_bytes(&self.base.paths),
                    self.dev_text_encoder
                        .as_ref()
                        .map_or(0, |encoder| encoder.parked_bytes()),
                );
                let encoder = self
                    .dev_text_encoder
                    .as_mut()
                    .expect("just installed above");
                match park {
                    super::text_encoder_residency::TextEncoderResidency::HostParked { pinned } => {
                        if !encoder.is_parked() {
                            let park_label = "Parking Mistral3 prefix in host RAM";
                            self.base.progress.stage_start(park_label);
                            let park_start = Instant::now();
                            encoder.park_prefix(pinned)?;
                            self.base
                                .progress
                                .stage_done(park_label, park_start.elapsed());
                            tracing::info!(
                                pinned,
                                host_gb = encoder.parked_bytes() as f64 / 1e9,
                                "Mistral3 prefix parked in host RAM"
                            );
                        }
                    }
                    super::text_encoder_residency::TextEncoderResidency::StreamFromMmap => {
                        if encoder.is_parked() {
                            encoder.unpark();
                            tracing::info!(
                                "released the parked Mistral3 prefix: the host budget no longer \
                                 allows it"
                            );
                        }
                    }
                }
                let encoder = &*encoder;
                // Name the device and the streamed peak in the stage label:
                // the whole point of the planner's streaming charge is that
                // this phase runs on the GPU in bf16 at ~3.6 GB rather than on
                // the CPU in F32, and a log line that says neither cannot tell
                // the two apart after the fact.
                let encoder_label =
                    format!(
                    "Encoding prompt (streamed Mistral3, {device}, {peak:.1} GB peak, {dtype:?})",
                    device = if encoder_device.is_cpu() { "CPU" } else { "GPU" },
                    peak = super::text_encoder_residency::mistral3_streamed_device_peak_bytes(
                        encoder_dtype,
                        super::text_encoder_residency::MISTRAL3_DEFAULT_LOOKAHEAD,
                    ) as f64
                        / 1e9,
                    dtype = encoder_dtype,
                );
                self.base.progress.stage_start(&encoder_label);
                let encode_start = Instant::now();
                // Dev is guidance-distilled, so `prompts` is always the single
                // positive prompt here — the loop is over one element.
                let mut encoded = Vec::with_capacity(prompts.len());
                for prompt in &prompts {
                    let (txt_emb, _) =
                        encoder.encode(prompt, &device, gpu_dtype, &self.base.progress)?;
                    let cached = CachedTensor::from_tensor(&txt_emb)?;
                    self.prompt_cache
                        .lock()
                        .unwrap()
                        .insert(flux2_prompt_cache_key(prompt), cached);
                    encoded.push(txt_emb);
                }
                self.base.progress.phase_done(
                    crate::ProgressPhase::PromptEncode,
                    &encoder_label,
                    encode_start.elapsed(),
                );
                // The DEVICE side of the encoder is released by `encode`
                // itself — every layer is built and dropped inside the stream
                // loop — so there is nothing to free here beyond the sync. The
                // shell (and its host park, when it has one) stays.
                encoder_device.synchronize()?;
                encoded
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

                let mut encoded = Vec::with_capacity(prompts.len());
                for (index, prompt) in prompts.iter().enumerate() {
                    encoded.push(Self::encode_prompt_cached(
                        &self.base.progress,
                        &self.prompt_cache,
                        &mut text_encoder,
                        prompt,
                        &device,
                        gpu_dtype,
                        index == 0,
                    )?);
                }

                // Drop text encoder to free memory
                drop(text_encoder);
                self.base.progress.info("Freed Qwen3 encoder");
                tracing::info!("Qwen3 encoder dropped (sequential mode)");

                encoded
            }
        };
        let mut embeddings = embeddings.into_iter();
        let txt_emb = embeddings
            .next()
            .ok_or_else(|| anyhow::anyhow!("prompt conditioning missing"))?;
        let neg_emb = embeddings.next();

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

        // FLUX.2 conditions on independently VAE-encoded reference images
        // appended after the noisy target tokens. References retain their
        // order through time coordinates 10, 20, ... and remain fixed
        // throughout denoising. Every tier speaks this protocol — see
        // `Self::reference_images` for the upstream citations.
        let reference_tokens = if let Some(references) = Self::reference_images(req) {
            let max_pixels = if references.len() == 1 {
                mold_core::validation::FLUX2_SINGLE_REFERENCE_MAX_PIXELS
            } else {
                mold_core::validation::FLUX2_MULTI_REFERENCE_MAX_PIXELS
            };
            let (vae, vae_dtype) = self.load_sequential_vae(&device, gpu_dtype)?;
            self.base
                .progress
                .stage_start("Encoding FLUX.2 reference images (VAE)");
            let encode_start = Instant::now();
            let mut latents = Vec::with_capacity(references.len());
            for image_bytes in references {
                // Decode at the VAE's OWN resolved dtype, not the
                // transformer's: `MOLD_VAE_DTYPE` can pin the autoencoder to
                // f32 on a bf16 render, and handing it a bf16 input is a
                // dtype mismatch inside `encode`. The latent is cast back to
                // the transformer's dtype before it is packed.
                let source = crate::img_utils::decode_flux2_reference_image(
                    image_bytes,
                    max_pixels,
                    &device,
                    vae_dtype,
                )?;
                let _conv = crate::conv_policy::ConvScope::for_family("flux2");
                latents.push(vae.encode(&source)?.to_dtype(gpu_dtype)?);
            }
            let (tokens, ids) = sampling::pack_reference_group(&latents)?;
            self.base.progress.phase_done(
                crate::ProgressPhase::Vae,
                "Encoding FLUX.2 reference images (VAE)",
                encode_start.elapsed(),
            );
            drop(latents);
            drop(vae);
            device.synchronize()?;
            self.base
                .progress
                .info("Freed VAE after reference encoding");
            Some((tokens, ids))
        } else {
            None
        };

        // Generate noise / encode source image for img2img. Source-image
        // requests pre-encode in a VAE-only phase, then drop VAE before the
        // transformer load. Klein-9B BF16 cannot keep transformer+VAE
        // co-resident on 24 GB cards.
        let (img, inpaint_ctx) = if let Some(ref source_bytes) = req.source_image {
            let start_t = timesteps[0];
            let (vae, vae_dtype) = self.load_sequential_vae(&device, gpu_dtype)?;

            self.base
                .progress
                .stage_start("Encoding source image (VAE)");
            let encode_start = Instant::now();
            // Same rule as the reference path: decode at the VAE's resolved
            // dtype (`MOLD_VAE_DTYPE` may differ from the transformer's) and
            // cast the latent back afterwards.
            let source_tensor = crate::img_utils::decode_source_image(
                source_bytes,
                req.width,
                req.height,
                Self::img2img_source_normalize_range(),
                &device,
                vae_dtype,
            )?;
            let encoded = {
                let _conv = crate::conv_policy::ConvScope::for_family("flux2");
                vae.encode(&source_tensor)?.to_dtype(gpu_dtype)?
            };
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
        let xformer_paths = if self.base.paths.transformer_shards.is_empty() {
            std::slice::from_ref(&self.base.paths.transformer)
        } else {
            self.base.paths.transformer_shards.as_slice()
        };
        let xformer_size = xformer_paths
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok().map(|metadata| metadata.len()))
            .sum::<u64>();
        let xformer_activation_budget = crate::device::activation_bytes(
            req.width,
            req.height,
            1,
            crate::device::dtype_bytes(gpu_dtype),
            crate::device::ActivationFamily::Flux2Dit,
        );
        // Reference tokens lengthen the ONE sequence the transformer attends
        // over, so the workspace scales with the packed length rather than
        // with the canvas. Here the references are already encoded, so the
        // ratio is EXACT — token counts, not the server's pixel estimate —
        // and it is the same `flux2_reference_scaled_activation_bytes` admission planned
        // against.
        let xformer_activation_budget = match reference_tokens.as_ref() {
            Some((tokens, _)) => crate::device::flux2_reference_scaled_activation_bytes(
                xformer_activation_budget,
                state.img.dim(1)? as u64,
                tokens.dim(1)? as u64,
            ),
            None => xformer_activation_budget,
        };
        // Block offload reserves a bounded GPU working set; the full
        // transformer remains host-mapped and is accounted by Scheduler V2's
        // host-memory ledger.
        let resident_xformer_size = if self.block_offload_enabled() {
            xformer_size.min(crate::device::STREAMING_TRANSFORMER_CAP_BYTES)
        } else {
            xformer_size
        };
        let flux2_cfg = self.resolve_config()?;
        let lora_fingerprint = flux2_lora_fingerprint(&self.pending_loras);
        let config_hash = flux2_config_hash(&flux2_cfg);
        let reuse = self.retained_transformer.as_ref().is_some_and(|retained| {
            retained.matches(
                self.base.gpu_ordinal,
                gpu_dtype,
                &lora_fingerprint,
                config_hash,
            )
        });
        // A retained transformer is already on the card, so the preflight it
        // would otherwise pay is not a question about this render — asking it
        // would charge the weights a second time against the free VRAM they
        // are already occupying.
        let transformer = if reuse {
            self.base.progress.cache_hit("Flux.2 transformer");
            tracing::info!("Flux.2 transformer reused from the previous render (no reload)");
            self.retained_transformer
                .take()
                .expect("just matched")
                .transformer
        } else {
            // A stale slot that did not match is released BEFORE the
            // replacement loads, or the two co-reside at the exact moment the
            // sequential path exists to avoid.
            self.retained_transformer = None;
            preflight_memory_check(
                "Flux.2 transformer",
                resident_xformer_size,
                xformer_activation_budget,
            )?;
            if let Some(status) = memory_status_string() {
                self.base.progress.info(&status);
            }
            let xformer_stage = Instant::now();
            let (transformer, xformer_label) =
                self.load_transformer(&flux2_cfg, gpu_dtype, &device, xformer_activation_budget)?;
            self.base
                .progress
                .stage_done(xformer_label, xformer_stage.elapsed());
            transformer
        };

        let denoise_label = format!("Denoising ({} steps)", timesteps.len().saturating_sub(1));
        self.base.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        let previewer = crate::latent_preview::LatentPreviewer::flux2(height, width);
        // Both branches denoise the SAME image tokens, so the unconditional
        // branch needs only its own text conditioning — batched to match the
        // positive one, which `state.img` has already been packed into.
        let neg_conditioning = neg_emb
            .as_ref()
            .map(|emb| sampling::text_conditioning(emb, state.img.dim(0)?, state.img.device()))
            .transpose()?;
        let cfg_batching = neg_conditioning.as_ref().map(|(neg_txt, _)| {
            resolve_cfg_batching(
                &flux2_cfg,
                &state.txt,
                neg_txt,
                state.img.dim(1).unwrap_or(0),
                self.transformer_file_bytes(),
                gpu_dtype,
                &device,
            )
        });
        let cfg_branch =
            neg_conditioning
                .as_ref()
                .map(|(txt, txt_ids)| super::transformer::Flux2CfgBranch {
                    scale: req.guidance,
                    txt,
                    txt_ids,
                    batching: cfg_batching
                        .unwrap_or(super::transformer::Flux2CfgBatching::Sequential),
                });
        if let Some(branch) = cfg_branch.as_ref() {
            self.base.progress.info(&format!(
                "Undistilled FLUX.2 base: classifier-free guidance at {:.2} ({})",
                req.guidance,
                branch.batching.progress_note()
            ));
        }
        let img = transformer.denoise(
            &state.img,
            &state.img_ids,
            reference_tokens.as_ref().map(|(tokens, ids)| (tokens, ids)),
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            req.guidance,
            &self.base.progress,
            inpaint_ctx.as_ref(),
            Some(&previewer),
            cfg_branch.as_ref(),
        )?;

        let img = sampling::unpack(&img, height, width)?;

        self.base
            .progress
            .stage_done(&denoise_label, denoise_start.elapsed());

        // The transformer either goes back to the card's free list or stays on
        // it for the next render. The VAE decode runs next either way, so this
        // is the same budget the eager path weighs — not a rule about which
        // generate path is running.
        drop(inpaint_ctx);
        let budget = self.still_transformer_budget(
            req,
            gpu_dtype,
            crate::device::resolve_vae_dtype(gpu_dtype),
            &flux2_cfg,
            xformer_size,
        );
        // Sampled with the transformer still resident, so its bytes are added
        // back: the budget is defined against the card as if nothing this
        // render loaded were on it.
        let usable_free =
            crate::device::usable_free_for_residency(&device, self.base.gpu_ordinal, xformer_size);
        let residency = crate::device::still_transformer_residency(&budget, usable_free);
        if residency.keeps() {
            self.retained_transformer = Some(RetainedFlux2Transformer {
                transformer,
                device_bytes: xformer_size,
                ordinal: self.base.gpu_ordinal,
                dtype: gpu_dtype,
                lora_fingerprint,
                config_hash,
            });
            self.base
                .progress
                .info("Kept Flux.2 transformer resident for the next render");
        } else {
            drop(transformer);
            self.base.progress.info("Freed Flux.2 transformer");
            tracing::info!(
                shortfall_mb = residency.shortfall_bytes() / 1024 / 1024,
                required_mb = budget.required_bytes() / 1024 / 1024,
                "Flux.2 transformer dropped before VAE decode: the residency budget does not fit"
            );
        }
        drop(state);
        drop(txt_emb);
        device.synchronize()?;
        tracing::info!(
            retained = residency.keeps(),
            "Flux.2 transformer settled (sequential mode), decoding VAE..."
        );

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
        // See the FLUX.1 decode: the transformer is linear throughout, so the
        // VAE is where `ConvPolicy::FastStill` has anything to decide.
        let cudnn_dispatches_before = crate::conv_policy::cudnn_dispatch_count();
        let _conv = crate::conv_policy::ConvScope::for_family("flux2");
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
        crate::conv_policy::report_vae_decode_backend("flux2", cudnn_dispatches_before);

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
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
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
        if !self.is_dev() && req.guidance != 0.0 {
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

        // Eager mode: use pre-loaded components, held in `base.loaded`. The
        // sequential path's retained slot belongs to that path alone — holding
        // both would double the peak this engine's two paths each exist to
        // bound — so it is released here.
        self.retained_transformer = None;
        if self.base.loaded.is_none() {
            self.load()?;
        }
        // Derived before the `loaded` borrow below, because the residency
        // question needs `self` and the encode loop needs `&mut loaded`.
        let transformer_bytes = xformer_component_bytes(&self.base.paths);
        let encoder_peak_bytes = self.text_encoder_peak_bytes(
            self.base
                .loaded
                .as_ref()
                .map(|loaded| loaded.dtype)
                .unwrap_or(DType::BF16),
        );
        let eager_usable_free = {
            let resident = self
                .base
                .loaded
                .as_ref()
                .is_some_and(|loaded| loaded.transformer.is_some());
            let device = self
                .base
                .loaded
                .as_ref()
                .map(|loaded| loaded.device.clone())
                .unwrap_or(Device::Cpu);
            crate::device::usable_free_for_residency(
                &device,
                self.base.gpu_ordinal,
                if resident { transformer_bytes } else { 0 },
            )
        };
        // The post-denoise residency budget, resolved here for the same
        // borrow reason: `loaded` is mutably borrowed across the whole render.
        let eager_budget = {
            let cfg = self.resolve_config()?;
            let (gpu_dtype, vae_dtype) = self
                .base
                .loaded
                .as_ref()
                .map(|loaded| (loaded.dtype, loaded.vae_dtype))
                .unwrap_or((DType::BF16, DType::BF16));
            self.still_transformer_budget(req, gpu_dtype, vae_dtype, &cfg, transformer_bytes)
        };
        let gpu_ordinal_for_budget = self.base.gpu_ordinal;

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

        // 1. Encode prompt with Qwen3 (check cache first to avoid unnecessary
        //    reload). An undistilled base render also encodes its
        //    unconditional branch here, while the encoder is up.
        let cfg_prompt = self.cfg_branch_prompt(req);
        let prompts = self.prompts_to_encode(req, cfg_prompt.as_deref());
        let embeddings = {
            let prompt_cache = &self.prompt_cache;
            let progress = &self.base.progress;
            let loaded = self
                .base
                .loaded
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
            if let Some(hits) =
                Self::restore_cached_prompts(prompt_cache, &prompts, &loaded.device, loaded.dtype)?
            {
                progress.cache_hit("prompt conditioning");
                hits
            } else {
                // Cache miss — the encoder is about to be resident. On a warm
                // engine the transformer may be too, and reloading it BEFORE
                // the encode recreates the highest peak of the whole render
                // (transformer + Qwen3), which is what OOM'd 24 GB cards on
                // back-to-back queued requests. Ask the budget rather than
                // always dropping: on a card with room, the two co-reside and
                // the next render skips a full reload.
                if loaded.transformer.is_some()
                    && Self::encoder_needs_transformer_dropped(
                        transformer_bytes,
                        encoder_peak_bytes,
                        eager_usable_free,
                    )
                {
                    loaded.transformer = None;
                    tracing::info!(
                        transformer_mb = transformer_bytes / 1024 / 1024,
                        encoder_peak_mb = encoder_peak_bytes / 1024 / 1024,
                        "dropped the resident Flux.2 transformer so the text encoder fits; it \
                         reloads after the encode"
                    );
                }

                // Restore the encoder if it was dropped or parked after a
                // previous generation.
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

                let mut encoded = Vec::with_capacity(prompts.len());
                for (index, prompt) in prompts.iter().enumerate() {
                    encoded.push(Self::encode_prompt_cached(
                        progress,
                        prompt_cache,
                        &mut loaded.text_encoder,
                        prompt,
                        &loaded.device,
                        loaded.dtype,
                        index == 0,
                    )?);
                }
                tracing::info!("Qwen3 encoding complete");

                // Free GPU VRAM for denoising. Whether the parameters move
                // to host RAM or are released is `decide_text_encoder_residency`'s
                // answer, not a flag: a host with room keeps them and the next
                // cache-miss prompt costs a copy instead of a re-read (3.9 s
                // to under 1 s on Klein). GGUF parks too now — a quantized
                // tensor round-trips byte-exact through
                // `wan::block_offload` — and Metal still does not, because
                // there the "parked" copy is in the pool the encoder already
                // runs from.
                if loaded.text_encoder.on_gpu || loaded.device.is_metal() {
                    let park_mode = super::text_encoder_residency::qwen3_park_residency(
                        &loaded.device,
                        loaded.text_encoder.encoder_paths(),
                        transformer_bytes,
                        loaded.text_encoder.parked_bytes(),
                    )
                    .parks();
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

                encoded
            }
        };
        let mut embeddings = embeddings.into_iter();
        let txt_emb = embeddings
            .next()
            .ok_or_else(|| anyhow::anyhow!("prompt conditioning missing"))?;
        let neg_emb = embeddings.next();

        self.reload_transformer_if_needed()?;

        // Read before the mutable borrow below: the CFG batching decision
        // needs both, and neither changes during the denoise.
        let eager_cfg = self.resolve_config()?;
        let eager_transformer_bytes = self.transformer_file_bytes();

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
            let encoded = {
                let _conv = crate::conv_policy::ConvScope::for_family("flux2");
                loaded.vae.encode(&source_tensor)?
            };
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
        // Both branches denoise the SAME image tokens, so the unconditional
        // branch needs only its own text conditioning — batched to match the
        // positive one, which `state.img` has already been packed into.
        let neg_conditioning = neg_emb
            .as_ref()
            .map(|emb| sampling::text_conditioning(emb, state.img.dim(0)?, state.img.device()))
            .transpose()?;
        let cfg_batching = neg_conditioning.as_ref().map(|(neg_txt, _)| {
            resolve_cfg_batching(
                &eager_cfg,
                &state.txt,
                neg_txt,
                state.img.dim(1).unwrap_or(0),
                eager_transformer_bytes,
                loaded.dtype,
                &loaded.device,
            )
        });
        let cfg_branch =
            neg_conditioning
                .as_ref()
                .map(|(txt, txt_ids)| super::transformer::Flux2CfgBranch {
                    scale: req.guidance,
                    txt,
                    txt_ids,
                    batching: cfg_batching
                        .unwrap_or(super::transformer::Flux2CfgBatching::Sequential),
                });
        if let Some(branch) = cfg_branch.as_ref() {
            progress.info(&format!(
                "Undistilled FLUX.2 base: classifier-free guidance at {:.2} ({})",
                req.guidance,
                branch.batching.progress_note()
            ));
        }
        let img = transformer.denoise(
            &state.img,
            &state.img_ids,
            None,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            req.guidance,
            progress,
            inpaint_ctx.as_ref(),
            Some(&previewer),
            cfg_branch.as_ref(),
        )?;

        // 6. Unpack latent to spatial
        let img = sampling::unpack(&img, height, width)?;
        progress.stage_done(&denoise_label, denoise_start.elapsed());
        tracing::info!("denoising complete, decoding VAE...");

        // Free denoising intermediates before VAE decode. Whether the
        // transformer goes with them is the budget's answer, not a default:
        // the decode wants a large contiguous conv2d workspace, and on a card
        // that has room for both, dropping it only buys a 34 s reload on the
        // next render.
        drop(inpaint_ctx);
        drop(state);
        drop(txt_emb);
        let free_before_vae = crate::device::free_vram_bytes(gpu_ordinal_for_budget).unwrap_or(0);
        let usable_free = crate::device::usable_free_for_residency(
            &loaded.device,
            gpu_ordinal_for_budget,
            transformer_bytes,
        );
        let residency = crate::device::still_transformer_residency(&eager_budget, usable_free);
        if residency.keeps() {
            tracing::info!(
                free_mb = free_before_vae / 1024 / 1024,
                required_mb = eager_budget.required_bytes() / 1024 / 1024,
                "Flux.2 transformer kept resident: the residency budget fits, so the next render \
                 skips the reload"
            );
        } else {
            loaded.transformer = None;
            tracing::info!(
                free_mb = free_before_vae / 1024 / 1024,
                shortfall_mb = residency.shortfall_bytes() / 1024 / 1024,
                "Flux.2 transformer dropped before VAE decode: the residency budget does not fit \
                 this card at this resolution"
            );
        }
        // Force CUDA to complete pending operations and release freed memory.
        // Without this, cuMemFree is asynchronous and the freed VRAM may not
        // be available when VAE decode allocates its conv2d intermediates.
        loaded.device.synchronize()?;

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
        let cudnn_dispatches_before = crate::conv_policy::cudnn_dispatch_count();
        let _conv = crate::conv_policy::ConvScope::for_family("flux2");
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
        crate::conv_policy::report_vae_decode_backend("flux2", cudnn_dispatches_before);

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
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
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
        // A retained transformer is GPU residency this engine owns, and the
        // model cache classifies residency from exactly this answer
        // (`ModelCache::insert` / `restore`). `EngineBase::is_loaded` already
        // answers true for a Sequential STRATEGY, but a FLUX.2 [dev] engine
        // takes the sequential generate PATH on an Eager strategy, and
        // `generate_inner` clears `base.loaded` before it — so without this
        // the cache would reclassify a 33 GB resident engine as parked and
        // zero its VRAM credit while the weights sat on the card.
        self.base.is_loaded() || self.retained_transformer.is_some()
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

    fn resident_vram_bytes(&self) -> Option<u64> {
        // Only the SEQUENTIAL path's retained slot. An eager engine's
        // transformer lives in `base.loaded` and was already measured by the
        // cache's `vram_load_delta`, so reporting it here would double it.
        let retained = self.retained_transformer.as_ref()?;
        Some(retained.device_bytes)
    }

    fn release_retained_residency(&mut self) -> u64 {
        let freed = self
            .retained_transformer
            .as_ref()
            .map_or(0, |retained| retained.device_bytes);
        self.retained_transformer = None;
        freed
    }

    fn unload(&mut self) {
        self.base.unload();
        // The retained slot is the one piece of GPU state that does NOT live
        // in `base.loaded`, so `base.unload()` cannot release it and the model
        // cache's eviction would otherwise leave ~33 GB on the card with no
        // owner that can be asked about it. The parked Mistral3 prefix is the
        // same story in HOST memory: ~35 GB the host ledger has been told is
        // spent, which only this engine can give back.
        self.retained_transformer = None;
        self.dev_text_encoder = None;
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

    #[test]
    fn flux2_dev_refuses_eager_preload_before_touching_model_files() {
        let dir = temp_test_dir("mold-flux2-dev-eager-preload");
        let mut engine = Flux2Engine::new(
            "flux2-dev:bf16".to_string(),
            flux2_model_paths(&dir, "missing-transformer.safetensors", vec![], None),
            None,
            LoadStrategy::Eager,
            0,
            false,
            None,
        );

        engine.load().unwrap();
        assert!(
            !engine.is_loaded(),
            "FLUX.2 Dev must remain sequential even under a stale eager plan"
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
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
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

    /// Reference images are VAE-encoded in a phase of their own and the VAE is
    /// dropped before the transformer loads, exactly like a source image
    /// (`load_sequential_vae` builds the shared config for every tier). An
    /// eager plan keeps both resident, which is what OOMs Klein-9B BF16 on a
    /// 24 GB card — so a Klein render carrying references takes the sequential
    /// path whatever the engine's configured strategy says, and a LoRA on top
    /// of it changes nothing.
    #[test]
    fn klein_reference_images_require_sequential_generation_even_for_eager_engines() {
        let dir = temp_test_dir("mold-flux2-reference-sequential");
        let mut engine = Flux2Engine::new(
            "flux2-klein:bf16".to_string(),
            flux2_model_paths(&dir, "transformer.safetensors", vec![], None),
            None,
            LoadStrategy::Eager,
            0,
            false,
            None,
        );
        let mut req = test_generate_request();
        assert!(!engine.uses_sequential_generate_path(&req));

        req.edit_images = Some(vec![vec![0x89, 0x50, 0x4e, 0x47]]);
        assert!(engine.uses_sequential_generate_path(&req));

        // An empty vector is not a reference request — it must not silently
        // reroute a plain text-to-image render.
        req.edit_images = Some(vec![]);
        assert!(!engine.uses_sequential_generate_path(&req));

        // References plus a LoRA: Klein keeps LoRA (only [dev]'s checkpoint
        // refuses one), and either reason alone already forces the sequential
        // path.
        req.edit_images = Some(vec![vec![0x89, 0x50, 0x4e, 0x47]]);
        engine.pending_loras = vec![LoraWeight {
            path: dir.join("style.safetensors").display().to_string(),
            scale: 0.8,
            expert: None,
        }];
        assert!(engine.uses_sequential_generate_path(&req));
    }

    /// The prompt-encode ordering rule is arithmetic, and it FAILS CLOSED.
    ///
    /// * FLUX.2 [dev] on a 46 GB card — a 33 GB Q8 transformer beside a
    ///   ~3.6 GB streamed Mistral3 peak is fine, but beside the ~35 GB prefix
    ///   a host-parked encoder would hold, it is not;
    /// * Klein on a 24 GB card — a 9.5 GB Q8 transformer and a ~8 GB Qwen3
    ///   co-reside;
    /// * nothing resident — there is nothing to drop, whatever the numbers;
    /// * an UNMEASURABLE or zero-free card drops, like the three other probe
    ///   sites. This was the fourth one still failing open: it returned
    ///   "keep" on a zero sentinel, on the very path whose own comment
    ///   describes 68 GB landing on a 46 GB card.
    #[test]
    fn the_encoder_drops_the_transformer_only_when_the_two_do_not_fit() {
        use crate::device::UsableFreeVram::{Measured, NotApplicable, Unmeasurable};
        const GB: u64 = 1_000_000_000;

        assert!(
            !Flux2Engine::encoder_needs_transformer_dropped(
                33 * GB,
                3_600_000_000,
                Measured(46 * GB)
            ),
            "a streamed Mistral3 peak co-resides with a 33 GB transformer on a 46 GB card"
        );
        assert!(
            Flux2Engine::encoder_needs_transformer_dropped(33 * GB, 35 * GB, Measured(46 * GB)),
            "a resident 35 GB prefix beside a 33 GB transformer is 68 GB on a 46 GB card"
        );
        assert!(
            !Flux2Engine::encoder_needs_transformer_dropped(
                9_500_000_000,
                8 * GB,
                Measured(24 * GB)
            ),
            "Klein's Qwen3 and its Q8 transformer fit a 24 GB card together"
        );
        assert!(
            Flux2Engine::encoder_needs_transformer_dropped(20 * GB, 8 * GB, Measured(24 * GB)),
            "a BF16 Klein-9B plus its encoder does not"
        );
        assert!(
            !Flux2Engine::encoder_needs_transformer_dropped(0, 35 * GB, Measured(24 * GB)),
            "nothing resident means nothing to drop"
        );

        // The fail-closed rows.
        assert!(
            Flux2Engine::encoder_needs_transformer_dropped(33 * GB, 35 * GB, Unmeasurable),
            "a failed accelerator probe must drop, not keep 33 GB resident"
        );
        assert!(
            Flux2Engine::encoder_needs_transformer_dropped(33 * GB, 35 * GB, Measured(0)),
            "a card measured at zero free is the most pressured reading there is"
        );
        assert!(
            !Flux2Engine::encoder_needs_transformer_dropped(33 * GB, 35 * GB, NotApplicable),
            "a CPU render's encoder and transformer share host memory; this drop frees nothing"
        );
    }

    /// The retained slot is refused on ANY mismatch. A transformer IS the
    /// render, so reusing one across a different LoRA stack, dtype, GPU, or
    /// architecture is the failure mode that produces a plausible wrong
    /// picture rather than an error.
    #[test]
    fn retained_transformer_is_reused_only_when_lora_and_dtype_match() {
        use crate::flux2::quantized_transformer::test_support::{tiny_cfg, tiny_transformer};

        let cfg = tiny_cfg(false);
        let config_hash = flux2_config_hash(&cfg);
        let loras = vec![(11u64, 22u64)];
        let retained = RetainedFlux2Transformer {
            transformer: super::super::transformer::Flux2TransformerWrapper::Quantized(
                tiny_transformer(&cfg),
            ),
            device_bytes: 33_000_000_000,
            ordinal: 1,
            dtype: DType::BF16,
            lora_fingerprint: loras.clone(),
            config_hash,
        };

        assert!(retained.matches(1, DType::BF16, &loras, config_hash));
        assert!(
            !retained.matches(0, DType::BF16, &loras, config_hash),
            "a tensor is bound to the GPU it was built on"
        );
        assert!(
            !retained.matches(1, DType::F16, &loras, config_hash),
            "the working dtype is the linears' materialized precision"
        );
        assert!(
            !retained.matches(1, DType::BF16, &[], config_hash),
            "an unadapted request must not render through a merged transformer"
        );
        assert!(
            !retained.matches(1, DType::BF16, &[(11, 23)], config_hash),
            "a changed scale is a different merge"
        );
        assert!(
            !retained.matches(1, DType::BF16, &[(11, 22), (33, 44)], config_hash),
            "a second adapter is a different merge"
        );
        assert!(
            !retained.matches(1, DType::BF16, &loras, config_hash ^ 1),
            "a different architecture is a different transformer"
        );

        // The fingerprint is order-sensitive and compares scales by BITS,
        // because the merge is both.
        let weight = |path: &str, scale: f64| LoraWeight {
            path: path.to_string(),
            scale,
            expert: None,
        };
        let forward = flux2_lora_fingerprint(&[weight("/a", 0.8), weight("/b", 0.4)]);
        let reversed = flux2_lora_fingerprint(&[weight("/b", 0.4), weight("/a", 0.8)]);
        assert_ne!(forward, reversed);
        assert_ne!(
            flux2_lora_fingerprint(&[weight("/a", 0.8)]),
            flux2_lora_fingerprint(&[weight("/a", 0.8 + f64::EPSILON)])
        );
    }

    /// `unload()` is what the model cache calls on eviction, and the retained
    /// slot is the one piece of GPU state that does not live in
    /// `base.loaded` — so `base.unload()` alone would leave the weights on the
    /// card with no owner that could be asked about them.
    #[test]
    fn unload_clears_the_retained_transformer() {
        use crate::flux2::quantized_transformer::test_support::{tiny_cfg, tiny_transformer};
        use crate::InferenceEngine;

        let dir = temp_test_dir("mold-flux2-retained-unload");
        let mut engine = Flux2Engine::new(
            "flux2-klein:bf16".to_string(),
            flux2_model_paths(&dir, "transformer.safetensors", vec![], None),
            None,
            LoadStrategy::Eager,
            0,
            false,
            None,
        );
        assert!(
            !engine.is_loaded(),
            "an Eager engine with nothing loaded is not resident"
        );

        let cfg = tiny_cfg(false);
        engine.retained_transformer = Some(RetainedFlux2Transformer {
            transformer: super::super::transformer::Flux2TransformerWrapper::Quantized(
                tiny_transformer(&cfg),
            ),
            device_bytes: 33_000_000_000,
            ordinal: 0,
            dtype: DType::F32,
            lora_fingerprint: Vec::new(),
            config_hash: flux2_config_hash(&cfg),
        });
        assert!(
            engine.is_loaded(),
            "a retained transformer is GPU residency the cache must see"
        );

        InferenceEngine::unload(&mut engine);
        assert!(engine.retained_transformer.is_none());
        assert!(!engine.is_loaded());
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

        let dev = Flux2Engine::new(
            "flux2-dev:bf16".to_string(),
            flux2_model_paths(&base_dir, "transformer.safetensors", vec![], None),
            None,
            LoadStrategy::Sequential,
            0,
            true,
            None,
        );

        let standard_cfg = standard.resolve_config().unwrap();
        let nine_b_cfg = nine_b.resolve_config().unwrap();
        let dev_cfg = dev.resolve_config().unwrap();

        assert_eq!(standard_cfg.hidden_size, 3072);
        assert_eq!(standard_cfg.context_in_dim, 7680);
        assert_eq!(standard.qwen3_size(), Qwen3Size::B4);
        assert_eq!(standard.qwen3_bf16_config().hidden_size, 2560);

        assert_eq!(nine_b_cfg.hidden_size, 4096);
        assert_eq!(nine_b_cfg.context_in_dim, 12288);
        assert_eq!(nine_b.qwen3_size(), Qwen3Size::B8);
        assert_eq!(nine_b.qwen3_bf16_config().hidden_size, 4096);

        assert_eq!(dev_cfg.hidden_size, 6144);
        assert_eq!(dev_cfg.context_in_dim, 15360);
        assert!(dev_cfg.guidance_embed);
        assert!(dev.is_dev());

        fs::remove_dir_all(base_dir).ok();
    }

    #[test]
    fn opaque_gguf_keeps_the_established_klein_four_b_config() {
        let dir = temp_test_dir("mold-flux2-opaque-gguf-config");
        let engine = Flux2Engine::new(
            "cv:2759597".to_string(),
            flux2_model_paths(&dir, "opaque.gguf", vec![], None),
            None,
            LoadStrategy::Sequential,
            0,
            false,
            None,
        );

        let config = engine.resolve_config().unwrap();
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.context_in_dim, 7680);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn detected_dev_config_rejects_lora_at_the_runtime_boundary() {
        assert!(validate_dev_lora_runtime(&Flux2Config::dev(), true)
            .unwrap_err()
            .to_string()
            .contains("LoRA loading is not implemented"));
        assert!(validate_dev_lora_runtime(&Flux2Config::klein(), true).is_ok());
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
                low_noise_transformer: None,
                low_noise_distilled_lora: None,
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
        let cfg = engine.resolve_config().unwrap();
        let txt_emb = Tensor::zeros((1, 1, cfg.context_in_dim), DType::F32, &Device::Cpu).unwrap();
        engine.prompt_cache.lock().unwrap().insert(
            flux2_prompt_cache_key("a cat"),
            CachedTensor::from_tensor(&txt_emb).unwrap(),
        );
        let req = GenerateRequest {
            mesh_workflow: None,
            offload: None,
            mesh: None,
            video_only: None,
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            reference_weight: None,
            references: None,
            strength: 1.0,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
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
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
            save_to_gallery: None,
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
                low_noise_transformer: None,
                low_noise_distilled_lora: None,
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
                low_noise_transformer: None,
                low_noise_distilled_lora: None,
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
