use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use candle_transformers::models::z_image::{
    calculate_shift, postprocess_image, AutoEncoderKL, Config, FlowMatchEulerDiscreteScheduler,
    SchedulerConfig, VaeConfig,
};
use mold_candle::quantized as quantized_var_builder;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, LoraWeight, ModelPaths};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;

use super::transformer::{MoldZImageTransformer2DModel, ZImageTransformer};
use crate::cache::{
    clear_cache, get_or_insert_cached_tensor, prompt_text_key, restore_cached_tensor, CachedTensor,
    LruCache, DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{
    check_memory_budget, effective_device_ref, fmt_gb, free_vram_bytes, memory_status_string,
    preflight_memory_check, should_use_gpu, usable_free_vram_bytes,
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
/// Eager mode loads the VAE before denoising but drops the transformer before
/// decode. Use a weight-load threshold here, not the decode workspace threshold,
/// so CUDA/Metal decode still happens on GPU after the transformer is freed.
const VAE_WEIGHT_LOAD_VRAM_THRESHOLD: u64 = 600_000_000;

/// Z-Image scheduler shift constants from the reference implementation.
const BASE_IMAGE_SEQ_LEN: usize = 256;
const MAX_IMAGE_SEQ_LEN: usize = 4096;
const ZIMAGE_SINGLE_FILE_PREFIX: &str = "model.diffusion_model.";

struct ZImageSafetensorsBackend {
    st: candle_core::safetensors::MmapedSafetensors,
}

impl ZImageSafetensorsBackend {
    fn new(st: candle_core::safetensors::MmapedSafetensors) -> Self {
        Self { st }
    }

    fn resolve_stored_name<'a>(&'a self, name: &'a str) -> Option<Cow<'a, str>> {
        if self.st.get(name).is_ok() {
            return Some(Cow::Borrowed(name));
        }
        if let Some(alias) = zimage_safetensors_alias(name) {
            if self.st.get(alias.as_ref()).is_ok() {
                return Some(alias);
            }
        }
        let prefixed = format!("{ZIMAGE_SINGLE_FILE_PREFIX}{name}");
        if self.st.get(&prefixed).is_ok() {
            return Some(Cow::Owned(prefixed));
        }
        if let Some(alias) = zimage_safetensors_alias(name) {
            let prefixed_alias = format!("{ZIMAGE_SINGLE_FILE_PREFIX}{}", alias.as_ref());
            if self.st.get(&prefixed_alias).is_ok() {
                return Some(Cow::Owned(prefixed_alias));
            }
        }
        None
    }

    fn stored_name<'a>(&'a self, name: &'a str) -> Cow<'a, str> {
        self.resolve_stored_name(name)
            .unwrap_or(Cow::Borrowed(name))
    }

    fn load_cast(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let stored_name = self.stored_name(name);
        let tensor = self.st.load(stored_name.as_ref(), dev)?;
        if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)
        } else {
            Ok(tensor)
        }
    }

    fn load_tensor(
        &self,
        name: &str,
        expected_shape: Option<&Shape>,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        // A directly-stored name always wins: official Tongyi BF16
        // checkpoints (z-image-turbo) ship SPLIT diffusers attention tensors
        // (`to_q`/`to_k`/`to_v`), while single-file/community conversions
        // fuse them into `attention.qkv`. Only translate a split request
        // into a fused slice when the split tensor itself isn't stored.
        if self.resolve_stored_name(name).is_some() {
            return self.load_cast(name, dtype, dev);
        }
        if let Some((source_name, component)) = zimage_qkv_request(name) {
            return self.load_qkv_split(&source_name, component, expected_shape, dtype, dev);
        }
        self.load_cast(name, dtype, dev)
    }

    fn load_qkv_split(
        &self,
        source_name: &str,
        component: usize,
        expected_shape: Option<&Shape>,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let qkv = self.load_cast(source_name, dtype, dev)?;
        let rows = qkv.dim(0)?;
        let split_rows = expected_shape
            .and_then(|shape| shape.dims().first().copied())
            .unwrap_or(rows / 3);
        if component >= 3 || split_rows == 0 || rows != split_rows * 3 {
            return Err(candle_core::Error::msg(format!(
                "invalid fused QKV shape for {source_name}: rows={rows}, split_rows={split_rows}"
            )));
        }
        qkv.narrow(0, component * split_rows, split_rows)?
            .contiguous()
    }
}

impl candle_nn::var_builder::SimpleBackend for ZImageSafetensorsBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _init: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load_tensor(name, Some(&shape), dtype, dev)?;
        if tensor.shape() != &shape {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: shape,
                got: tensor.shape().clone(),
            })?
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.load_tensor(name, None, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        if self.resolve_stored_name(name).is_some() {
            return true;
        }
        if let Some((source_name, _)) = zimage_qkv_request(name) {
            return self.resolve_stored_name(&source_name).is_some();
        }
        false
    }
}

enum ZImageVaeTensorSource {
    Mmap(candle_core::safetensors::MmapedSafetensors),
    Cpu(Arc<HashMap<String, Tensor>>),
}

/// Z-Image VAE accepts both diffusers keys and LDM/Civitai aliases, plus a
/// narrow `[out, in, 1, 1]` to `[out, in]` reshape. Keep both mmap and cached
/// CPU tensor sources behind this backend so those rules cannot be bypassed.
struct ZImageVaeSafetensorsBackend {
    source: ZImageVaeTensorSource,
    aliases: BTreeMap<String, String>,
}

impl ZImageVaeSafetensorsBackend {
    fn new(st: candle_core::safetensors::MmapedSafetensors) -> Self {
        let aliases = Self::aliases_from_names(st.tensors().into_iter().map(|(name, _)| name));
        Self {
            source: ZImageVaeTensorSource::Mmap(st),
            aliases,
        }
    }

    fn from_cpu_tensors(tensors: Arc<HashMap<String, Tensor>>) -> Self {
        let aliases = Self::aliases_from_names(tensors.keys().cloned());
        Self {
            source: ZImageVaeTensorSource::Cpu(tensors),
            aliases,
        }
    }

    fn aliases_from_names(names: impl IntoIterator<Item = String>) -> BTreeMap<String, String> {
        names
            .into_iter()
            .filter_map(|name| zimage_vae_diffusers_name(&name).map(|diffusers| (diffusers, name)))
            .collect()
    }

    fn resolve_stored_name<'a>(&'a self, name: &'a str) -> Cow<'a, str> {
        if self.contains_stored_tensor(name) {
            return Cow::Borrowed(name);
        }
        self.aliases
            .get(name)
            .map(|source| Cow::Borrowed(source.as_str()))
            .unwrap_or(Cow::Borrowed(name))
    }

    fn contains_stored_tensor(&self, name: &str) -> bool {
        match &self.source {
            ZImageVaeTensorSource::Mmap(st) => st.get(name).is_ok(),
            ZImageVaeTensorSource::Cpu(tensors) => tensors.contains_key(name),
        }
    }

    fn load_cast(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let stored_name = self.resolve_stored_name(name);
        let tensor = match &self.source {
            ZImageVaeTensorSource::Mmap(st) => st.load(stored_name.as_ref(), dev)?,
            ZImageVaeTensorSource::Cpu(tensors) => tensors
                .get(stored_name.as_ref())
                .ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "missing Z-Image VAE tensor {}",
                        stored_name.as_ref()
                    ))
                })?
                .to_device(dev)?,
        };
        if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)
        } else {
            Ok(tensor)
        }
    }
}

impl candle_nn::var_builder::SimpleBackend for ZImageVaeSafetensorsBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _init: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let mut tensor = self.load_cast(name, dtype, dev)?;
        if tensor.shape() != &shape
            && tensor.dims().len() == 4
            && shape.dims().len() == 2
            && tensor.dims()[0] == shape.dims()[0]
            && tensor.dims()[1] == shape.dims()[1]
            && tensor.dims()[2] == 1
            && tensor.dims()[3] == 1
        {
            tensor = tensor.reshape(shape.dims())?;
        }
        if tensor.shape() != &shape {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: shape,
                got: tensor.shape().clone(),
            })?
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.load_cast(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.contains_stored_tensor(name) || self.aliases.contains_key(name)
    }
}

fn zimage_vae_diffusers_name(source_name: &str) -> Option<String> {
    if source_name.starts_with("first_stage_model.") {
        return crate::loader::vae_keys::apply_vae_rename(source_name);
    }
    if source_name.starts_with("encoder.")
        || source_name.starts_with("decoder.")
        || source_name.starts_with("quant_conv.")
        || source_name.starts_with("post_quant_conv.")
    {
        return crate::loader::vae_keys::apply_vae_rename(&format!(
            "first_stage_model.{source_name}"
        ));
    }
    None
}

fn zimage_qkv_request(name: &str) -> Option<(String, usize)> {
    for (suffix, component) in [
        (".attention.to_q.weight", 0),
        (".attention.to_k.weight", 1),
        (".attention.to_v.weight", 2),
    ] {
        if let Some(prefix) = name.strip_suffix(suffix) {
            return Some((format!("{prefix}.attention.qkv.weight"), component));
        }
    }
    None
}

fn zimage_safetensors_alias(name: &str) -> Option<Cow<'_, str>> {
    match name {
        "all_x_embedder.2-1.weight" => return Some(Cow::Borrowed("x_embedder.weight")),
        "all_x_embedder.2-1.bias" => return Some(Cow::Borrowed("x_embedder.bias")),
        "all_final_layer.2-1.linear.weight" => {
            return Some(Cow::Borrowed("final_layer.linear.weight"));
        }
        "all_final_layer.2-1.linear.bias" => {
            return Some(Cow::Borrowed("final_layer.linear.bias"));
        }
        "all_final_layer.2-1.adaLN_modulation.1.weight" => {
            return Some(Cow::Borrowed("final_layer.adaLN_modulation.1.weight"));
        }
        "all_final_layer.2-1.adaLN_modulation.1.bias" => {
            return Some(Cow::Borrowed("final_layer.adaLN_modulation.1.bias"));
        }
        _ => {}
    }
    for (requested, stored) in [
        (".attention.to_out.0.weight", ".attention.out.weight"),
        (".attention.norm_q.weight", ".attention.q_norm.weight"),
        (".attention.norm_k.weight", ".attention.k_norm.weight"),
    ] {
        if let Some(prefix) = name.strip_suffix(requested) {
            return Some(Cow::Owned(format!("{prefix}{stored}")));
        }
    }
    None
}
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

fn load_zimage_vae(
    path: &std::path::Path,
    dtype: DType,
    device: &Device,
    progress: &ProgressReporter,
    cached_tensors: Option<Arc<HashMap<String, Tensor>>>,
) -> Result<AutoEncoderKL> {
    use candle_core::safetensors::MmapedSafetensors;

    let bytes_total = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    progress.weight_load("VAE", 0, bytes_total);
    let backend = if let Some(tensors) = cached_tensors {
        ZImageVaeSafetensorsBackend::from_cpu_tensors(tensors)
    } else {
        let st = unsafe { MmapedSafetensors::multi(&[path])? };
        ZImageVaeSafetensorsBackend::new(st)
    };
    let vae_vb = candle_nn::VarBuilder::from_backend(Box::new(backend), dtype, device.clone());
    progress.weight_load("VAE", bytes_total, bytes_total);
    AutoEncoderKL::new(&VaeConfig::z_image(), vae_vb).map_err(Into::into)
}

fn zimage_qwen3_preference<'a>(
    configured: Option<&'a str>,
    text_encoder_paths: &[std::path::PathBuf],
) -> Option<&'a str> {
    if configured.is_none() && zimage_has_recipe_text_encoder(text_encoder_paths) {
        Some("bf16")
    } else {
        configured
    }
}

fn zimage_has_recipe_text_encoder(text_encoder_paths: &[std::path::PathBuf]) -> bool {
    text_encoder_paths.iter().any(|path| {
        path.components()
            .any(|component| component.as_os_str() == "civitai")
    })
}

fn model_timestep(scheduler: &FlowMatchEulerDiscreteScheduler) -> f64 {
    1.0 - scheduler.current_sigma()
}

/// Largest latent axis candle's GPU VAE decode is measured to handle whole.
///
/// Determined by bisection against a CPU reference on an M4 Max: the Z-Image
/// VAE decode is exact at latent 96, 104, 112 and 120 (relative error ~2e-5)
/// and wrong at 128 — the 1024x1024 render — returning partial data or all
/// zeros. Every constituent op is exact at those shapes in isolation (conv2d
/// including its 4.5 GiB im2col, group_norm, silu, upsample, residual add, and
/// the 16384-token mid-block attention), so it is the composition that fails.
const MAX_WHOLE_VAE_DECODE_LATENT_AXIS: usize = 120;

/// Whether this decode has to be tiled rather than run whole.
///
/// The reactive ladder in [`crate::vae_tiling::decode_with_oom_fallback`] tiles
/// after catching an OOM, which cannot help here: the failure is silent. candle
/// returns a wrong tensor instead of an error, even across a device
/// synchronization, so the only defence is to not ask it for a decode that
/// large in the first place.
///
/// Metal only. The corruption was measured on Metal (its transient buffer
/// pool crosses the GPU working set past latent ~121 — see the candle-side
/// allocator issue); CUDA and CPU decode the same shapes whole and correctly,
/// and CUDA's genuine OOMs are already covered by the reactive ladder.
fn requires_tiled_vae_decode(on_metal: bool, latent_h: usize, latent_w: usize) -> bool {
    on_metal && latent_h.max(latent_w) > MAX_WHOLE_VAE_DECODE_LATENT_AXIS
}

/// Latent-space overlap between adjacent proactive tiles. The value all three
/// upstream tilers converge on (diffusers 0.25·64, ComfyUI's fallback, and
/// sd.cpp's recomputed half-tile target at its defaults).
const PROACTIVE_TILE_OVERLAP: usize = 16;

/// Tile configuration for the proactive Metal decode: fewest tiles whose
/// every tile fits the measured whole-decode cap, single pass.
fn proactive_zimage_tile_config(latent_h: usize, latent_w: usize) -> crate::vae_tiling::TileConfig {
    crate::vae_tiling::proactive_tile_config(
        MAX_WHOLE_VAE_DECODE_LATENT_AXIS,
        PROACTIVE_TILE_OVERLAP,
        latent_h,
        latent_w,
    )
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
    /// Effective VAE dtype after `MOLD_VAE_DTYPE` resolution. May differ from
    /// `dtype` when fp32 VAE decode is forced on GPU. Captured at load time.
    /// CPU VAE is always F32 regardless of this field.
    vae_dtype: DType,
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
    /// Force adaptive block-level transformer offload.
    offload: bool,
    prompt_cache: Mutex<LruCache<String, CachedTensor>>,
    /// Per-request placement override.
    pending_placement: Option<mold_core::types::DevicePlacement>,
    /// Per-request LoRA stack (effective: zero-scale entries already filtered).
    ///
    /// Set in [`InferenceEngine::generate`] at the entry point and cleared
    /// at exit. `load_transformer` / `reload_transformer` consult this to
    /// decide whether to wrap the constructed `VarBuilder` with a
    /// `ZImageLoraBackend`.
    pending_loras: Vec<LoraWeight>,
    shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
}

/// Resolve the effective LoRA list for a request: `loras` (plural) wins
/// over the singular `lora` when both are set, and zero-scale entries are
/// dropped so they don't trigger a needless transformer rebuild.
pub(crate) fn effective_zimage_loras(req: &GenerateRequest) -> Vec<LoraWeight> {
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
                    "dropping zero-scale Z-Image LoRA"
                );
            }
            keep
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
enum ZImageOffloadDecision {
    Disabled,
    Selected,
    Unsupported(&'static str),
}

fn zimage_offload_decision(
    forced_offload: bool,
    is_gguf: bool,
    has_lora: bool,
) -> ZImageOffloadDecision {
    if !forced_offload {
        return ZImageOffloadDecision::Disabled;
    }
    if is_gguf {
        return ZImageOffloadDecision::Unsupported(
            "Z-Image block-level offload is only planned for BF16/FP transformers; \
             GGUF variants already use quantized/dense GGUF-specific paths",
        );
    }
    if has_lora {
        return ZImageOffloadDecision::Unsupported(
            "Z-Image block-level offload with LoRA is not wired yet; \
             LoRA merge/bypass semantics need a dedicated offload design",
        );
    }
    ZImageOffloadDecision::Selected
}

impl ZImageEngine {
    /// The Z-Image VAE (FLUX autoencoder) encodes RGB in [-1, 1]; its decode
    /// side returns [-1, 1] and `postprocess_image` maps that to pixels.
    fn img2img_source_normalize_range() -> img_utils::NormalizeRange {
        img_utils::NormalizeRange::MinusOneToOne
    }

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

    fn load_text_tokenizer(&self, tokenizer_path: &Path) -> Result<Arc<Tokenizer>> {
        if let Some(shared_pool) = &self.shared_pool {
            return shared_pool.lock().unwrap().load_tokenizer(tokenizer_path);
        }
        Tokenizer::from_file(tokenizer_path)
            .map(Arc::new)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))
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
                progress.phase_done(
                    crate::ProgressPhase::PromptEncode,
                    "Encoding prompt (Qwen3)",
                    encode_start.elapsed(),
                );
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

    /// Load transformer from disk, optionally merging LoRA deltas inline.
    ///
    /// When `self.pending_loras` is non-empty:
    /// * **BF16 path**: build an mmap-backed `SimpleBackend` ourselves
    ///   (`weight_loader::load_safetensors_with_progress` returns a
    ///   `VarBuilder` whose backend can't be wrapped after the fact),
    ///   wrap it with a `ZImageLoraBackend`, and feed the resulting
    ///   `VarBuilder` to `ZImageTransformer2DModel::new`.
    /// * **GGUF path**: patch only affected quantized tensors and build the
    ///   quantized transformer. The fused-QKV `attention.qkv` LoRA targets the
    ///   split `to_q`/`to_k`/`to_v` candle keys — see [`super::lora`] for the
    ///   splat math.
    fn load_transformer(
        &self,
        device: &Device,
        dtype: DType,
        cfg: &Config,
        activation_budget: u64,
    ) -> Result<ZImageTransformer> {
        let is_gguf = self.detect_is_gguf();
        let xformer_paths = self.transformer_paths();
        let has_lora = !self.pending_loras.is_empty();

        if is_gguf {
            if has_lora {
                let adapters =
                    super::lora::load_lora_adapters(&self.pending_loras, &self.base.progress)?;
                let specs: Vec<super::lora::ZImageLoraSpec<'_>> = adapters
                    .iter()
                    .zip(self.pending_loras.iter())
                    .map(|(adapter, w)| super::lora::ZImageLoraSpec {
                        adapter: adapter.as_ref(),
                        scale: w.scale,
                        path_hash: super::lora::lora_path_hash(&w.path),
                    })
                    .collect();
                let vb = super::lora::gguf_lora_var_builder(
                    &self.base.paths.transformer,
                    &specs,
                    device,
                    &self.base.progress,
                )?;
                return Ok(ZImageTransformer::Quantized(Box::new(
                    super::quantized_transformer::QuantizedZImageTransformer2DModel::new(
                        cfg, dtype, vb,
                    )?,
                )));
            }
            let qvb =
                quantized_var_builder::VarBuilder::from_gguf(&self.base.paths.transformer, device)?;
            // Keep GGUF weights quantized at runtime instead of inflating them
            // into a dense map. Dequantizing the whole checkpoint turns the
            // 3.4 GB Q4_K file into ~24 GB resident on Metal (F32) — mold
            // reported 15.3 GB free after load, against 30.0 GB quantized — and
            // that pressure made Metal command buffers fail mid-denoise, which
            // is silent: the render completed and produced a coherent image
            // that had nothing to do with the prompt. Quantized is also what
            // stable-diffusion.cpp does with these same files, and it is both
            // correct and ~2x faster here (40 s vs 85 s at 768x768).
            Ok(ZImageTransformer::Quantized(Box::new(
                super::quantized_transformer::QuantizedZImageTransformer2DModel::new(
                    cfg, dtype, qvb,
                )?,
            )))
        } else if has_lora {
            // Build an mmap-backed SimpleBackend so we can layer a LoRA
            // wrapper on top. Drops the streaming progress bar that
            // `load_safetensors_with_progress` would emit; the LoRA
            // info-line is good enough signal for a single load.
            use candle_core::safetensors::MmapedSafetensors;
            let path_refs: Vec<&std::path::Path> =
                xformer_paths.iter().map(|p| p.as_path()).collect();
            let st = unsafe { MmapedSafetensors::multi(&path_refs)? };
            let inner: Box<dyn candle_nn::var_builder::SimpleBackend> =
                Box::new(ZImageSafetensorsBackend::new(st));
            let adapters =
                super::lora::load_lora_adapters(&self.pending_loras, &self.base.progress)?;
            let specs: Vec<super::lora::ZImageLoraSpec<'_>> = adapters
                .iter()
                .zip(self.pending_loras.iter())
                .map(|(adapter, w)| super::lora::ZImageLoraSpec {
                    adapter: adapter.as_ref(),
                    scale: w.scale,
                    path_hash: super::lora::lora_path_hash(&w.path),
                })
                .collect();
            let wrapped =
                super::lora::wrap_backend_with_lora(inner, &specs, &self.base.progress, None)?;
            let vb = candle_nn::VarBuilder::from_backend(wrapped, dtype, device.clone());
            Ok(ZImageTransformer::Dense(Box::new(
                MoldZImageTransformer2DModel::new(cfg, vb)?,
            )))
        } else if self.offload {
            use candle_core::safetensors::MmapedSafetensors;
            let path_refs: Vec<&std::path::Path> =
                xformer_paths.iter().map(|p| p.as_path()).collect();
            let bytes_total: u64 = xformer_paths
                .iter()
                .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
                .sum();
            self.base
                .progress
                .weight_load("Z-Image transformer (offload stems)", 0, bytes_total);
            let gpu_st = unsafe { MmapedSafetensors::multi(&path_refs)? };
            let cpu_st = unsafe { MmapedSafetensors::multi(&path_refs)? };
            let gpu_vb = candle_nn::VarBuilder::from_backend(
                Box::new(ZImageSafetensorsBackend::new(gpu_st)),
                dtype,
                device.clone(),
            );
            let cpu_vb = candle_nn::VarBuilder::from_backend(
                Box::new(ZImageSafetensorsBackend::new(cpu_st)),
                dtype,
                Device::Cpu,
            );
            self.base.progress.weight_load(
                "Z-Image transformer (offload stems)",
                bytes_total,
                bytes_total,
            );
            Ok(ZImageTransformer::Offloaded(Box::new(
                super::offload::OffloadedZImageTransformer::new(
                    cfg,
                    gpu_vb,
                    cpu_vb,
                    self.base.gpu_ordinal,
                    activation_budget,
                    &self.base.progress,
                )?,
            )))
        } else {
            use candle_core::safetensors::MmapedSafetensors;
            let path_refs: Vec<&std::path::Path> =
                xformer_paths.iter().map(|p| p.as_path()).collect();
            let bytes_total = xformer_paths
                .iter()
                .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
                .sum();
            self.base
                .progress
                .weight_load("Z-Image transformer", 0, bytes_total);
            let st = unsafe { MmapedSafetensors::multi(&path_refs)? };
            let xformer_vb = candle_nn::VarBuilder::from_backend(
                Box::new(ZImageSafetensorsBackend::new(st)),
                dtype,
                device.clone(),
            );
            self.base
                .progress
                .weight_load("Z-Image transformer", bytes_total, bytes_total);
            Ok(ZImageTransformer::Dense(Box::new(
                MoldZImageTransformer2DModel::new(cfg, xformer_vb)?,
            )))
        }
    }

    /// Load VAE from disk.
    fn load_vae(&self, device: &Device, dtype: DType) -> Result<AutoEncoderKL> {
        let cached_tensors = self.load_vae_cpu_tensors()?;
        load_zimage_vae(
            self.base.paths.vae.as_path(),
            dtype,
            device,
            &self.base.progress,
            cached_tensors,
        )
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

        let transformer_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.transformer.clone()),
            false,
        );
        let device = crate::device::resolve_device(Some(transformer_ref), || {
            crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)
        })?;
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

        let transformer = self.load_transformer(&device, dtype, &transformer_cfg, 0)?;

        self.base
            .progress
            .stage_done(&xformer_label, xformer_start.elapsed());
        tracing::info!(quantized = is_gguf, "Z-Image transformer loaded");

        // --- Decide where to place VAE and Qwen3 text encoder based on remaining VRAM ---
        // Log the raw driver reading; budget the placement decisions
        // against the reserve-adjusted value below.
        let free_raw = free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let is_cuda = device.is_cuda();
        let is_metal = device.is_metal();
        if free_raw > 0 {
            self.base.progress.info(&format!(
                "Free VRAM after transformer: {}",
                fmt_gb(free_raw)
            ));
            tracing::info!(
                free_vram = free_raw,
                free_vram_usable = free,
                "free VRAM after loading transformer"
            );
        }

        // Eager mode drops the transformer before decode, so placement only
        // needs enough room for VAE weights at load time. Decode workspace is
        // allocated later after the large transformer has been released.
        let vae_on_gpu = should_use_gpu(is_cuda, is_metal, free, VAE_WEIGHT_LOAD_VRAM_THRESHOLD);
        let vae_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.vae.clone()),
            false,
        );
        let vae_device = crate::device::resolve_device(Some(vae_ref), || {
            Ok(if vae_on_gpu {
                device.clone()
            } else {
                Device::Cpu
            })
        })?;
        let vae_on_gpu = !vae_device.is_cpu();
        // GPU branch honours `MOLD_VAE_DTYPE`; CPU is already F32 (the highest
        // precision we'd ever want), so the env knob is a no-op there.
        let vae_dtype = if vae_on_gpu {
            crate::device::resolve_vae_dtype(dtype)
        } else {
            DType::F32
        };
        let vae_device_label = if vae_on_gpu { "GPU" } else { "CPU" };

        if !vae_on_gpu && (is_cuda || is_metal) {
            self.base.progress.info(&format!(
                "VAE on CPU ({} free < {} threshold for VAE weight load)",
                fmt_gb(free),
                fmt_gb(VAE_WEIGHT_LOAD_VRAM_THRESHOLD),
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
        let qwen3_preference = zimage_qwen3_preference(
            self.qwen3_variant.as_deref(),
            &self.base.paths.text_encoder_files,
        );
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

        let qwen3_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| adv.qwen.clone(),
            true,
        );
        let auto_te_device = if te_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let te_device =
            crate::device::resolve_device(Some(qwen3_ref), || Ok(auto_te_device.clone()))?;
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
        let text_tokenizer = self.load_text_tokenizer(&text_tokenizer_path)?;

        let text_encoder = if is_qwen3_gguf {
            encoders::qwen3::Qwen3Encoder::load_gguf_with_tokenizer(
                &resolved_paths[0],
                &text_tokenizer_path,
                Some(text_tokenizer),
                &te_device,
                &bf16_cfg,
            )?
        } else {
            encoders::qwen3::Qwen3Encoder::load_bf16_with_tokenizer(
                &resolved_paths,
                &text_tokenizer_path,
                Some(text_tokenizer),
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
            vae_dtype,
            is_gguf,
            vae_path: self.base.paths.vae.clone(),
        });

        tracing::info!(model = %self.base.model_name, "all Z-Image components loaded successfully");
        Ok(())
    }

    /// Reload the transformer from disk (called when it was dropped to free VRAM for VAE decode).
    fn reload_transformer(&self, loaded: &mut LoadedZImage) -> Result<()> {
        let transformer =
            self.load_transformer(&loaded.device, loaded.dtype, &loaded.transformer_cfg, 0)?;
        loaded.transformer = Some(transformer);
        Ok(())
    }

    fn uses_sequential_generate_path(&self) -> bool {
        self.base.load_strategy == LoadStrategy::Sequential
            || self.offload
            || !self.pending_loras.is_empty()
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

        match zimage_offload_decision(self.offload, is_gguf, !self.pending_loras.is_empty()) {
            ZImageOffloadDecision::Disabled => {}
            ZImageOffloadDecision::Unsupported(reason) => bail!("{reason}"),
            ZImageOffloadDecision::Selected => {}
        }

        // Check memory budget
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
            // Reserve-adjusted reading drives the Qwen3 variant selection.
            let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
            self.base.progress.stage_start("Selecting Qwen3 encoder");
            let qwen3_resolve_start = Instant::now();
            let qwen3_preference = zimage_qwen3_preference(
                self.qwen3_variant.as_deref(),
                &self.base.paths.text_encoder_files,
            );
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

            let qwen3_ref = effective_device_ref(
                self.pending_placement.as_ref(),
                |adv| adv.qwen.clone(),
                true,
            );
            let auto_te_device = if te_on_gpu {
                device.clone()
            } else {
                Device::Cpu
            };
            let te_device =
                crate::device::resolve_device(Some(qwen3_ref), || Ok(auto_te_device.clone()))?;
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
            let te_activation_budget = crate::device::activation_bytes(
                req.width,
                req.height,
                1,
                crate::device::dtype_bytes(te_dtype),
                crate::device::ActivationFamily::SmallTransformer,
            );
            preflight_memory_check("Qwen3 text encoder", te_size, te_activation_budget)?;

            if let Some(status) = memory_status_string() {
                self.base.progress.info(&status);
            }

            self.base.progress.stage_start(&te_label);
            let te_start = Instant::now();
            let text_tokenizer = self.load_text_tokenizer(&text_tokenizer_path)?;

            let mut text_encoder = if is_qwen3_gguf {
                encoders::qwen3::Qwen3Encoder::load_gguf_with_tokenizer(
                    &resolved_paths[0],
                    &text_tokenizer_path,
                    Some(text_tokenizer),
                    &te_device,
                    &bf16_cfg,
                )?
            } else {
                encoders::qwen3::Qwen3Encoder::load_bf16_with_tokenizer(
                    &resolved_paths,
                    &text_tokenizer_path,
                    Some(text_tokenizer),
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
                crate::device::resolve_vae_dtype(dtype)
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
                Self::img2img_source_normalize_range(),
                &encode_vae_device,
                encode_vae_dtype,
            )?;
            let encoded = encode_vae.encode(&source_tensor)?;
            self.base.progress.phase_done(
                crate::ProgressPhase::Vae,
                "Encoding source image (VAE)",
                encode_start.elapsed(),
            );

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
        let xformer_activation_budget = crate::device::activation_bytes(
            req.width,
            req.height,
            1,
            crate::device::dtype_bytes(dtype),
            crate::device::ActivationFamily::ZImageDit,
        );
        preflight_memory_check(
            "Z-Image transformer",
            xformer_size,
            xformer_activation_budget,
        )?;

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
        let transformer =
            self.load_transformer(&device, dtype, &transformer_cfg, xformer_activation_budget)?;
        self.base
            .progress
            .stage_done(&xformer_label, xformer_start.elapsed());

        let num_steps = scheduler.sigmas.len().saturating_sub(1);
        let denoise_label = format!("Denoising ({} steps)", num_steps);
        self.base.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();
        let previewer = crate::latent_preview::LatentPreviewer::zimage();

        for step in 0..num_steps {
            self.base.progress.checkpoint()?;
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
            if previewer.due(step + 1, num_steps) {
                // Preview the predicted clean image (x0 = x_t - sigma*v),
                // not the still-noisy working latent.
                let sigma_next = scheduler.sigmas[step + 1];
                match latents
                    .squeeze(2)
                    .and_then(|x| &x - &(&noise_pred_4d * sigma_next)?)
                {
                    Ok(x0_est) => {
                        previewer.maybe_emit(&self.base.progress, &x0_est, step + 1, num_steps)
                    }
                    Err(e) => tracing::warn!("skipping denoise preview: {e}"),
                }
            }
        }

        self.base.progress.checkpoint()?;
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
        // With sequential loading, we can always try GPU for VAE since transformer is freed.
        // Reserve-adjusted reading: should_use_gpu must respect the OS reserve.
        let free_for_vae = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let vae_on_gpu = should_use_gpu(
            device.is_cuda(),
            device.is_metal(),
            free_for_vae,
            VAE_DECODE_VRAM_THRESHOLD,
        );
        let vae_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.vae.clone()),
            false,
        );
        let vae_device = crate::device::resolve_device(Some(vae_ref), || {
            Ok(if vae_on_gpu {
                device.clone()
            } else {
                Device::Cpu
            })
        })?;
        let vae_on_gpu = !vae_device.is_cpu();
        // GPU branch honours `MOLD_VAE_DTYPE`; CPU is already F32 (the highest
        // precision we'd ever want), so the env knob is a no-op there.
        let vae_dtype = if vae_on_gpu {
            crate::device::resolve_vae_dtype(dtype)
        } else {
            DType::F32
        };
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
        // Same Metal whole-decode span guard as the eager path. This decode
        // ran unguarded, so every sequential-mode render (Sequential load
        // strategy, --offload, or any LoRA request) past latent 120 hit the
        // silent Metal corruption the eager path was already tiling around.
        let (_, _, latent_h, latent_w) = latents.dims4()?;
        let image = if requires_tiled_vae_decode(vae_device.is_metal(), latent_h, latent_w) {
            let cfg = proactive_zimage_tile_config(latent_h, latent_w);
            tracing::info!(
                latent_h,
                latent_w,
                tile_size = cfg.tile_size,
                overlap = cfg.overlap,
                "Z-Image latent exceeds the whole-decode span; tiling VAE decode"
            );
            self.base
                .progress
                .info("VAE decode tiled (latent exceeds whole-decode span)");
            crate::vae_tiling::decode_tiled(
                &latents,
                |tile| vae.decode(tile).map_err(Into::into),
                &cfg,
            )?
        } else {
            vae.decode(&latents)?
        };
        let image = postprocess_image(&image)?;
        let image = image.i(0)?;

        self.base.progress.phase_done(
            crate::ProgressPhase::Vae,
            "VAE decode",
            vae_decode_start.elapsed(),
        );

        // VAE dropped here
        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &image,
            req.resolved_output_format(),
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

impl ZImageEngine {
    fn generate_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if req.scheduler.is_some() {
            tracing::warn!(
                "scheduler selection not supported for Z-Image (flow-matching), ignoring"
            );
        }
        // Sequential mode: load-use-drop each component. LoRA requests also
        // use this path because LoRA-merged transformer construction has
        // higher transient memory and must not run while eager-mode VAE/text
        // encoders are still GPU-resident.
        if self.uses_sequential_generate_path() {
            self.base.unload();
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components
        if self.base.loaded.is_none() {
            self.load()?;
        }
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
            // Cache miss — restore encoder if it was dropped or parked after
            // a previous generation. is_parked() is true only on the BF16
            // path; GGUF flows through the reload() branch.
            if loaded.text_encoder.model.is_none() {
                let te_label = if loaded.text_encoder.is_parked() {
                    "Unparking Qwen3 encoder (CPU→GPU)"
                } else if loaded.text_encoder.is_quantized {
                    "Reloading Qwen3 encoder (GGUF)"
                } else {
                    "Reloading Qwen3 encoder (BF16)"
                };
                progress.stage_start(te_label);
                let reload_start = Instant::now();
                if loaded.text_encoder.is_parked() {
                    loaded.text_encoder.unpark_to_gpu(progress)?;
                } else {
                    loaded.text_encoder.reload(progress)?;
                }
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

            // Free GPU VRAM for denoising + VAE decode. With
            // `MOLD_KEEP_TE_RAM=1` and the BF16 encoder, parameters move
            // to host RAM instead of being released — saves ~10 s of reload
            // on the next request. GGUF and Metal flow through the original
            // drop path (Metal is unified memory, GGUF is device-tied).
            if loaded.text_encoder.on_gpu || loaded.device.is_metal() {
                let park_mode = crate::device::keep_te_in_ram()
                    && !loaded.device.is_metal()
                    && !loaded.text_encoder.is_quantized;
                if park_mode {
                    loaded.text_encoder.park_to_cpu()?;
                    tracing::info!(
                        on_gpu = loaded.text_encoder.on_gpu,
                        "Qwen3 text encoder parked to CPU host RAM"
                    );
                } else {
                    loaded.text_encoder.drop_weights();
                    tracing::info!(
                        on_gpu = loaded.text_encoder.on_gpu,
                        "Qwen3 text encoder dropped to free memory for denoising"
                    );
                }
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
                Self::img2img_source_normalize_range(),
                vae_encode_device,
                vae_encode_dtype,
            )?;
            let encoded = loaded.vae.encode(&source_tensor)?;
            progress.phase_done(
                crate::ProgressPhase::Vae,
                "Encoding source image (VAE)",
                encode_start.elapsed(),
            );

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
        let previewer = crate::latent_preview::LatentPreviewer::zimage();

        // Scope the transformer borrow so it can be dropped before VAE decode
        {
            let transformer = loaded
                .transformer
                .as_ref()
                .expect("transformer must be loaded for denoising");

            for step in 0..num_steps {
                progress.checkpoint()?;
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
                if previewer.due(step + 1, num_steps) {
                    let sigma_next = scheduler.sigmas[step + 1];
                    match latents
                        .squeeze(2)
                        .and_then(|x| &x - &(&noise_pred_4d * sigma_next)?)
                    {
                        Ok(x0_est) => previewer.maybe_emit(progress, &x0_est, step + 1, num_steps),
                        Err(e) => tracing::warn!("skipping denoise preview: {e}"),
                    }
                }
            }
        }

        progress.checkpoint()?;
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
                    loaded.vae_dtype
                },
            )?;
            let (_, _, latent_h, latent_w) = decode_latents.dims4()?;
            if requires_tiled_vae_decode(loaded.vae_device.is_metal(), latent_h, latent_w) {
                // Past the span candle's Metal decode handles whole. Tiling is
                // proactive rather than a reaction to an OOM because the
                // failure is silent — a whole decode here returns a wrong
                // tensor, not an error, so there is nothing to catch.
                let cfg = proactive_zimage_tile_config(latent_h, latent_w);
                tracing::info!(
                    latent_h,
                    latent_w,
                    tile_size = cfg.tile_size,
                    overlap = cfg.overlap,
                    "Z-Image latent exceeds the whole-decode span; tiling VAE decode"
                );
                progress.info("VAE decode tiled (latent exceeds whole-decode span)");
                crate::vae_tiling::decode_tiled(
                    &decode_latents,
                    |tile| loaded.vae.decode(tile).map_err(Into::into),
                    &cfg,
                )?
            } else {
                match loaded.vae.decode(&decode_latents) {
                    Ok(img) => img,
                    // Any GPU backend, not just CUDA. This retry was gated on
                    // `is_cuda()` and keyed off the CUDA-only OOM spellings, so on
                    // Metal — where the message is `Insufficient Memory
                    // (…kIOGPUCommandBufferCallbackErrorOutOfMemory)` — a decode
                    // that had already paid for the whole denoise died outright
                    // instead of finishing on the CPU.
                    Err(e)
                        if !loaded.vae_device.is_cpu()
                            && crate::vae_tiling::is_out_of_memory_error(&e) =>
                    {
                        tracing::warn!("VAE decode OOM on GPU, falling back to CPU");
                        progress.info("VAE decode OOM on GPU — retrying on CPU");
                        loaded.device.synchronize()?;
                        // Load a fresh VAE on CPU (can't call self.load_vae_cpu() due to borrow)
                        let cpu_vae = load_zimage_vae(
                            loaded.vae_path.as_path(),
                            DType::F32,
                            &Device::Cpu,
                            progress,
                            None,
                        )?;
                        let cpu_latents =
                            latents_4d.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                        cpu_vae.decode(&cpu_latents)?
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        };

        // Post-process: [-1, 1] → [0, 255] (candle z_image utility)
        let image = postprocess_image(&image)?;
        let image = image.i(0)?; // Remove batch dimension → [3, H, W]

        progress.phase_done(crate::ProgressPhase::Vae, "VAE decode", vae_start.elapsed());

        // 9. Encode to output format
        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &image,
            req.resolved_output_format(),
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "Z-Image generation complete");

        Ok(GenerateResponse {
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

impl InferenceEngine for ZImageEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.base.progress.checkpoint()?;
        self.pending_placement = req.placement.clone();
        self.pending_loras = effective_zimage_loras(req);
        let result = self.generate_inner(req);
        self.pending_placement = None;
        self.pending_loras.clear();
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

    fn load_for_request(&mut self, req: &GenerateRequest) -> Result<()> {
        self.pending_placement = req.placement.clone();
        self.pending_loras = effective_zimage_loras(req);
        let result = if !self.pending_loras.is_empty()
            && self.base.load_strategy != LoadStrategy::Sequential
        {
            Err(anyhow::anyhow!(
                "Z-Image LoRA requests require a sequential engine load plan; \
                 refusing to preload an unadapted transformer"
            ))
        } else {
            ZImageEngine::load(self)
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
        crate::batch_execution_capability_for_family("z-image")
            .expect("production Z-Image batch capability must be registered")
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
    use crate::shared_pool::SharedPool;
    use mold_core::ModelPaths;
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

    fn zimage_model_paths(
        transformer: PathBuf,
        transformer_shards: Vec<PathBuf>,
        vae: PathBuf,
        text_tokenizer: Option<PathBuf>,
    ) -> ModelPaths {
        ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
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
    fn zimage_safetensors_backend_accepts_civitai_diffusion_prefix() {
        use candle_nn::var_builder::SimpleBackend;
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;

        fn f32_bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        let dir = temp_test_dir("mold-zimage-prefix-backend");
        let path = dir.join("zimage.safetensors");
        let data = f32_bytes(&[42.0]);
        let qkv = f32_bytes(&[1.0, 2.0, 3.0]);
        let out = f32_bytes(&[7.0]);
        let q_norm = f32_bytes(&[8.0]);
        let k_norm = f32_bytes(&[9.0]);
        let mut tensors = HashMap::new();
        tensors.insert(
            format!("{ZIMAGE_SINGLE_FILE_PREFIX}t_embedder.mlp.0.weight"),
            TensorView::new(SafeDtype::F32, vec![1, 1], data.as_slice()).unwrap(),
        );
        tensors.insert(
            format!("{ZIMAGE_SINGLE_FILE_PREFIX}x_embedder.weight"),
            TensorView::new(SafeDtype::F32, vec![1, 1], data.as_slice()).unwrap(),
        );
        tensors.insert(
            format!("{ZIMAGE_SINGLE_FILE_PREFIX}noise_refiner.0.attention.qkv.weight"),
            TensorView::new(SafeDtype::F32, vec![3, 1], qkv.as_slice()).unwrap(),
        );
        tensors.insert(
            format!("{ZIMAGE_SINGLE_FILE_PREFIX}noise_refiner.0.attention.out.weight"),
            TensorView::new(SafeDtype::F32, vec![1, 1], out.as_slice()).unwrap(),
        );
        tensors.insert(
            format!("{ZIMAGE_SINGLE_FILE_PREFIX}noise_refiner.0.attention.q_norm.weight"),
            TensorView::new(SafeDtype::F32, vec![1], q_norm.as_slice()).unwrap(),
        );
        tensors.insert(
            format!("{ZIMAGE_SINGLE_FILE_PREFIX}noise_refiner.0.attention.k_norm.weight"),
            TensorView::new(SafeDtype::F32, vec![1], k_norm.as_slice()).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path.as_path()]) }
            .unwrap();
        let backend = ZImageSafetensorsBackend::new(st);
        assert!(backend.contains_tensor("t_embedder.mlp.0.weight"));
        let tensor = backend
            .get_unchecked("t_embedder.mlp.0.weight", DType::F32, &Device::Cpu)
            .unwrap();
        assert_eq!(tensor.to_vec2::<f32>().unwrap(), vec![vec![42.0]]);
        assert!(backend.contains_tensor("all_x_embedder.2-1.weight"));
        let alias_tensor = backend
            .get_unchecked("all_x_embedder.2-1.weight", DType::F32, &Device::Cpu)
            .unwrap();
        assert_eq!(alias_tensor.to_vec2::<f32>().unwrap(), vec![vec![42.0]]);
        assert!(backend.contains_tensor("noise_refiner.0.attention.to_q.weight"));
        assert!(backend.contains_tensor("noise_refiner.0.attention.to_k.weight"));
        assert!(backend.contains_tensor("noise_refiner.0.attention.to_v.weight"));
        let k = backend
            .get(
                Shape::from((1, 1)),
                "noise_refiner.0.attention.to_k.weight",
                candle_nn::Init::Const(0.0),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(k.to_vec2::<f32>().unwrap(), vec![vec![2.0]]);
        let out = backend
            .get_unchecked(
                "noise_refiner.0.attention.to_out.0.weight",
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(out.to_vec2::<f32>().unwrap(), vec![vec![7.0]]);
        let q_norm = backend
            .get_unchecked(
                "noise_refiner.0.attention.norm_q.weight",
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(q_norm.to_vec1::<f32>().unwrap(), vec![8.0]);
        let k_norm = backend
            .get_unchecked(
                "noise_refiner.0.attention.norm_k.weight",
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(k_norm.to_vec1::<f32>().unwrap(), vec![9.0]);

        let _ = std::fs::remove_dir_all(dir);
    }

    /// z-image-turbo:bf16 regression: the official Tongyi checkpoint stores
    /// SPLIT diffusers attention names (`to_q`/`to_k`/`to_v`, `norm_q`,
    /// `to_out.0`) — no fused `attention.qkv` anywhere. The backend used to
    /// translate every split request into a fused load unconditionally and
    /// die with "cannot find tensor noise_refiner.0.attention.qkv.weight".
    /// Directly-stored names must win before the fused-QKV adaptation.
    #[test]
    fn zimage_backend_serves_split_qkv_checkpoints_directly() {
        use candle_nn::var_builder::SimpleBackend;
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;

        fn f32_bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        let dir = temp_test_dir("mold-zimage-split-backend");
        let path = dir.join("zimage-split.safetensors");
        let q = f32_bytes(&[1.0]);
        let k = f32_bytes(&[2.0]);
        let v = f32_bytes(&[3.0]);
        let out = f32_bytes(&[7.0]);
        let norm_q = f32_bytes(&[8.0]);
        let mut tensors = HashMap::new();
        tensors.insert(
            "noise_refiner.0.attention.to_q.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1, 1], q.as_slice()).unwrap(),
        );
        tensors.insert(
            "noise_refiner.0.attention.to_k.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1, 1], k.as_slice()).unwrap(),
        );
        tensors.insert(
            "noise_refiner.0.attention.to_v.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1, 1], v.as_slice()).unwrap(),
        );
        tensors.insert(
            "noise_refiner.0.attention.to_out.0.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1, 1], out.as_slice()).unwrap(),
        );
        tensors.insert(
            "noise_refiner.0.attention.norm_q.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1], norm_q.as_slice()).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path.as_path()]) }
            .unwrap();
        let backend = ZImageSafetensorsBackend::new(st);

        assert!(backend.contains_tensor("noise_refiner.0.attention.to_q.weight"));
        let q = backend
            .get(
                Shape::from((1, 1)),
                "noise_refiner.0.attention.to_q.weight",
                candle_nn::Init::Const(0.0),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(q.to_vec2::<f32>().unwrap(), vec![vec![1.0]]);
        let v = backend
            .get_unchecked(
                "noise_refiner.0.attention.to_v.weight",
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(v.to_vec2::<f32>().unwrap(), vec![vec![3.0]]);
        let out = backend
            .get_unchecked(
                "noise_refiner.0.attention.to_out.0.weight",
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(out.to_vec2::<f32>().unwrap(), vec![vec![7.0]]);
        let norm_q = backend
            .get_unchecked(
                "noise_refiner.0.attention.norm_q.weight",
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(norm_q.to_vec1::<f32>().unwrap(), vec![8.0]);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn zimage_vae_backend_accepts_bare_ldm_vae_keys() {
        use candle_nn::var_builder::SimpleBackend;
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;

        fn f32_bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        let dir = temp_test_dir("mold-zimage-vae-backend");
        let path = dir.join("vae.safetensors");
        let norm = f32_bytes(&[5.0]);
        let conv = f32_bytes(&[7.0]);
        let attn_q = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let mut tensors = HashMap::new();
        tensors.insert(
            "encoder.down.0.block.0.norm1.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1], norm.as_slice()).unwrap(),
        );
        tensors.insert(
            "decoder.norm_out.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1], conv.as_slice()).unwrap(),
        );
        tensors.insert(
            "encoder.mid.attn_1.q.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2, 2, 1, 1], attn_q.as_slice()).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path.as_path()]) }
            .unwrap();
        let backend = ZImageVaeSafetensorsBackend::new(st);

        assert!(backend.contains_tensor("encoder.down_blocks.0.resnets.0.norm1.weight"));
        let norm = backend
            .get_unchecked(
                "encoder.down_blocks.0.resnets.0.norm1.weight",
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(norm.to_vec1::<f32>().unwrap(), vec![5.0]);

        assert!(backend.contains_tensor("decoder.conv_norm_out.weight"));
        let conv = backend
            .get_unchecked("decoder.conv_norm_out.weight", DType::F32, &Device::Cpu)
            .unwrap();
        assert_eq!(conv.to_vec1::<f32>().unwrap(), vec![7.0]);
        let q = backend
            .get(
                Shape::from((2, 2)),
                "encoder.mid_block.attentions.0.to_q.weight",
                candle_nn::Init::Const(0.0),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(
            q.to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn zimage_vae_cpu_tensor_backend_preserves_aliases_and_reshape() {
        use candle_nn::var_builder::SimpleBackend;
        use std::collections::HashMap;

        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "encoder.down.0.block.0.norm1.weight".to_string(),
            Tensor::new(&[5.0f32], &device).unwrap(),
        );
        tensors.insert(
            "encoder.mid.attn_1.q.weight".to_string(),
            Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)
                .unwrap()
                .reshape((2, 2, 1, 1))
                .unwrap(),
        );

        let backend = ZImageVaeSafetensorsBackend::from_cpu_tensors(Arc::new(tensors));

        assert!(backend.contains_tensor("encoder.down_blocks.0.resnets.0.norm1.weight"));
        let norm = backend
            .get_unchecked(
                "encoder.down_blocks.0.resnets.0.norm1.weight",
                DType::F32,
                &device,
            )
            .unwrap();
        assert_eq!(norm.to_vec1::<f32>().unwrap(), vec![5.0]);

        let q = backend
            .get(
                Shape::from((2, 2)),
                "encoder.mid_block.attentions.0.to_q.weight",
                candle_nn::Init::Const(0.0),
                DType::F32,
                &device,
            )
            .unwrap();
        assert_eq!(
            q.to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );
    }

    #[test]
    fn zimage_loads_vae_tensors_through_shared_pool() {
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;

        let dir = temp_test_dir("mold-zimage-vae-pool");
        let vae_path = dir.join("vae.safetensors");
        let weight = 1.0f32.to_le_bytes();
        let mut tensors = HashMap::new();
        tensors.insert(
            "encoder.conv_in.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1], weight.as_slice()).unwrap(),
        );
        serialize_to_file(&tensors, &None, &vae_path).unwrap();

        let shared_pool = Arc::new(Mutex::new(SharedPool::new()));
        let pooled = shared_pool
            .lock()
            .unwrap()
            .load_safetensors_cpu_tensors(std::slice::from_ref(&vae_path))
            .unwrap()
            .unwrap();

        let engine = ZImageEngine::new(
            "z-image-turbo:q4".to_string(),
            zimage_model_paths(
                dir.join("transformer.gguf"),
                vec![],
                vae_path,
                Some(dir.join("tokenizer.json")),
            ),
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
    fn zimage_recipe_text_encoder_defaults_to_bf16_variant() {
        let recipe_paths = vec![std::path::PathBuf::from(
            "/models/cv-2442439/z-image/civitai/2442439/zImageTurbo_turbo_txt.safetensors",
        )];
        let shared_paths = vec![std::path::PathBuf::from(
            "/models/shared/z-image/text_encoder/model-00001-of-00003.safetensors",
        )];

        assert_eq!(zimage_qwen3_preference(None, &recipe_paths), Some("bf16"));
        assert_eq!(zimage_qwen3_preference(None, &shared_paths), None);
        assert_eq!(
            zimage_qwen3_preference(Some("q8"), &recipe_paths),
            Some("q8")
        );
        assert_eq!(
            zimage_qwen3_preference(Some("auto"), &recipe_paths),
            Some("auto")
        );
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
    fn eager_vae_weight_load_threshold_is_below_decode_workspace_threshold() {
        const {
            assert!(VAE_WEIGHT_LOAD_VRAM_THRESHOLD < VAE_DECODE_VRAM_THRESHOLD);
        }
        assert!(should_use_gpu(
            true,
            false,
            1_000_000_000,
            VAE_WEIGHT_LOAD_VRAM_THRESHOLD
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

    // The Z-Image VAE is FLUX's autoencoder: encode expects RGB in [-1, 1]
    // (candle z_image::vae.rs encode contract; diffusers ZImageImg2ImgPipeline
    // preprocesses with VaeImageProcessor do_normalize=True). Feeding [0, 1]
    // anchors the denoise on a half-contrast, brightness-shifted latent and
    // renders washed-out img2img output.
    #[test]
    fn zimage_img2img_uses_minus_one_to_one_source_normalization() {
        assert_eq!(
            ZImageEngine::img2img_source_normalize_range(),
            img_utils::NormalizeRange::MinusOneToOne
        );
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
    fn zimage_img2img_source_decode_routes_through_normalize_contract() {
        let source = include_str!("pipeline.rs")
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("pipeline source should include production section");
        let decode_sites = source
            .split("let source_tensor = img_utils::decode_source_image(")
            .skip(1)
            .collect::<Vec<_>>();

        assert_eq!(decode_sites.len(), 2);
        for site in decode_sites {
            let args = site
                .split(")?;")
                .next()
                .expect("source decode call should terminate");
            assert!(
                args.contains("Self::img2img_source_normalize_range()"),
                "Z-Image source-image encoding must use the shared normalize contract"
            );
            assert!(
                !args.contains("img_utils::NormalizeRange::ZeroToOne"),
                "Z-Image source-image encoding must not hardcode [0, 1] normalization"
            );
        }
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
            false,
            None,
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
            false,
            None,
        );
        assert_eq!(sharded.validate_paths().unwrap(), tokenizer);
        assert!(!sharded.detect_is_gguf());

        let quantized = ZImageEngine::new(
            "z-image-turbo:q4".to_string(),
            zimage_model_paths(gguf, vec![], vae, Some(dir.join("tokenizer.json"))),
            None,
            LoadStrategy::Sequential,
            0,
            false,
            None,
        );
        assert!(quantized.detect_is_gguf());

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn zimage_lora_requests_use_sequential_generation_path() {
        let dir = temp_test_dir("mold-zimage-lora-sequential");
        let mut engine = ZImageEngine::new(
            "z-image-turbo:q8".to_string(),
            zimage_model_paths(
                dir.join("transformer.gguf"),
                vec![],
                dir.join("vae.safetensors"),
                Some(dir.join("tokenizer.json")),
            ),
            None,
            LoadStrategy::Eager,
            0,
            false,
            None,
        );
        engine.pending_loras = vec![LoraWeight {
            path: dir.join("adapter.safetensors").display().to_string(),
            scale: 1.0,

            expert: None,
        }];

        assert!(
            engine.uses_sequential_generate_path(),
            "Z-Image LoRA requests should use staged load-use-drop generation \
             so VAE/text encoders are not co-resident with the LoRA-merged transformer"
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn zimage_sequential_path_drops_eager_components_before_generation() {
        let source = include_str!("pipeline.rs");
        let sequential_branch = source
            .split("// Eager mode: use pre-loaded components")
            .next()
            .expect("generate_inner should contain eager-mode marker");

        assert!(
            sequential_branch.contains("self.base.unload();")
                && sequential_branch.contains("return self.generate_sequential(req);"),
            "Z-Image LoRA/offload sequential generation must drop eager-loaded \
             components before loading staged components"
        );
    }

    #[test]
    fn zimage_eager_path_reloads_after_sequential_generation_unloads_components() {
        let source = include_str!("pipeline.rs");
        let eager_branch = source
            .split("// Eager mode: use pre-loaded components")
            .nth(1)
            .expect("generate_inner should contain eager-mode branch");
        let reload_idx = eager_branch
            .find("self.load()?;")
            .expect("eager branch should reload an unloaded cached engine");
        let guard_idx = eager_branch
            .find("bail!(\"model not loaded")
            .expect("eager branch should retain a final loaded-state guard");

        assert!(
            reload_idx < guard_idx,
            "Z-Image eager generation must reload after a prior LoRA/offload \
             sequential request unloads cached components"
        );
    }

    #[test]
    fn zimage_forced_offload_uses_sequential_generation_path() {
        let dir = temp_test_dir("mold-zimage-offload-sequential");
        let engine = ZImageEngine::new(
            "z-image-turbo:bf16".to_string(),
            zimage_model_paths(
                dir.join("transformer.safetensors"),
                vec![],
                dir.join("vae.safetensors"),
                Some(dir.join("tokenizer.json")),
            ),
            None,
            LoadStrategy::Eager,
            0,
            true,
            None,
        );

        assert!(
            engine.uses_sequential_generate_path(),
            "Z-Image --offload requests must reach the engine and select the \
             staged generation path instead of being silently ignored"
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn zimage_offload_decision_gates_current_unsupported_cases() {
        assert_eq!(
            zimage_offload_decision(false, false, false),
            ZImageOffloadDecision::Disabled
        );
        assert_eq!(
            zimage_offload_decision(true, false, false),
            ZImageOffloadDecision::Selected
        );
        assert!(matches!(
            zimage_offload_decision(true, true, false),
            ZImageOffloadDecision::Unsupported(reason)
                if reason.contains("GGUF variants")
        ));
        assert!(matches!(
            zimage_offload_decision(true, false, true),
            ZImageOffloadDecision::Unsupported(reason)
                if reason.contains("LoRA")
        ));
    }

    #[test]
    fn zimage_selected_bf16_offload_reaches_runtime_loader() {
        let dir = temp_test_dir("mold-zimage-offload-loader");
        let mut engine = ZImageEngine::new(
            "z-image-turbo:bf16".to_string(),
            zimage_model_paths(
                touch(&dir, "transformer.safetensors"),
                vec![],
                touch(&dir, "vae.safetensors"),
                Some(touch(&dir, "tokenizer.json")),
            ),
            None,
            LoadStrategy::Sequential,
            0,
            true,
            None,
        );
        let hidden_size = encoders::qwen3_bf16::Qwen3BF16Config::qwen3_4b().hidden_size;
        let cap_feats = Tensor::zeros((1, 1, hidden_size), DType::F32, &Device::Cpu).unwrap();
        engine.prompt_cache.lock().unwrap().insert(
            prompt_text_key("a cat"),
            CachedTensor::from_tensor(&cap_feats).unwrap(),
        );
        let req = GenerateRequest {
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
            model: "z-image-turbo:bf16".to_string(),
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
            placement: None,
        };

        let err = engine.generate_sequential(&req).unwrap_err().to_string();

        assert!(
            !err.contains("streaming is not implemented yet"),
            "selected BF16 offload must reach the runtime loader, got: {err}"
        );
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
            false,
            None,
        );

        let err = engine.validate_paths().unwrap_err();
        assert!(err.to_string().contains("text tokenizer path required"));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn zimage_loads_qwen3_tokenizer_through_shared_pool() {
        let dir = temp_test_dir("mold-zimage-tokenizer-pool");
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

        let engine = ZImageEngine::new(
            "z-image-turbo:q4".to_string(),
            zimage_model_paths(
                dir.join("transformer.gguf"),
                vec![],
                dir.join("vae.safetensors"),
                Some(tokenizer_path.clone()),
            ),
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

    // ── Tiled VAE decode threshold ─────────────────────────────────────────

    #[test]
    fn large_latents_are_decoded_in_tiles_on_metal_only() {
        // Measured on an M4 Max against a CPU reference: candle's Metal decode
        // of this VAE is exact at latent 96, 104, 112 and 120, and returns
        // partial or all-zero data at 128 — silently, with no error to catch,
        // which is why the reactive OOM ladder cannot help here.
        for latent in [64usize, 96, 112, 120] {
            assert!(
                !requires_tiled_vae_decode(true, latent, latent),
                "latent {latent} decodes correctly whole; tiling would only cost time"
            );
        }
        for latent in [128usize, 160, 256] {
            assert!(
                requires_tiled_vae_decode(true, latent, latent),
                "latent {latent} is past the measured-safe span and must be tiled"
            );
        }
    }

    #[test]
    fn a_non_square_latent_is_judged_by_its_longest_axis() {
        // 1024x768 is a supported Z-Image shape and puts one axis past the
        // span while the other stays inside it.
        assert!(requires_tiled_vae_decode(true, 96, 128));
        assert!(requires_tiled_vae_decode(true, 128, 96));
    }

    #[test]
    fn non_metal_decodes_are_never_proactively_tiled() {
        // The corruption was measured on Metal only. CUDA and CPU decode
        // whole — charging CUDA the tiled-work multiple for a Metal bug cost
        // every 1024² render ~7× the decode work, and the reactive
        // decode_with_oom_fallback ladder still covers genuine CUDA OOMs.
        for latent in [128usize, 256, 512] {
            assert!(!requires_tiled_vae_decode(false, latent, latent));
        }
    }

    #[test]
    fn proactive_tiling_uses_the_cap_derived_single_pass_config() {
        // The forced-tiling path must size tiles from the measured cap
        // (fewest tiles that fit 120), not reuse the OOM-fallback default —
        // 64/16/3 cost 27 decode calls (6.75×) where 72/16/1 costs 4 (1.27×).
        let cfg = proactive_zimage_tile_config(128, 128);
        assert_eq!((cfg.tile_size, cfg.overlap, cfg.offsets), (72, 16, 1));
        assert!(cfg.tile_size <= MAX_WHOLE_VAE_DECODE_LATENT_AXIS);
    }
}
