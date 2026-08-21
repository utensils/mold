use anyhow::{bail, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::stable_diffusion::schedulers::PredictionType;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths, Scheduler};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::cache::{
    cfg_prompt_cache_key, clear_cache, get_or_insert_cached_tensor, image_size_cache_key,
    latent_size_cache_key, restore_cached_tensor, CachedTensor, CfgPromptCacheKey,
    ImageSizeCacheKey, LatentSizeCacheKey, LruCache, DEFAULT_IMAGE_CACHE_CAPACITY,
    DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::cfg_plus_ddim::DdimAlphaSchedule;
use crate::controlnet::ControlNetModel;
use crate::device::{check_memory_budget, memory_status_string, preflight_memory_check};
use crate::engine::{cfg_active, rand_seed, resolve_cfg_plus, InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image};
use crate::progress::{ProgressCallback, ProgressEvent};

/// VAE scaling factor for SD1.5 models.
const VAE_SCALE: f64 = 0.18215;

/// ControlNet context: loaded model, preprocessed control tensor, and scale.
struct ControlNetContext {
    model: ControlNetModel,
    control_tensor: Tensor,
    scale: f64,
}

/// Loaded SD1.5 model components, ready for inference.
struct LoadedSD15 {
    /// None after being dropped for VAE decode VRAM; reloaded on next generate.
    unet: Option<stable_diffusion::unet_2d::UNet2DConditionModel>,
    vae: stable_diffusion::vae::AutoEncoderKL,
    clip: stable_diffusion::clip::ClipTextTransformer,
    tokenizer: Arc<tokenizers::Tokenizer>,
    sd_config: stable_diffusion::StableDiffusionConfig,
    device: Device,
    /// Device the CLIP-L weights live on. May differ from `device` when the
    /// Tier 1 `text_encoders` placement override pins encoders to CPU.
    clip_device: Device,
    dtype: DType,
    /// Effective VAE dtype after `MOLD_VAE_DTYPE` resolution. May differ from
    /// `dtype` when fp32 VAE decode is forced. Captured at load time.
    vae_dtype: DType,
}

/// SD1.5 inference engine backed by candle's stable_diffusion module.
///
/// Simplified variant of SDXL: single CLIP-L encoder, smaller UNet, 512x512 default.
pub struct SD15Engine {
    base: EngineBase<LoadedSD15>,
    scheduler: Scheduler,
    shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
    prompt_cache: Mutex<LruCache<CfgPromptCacheKey, CachedTensor>>,
    source_latent_cache: Mutex<LruCache<ImageSizeCacheKey, CachedTensor>>,
    mask_cache: Mutex<LruCache<LatentSizeCacheKey, CachedTensor>>,
    control_tensor_cache: Mutex<LruCache<ImageSizeCacheKey, CachedTensor>>,
    pending_placement: Option<mold_core::types::DevicePlacement>,
    /// `Some(path)` when the engine was built from a Civitai single-file
    /// checkpoint via `from_single_file()`. `load()` branches on this to
    /// wire a custom `SingleFileBackend` (translating diffusers `vb.get`
    /// calls through `Sd15Remap` into mmap'd A1111 reads) instead of
    /// calling candle's per-component `build_*` helpers. `None` for
    /// diffusers-layout (HF) checkpoints.
    pub(crate) single_file_path: Option<PathBuf>,
    /// Per-request LoRA list, set from `req.lora` / `req.loras` in
    /// `generate()` and cleared after. Read by `build_unet_for_strategy` to
    /// decide whether to wrap the UNet's `VarBuilder` with `super::lora`'s
    /// `wrap_backend_with_lora`. Empty in the no-LoRA hot path. Mirrors the
    /// SDXL field of the same name.
    pending_loras: Vec<mold_core::LoraWeight>,
    /// Fingerprint of the LoRA stack currently merged into the loaded UNet
    /// (eager mode). When `effective_sd15_loras(req)` produces a different
    /// fingerprint, `generate_inner` drops the UNet so the next
    /// `reload_unet_if_needed` call rebuilds it with the new wrapper.
    /// Empty when no LoRA is active.
    active_lora_fingerprint: Vec<(String, u64)>,
}

/// Compute a stable fingerprint for a LoRA stack: ordered list of
/// `(path, scale_bits)`. Two stacks are equal iff they merge to the same
/// transformer, so a comparison drives UNet reload on any change (swap,
/// scale, add, remove, reorder).
fn lora_stack_fingerprint(loras: &[mold_core::LoraWeight]) -> Vec<(String, u64)> {
    loras
        .iter()
        .map(|w| (w.path.clone(), w.scale.to_bits()))
        .collect()
}

impl SD15Engine {
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        scheduler: Scheduler,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy, gpu_ordinal),
            scheduler,
            shared_pool,
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            source_latent_cache: Mutex::new(LruCache::new(DEFAULT_IMAGE_CACHE_CAPACITY)),
            mask_cache: Mutex::new(LruCache::new(DEFAULT_IMAGE_CACHE_CAPACITY)),
            control_tensor_cache: Mutex::new(LruCache::new(DEFAULT_IMAGE_CACHE_CAPACITY)),
            pending_placement: None,
            single_file_path: None,
            pending_loras: Vec::new(),
            active_lora_fingerprint: Vec::new(),
        }
    }

    /// Construct a SD1.5 engine from a Civitai single-file `.safetensors`
    /// checkpoint.
    ///
    /// Header-parses the file via `loader::single_file::load(_,
    /// Family::Sd15)`, validates that the SD1.5 A1111 → diffusers rename
    /// rules cover its UNet / VAE / CLIP-L tensor keys via
    /// `loader::sd15_keys::build_sd15_remap(_)`, and stashes the path in
    /// `single_file_path`. The actual UNet / VAE / CLIP-L materialisation
    /// (with a custom `SimpleBackend` that translates each diffusers
    /// `vb.get(name)` into the corresponding A1111 key inside the mmap'd
    /// single file) lands in a downstream phase — this constructor does
    /// **not** build the model.
    ///
    /// `clip_tokenizer` is the path to a companion-pulled CLIP-L
    /// tokenizer; the tokenizer never lives inside the
    /// single-file checkpoint.
    pub fn from_single_file(
        model_name: String,
        single_file_path: PathBuf,
        clip_tokenizer: PathBuf,
        scheduler: Scheduler,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
    ) -> Result<Self> {
        if !single_file_path.exists() {
            bail!(
                "single-file checkpoint not found: {}",
                single_file_path.display()
            );
        }

        // Header-parse and validate the SD1.5 layout. `single_file::load`
        // is header-only — no tensor data is mmap'd here.
        let bundle = crate::loader::single_file::load(
            &single_file_path,
            mold_catalog::families::Family::Sd15,
        )?;

        // Validate that every diffusers key the future SimpleBackend will
        // request resolves to a real A1111 key inside the file. Catches
        // malformed checkpoints at construction time rather than deep in
        // the eventual `load()` call.
        let _remap = crate::loader::sd15_keys::build_sd15_remap(&bundle)?;

        // `ModelPaths` is a diffusers-layout view; for single-file the
        // transformer / vae / clip_encoder all materialise from the same
        // checkpoint at load time. The real branch lives on the new
        // `single_file_path` field — future `load()` consults it before
        // falling through to the diffusers `build_*` helpers.
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: single_file_path.clone(),
            transformer_shards: Vec::new(),
            vae: single_file_path.clone(),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: Some(single_file_path.clone()),
            t5_tokenizer: None,
            clip_tokenizer: Some(clip_tokenizer),
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };

        Ok(Self {
            base: EngineBase::new(model_name, paths, load_strategy, gpu_ordinal),
            scheduler,
            shared_pool,
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            source_latent_cache: Mutex::new(LruCache::new(DEFAULT_IMAGE_CACHE_CAPACITY)),
            mask_cache: Mutex::new(LruCache::new(DEFAULT_IMAGE_CACHE_CAPACITY)),
            control_tensor_cache: Mutex::new(LruCache::new(DEFAULT_IMAGE_CACHE_CAPACITY)),
            pending_placement: None,
            single_file_path: Some(single_file_path),
            pending_loras: Vec::new(),
            active_lora_fingerprint: Vec::new(),
        })
    }

    /// Validate and return required SD1.5 paths (CLIP-L encoder + tokenizer).
    fn validate_paths(&self) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
        let clip_encoder = self
            .base
            .paths
            .clip_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-L encoder path required for SD1.5 models"))?
            .clone();
        let clip_tokenizer = self
            .base
            .paths
            .clip_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP-L tokenizer path required for SD1.5 models"))?
            .clone();

        for (label, path) in [
            ("transformer (UNet)", &self.base.paths.transformer),
            ("vae", &self.base.paths.vae),
            ("clip_encoder (CLIP-L)", &clip_encoder),
            ("clip_tokenizer (CLIP-L)", &clip_tokenizer),
        ] {
            if !path.exists() {
                bail!("{label} file not found: {}", path.display());
            }
        }

        Ok((clip_encoder, clip_tokenizer))
    }

    fn load_clip_tokenizer(
        &self,
        clip_tokenizer: &std::path::Path,
    ) -> Result<Arc<tokenizers::Tokenizer>> {
        if let Some(ref pool) = self.shared_pool {
            return pool.lock().unwrap().load_tokenizer(clip_tokenizer);
        }
        Ok(Arc::new(
            tokenizers::Tokenizer::from_file(clip_tokenizer)
                .map_err(|e| anyhow::anyhow!("failed to load CLIP-L tokenizer: {e}"))?,
        ))
    }

    /// Create the SD1.5 config.
    fn sd_config(&self) -> stable_diffusion::StableDiffusionConfig {
        stable_diffusion::StableDiffusionConfig::v1_5(None, None, None)
    }

    /// Reload UNet if it was dropped after VAE decode.
    fn reload_unet_if_needed(&mut self) -> Result<()> {
        let needs_reload = self
            .base
            .loaded
            .as_ref()
            .map(|l| l.unet.is_none())
            .unwrap_or(false);

        if needs_reload {
            let sd_config = self.sd_config();
            let loaded = self.base.loaded.as_ref().unwrap();
            let device = loaded.device.clone();
            let dtype = loaded.dtype;
            let _ = loaded;

            self.base.progress.stage_start("Reloading UNet (GPU)");
            let reload_start = Instant::now();
            let unet = self.build_unet_for_strategy(&sd_config, &device, dtype)?;
            self.base.loaded.as_mut().unwrap().unet = Some(unet);
            self.base
                .progress
                .stage_done("Reloading UNet (GPU)", reload_start.elapsed());
        }
        Ok(())
    }

    /// UNet load that branches on `single_file_path` — Civitai single-file
    /// checkpoints (A1111 naming) get the `SingleFileBackend` dispatch,
    /// diffusers-layout falls through to candle's `build_unet`. Used by
    /// every UNet load site (eager `load`, sequential `generate_sequential`,
    /// and `reload_unet_if_needed`) so the branch logic lives in exactly
    /// one place.
    ///
    /// When `self.pending_loras` is non-empty, the underlying tensor source
    /// (mmap'd diffusers safetensors or `SingleFileBackend`) is wrapped in
    /// `super::lora::wrap_backend_with_lora` so the UNet construction loads
    /// `W' = W + scale·(B @ A)` for every LoRA-targeted layer. The wrapper
    /// is transparent to `UNet2DConditionModel::new`.
    fn build_unet_for_strategy(
        &self,
        sd_config: &stable_diffusion::StableDiffusionConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<stable_diffusion::unet_2d::UNet2DConditionModel> {
        let has_lora = !self.pending_loras.is_empty();
        if let Some(single_file) = self.single_file_path.as_ref() {
            let remap = Self::load_sd15_remap(single_file)?;
            if has_lora {
                self.build_unet_single_file_with_lora(single_file, &remap, sd_config, device, dtype)
            } else {
                Self::build_unet_single_file(single_file, &remap, sd_config, device, dtype)
            }
        } else if has_lora {
            self.build_unet_diffusers_with_lora(sd_config, device, dtype)
        } else {
            Ok(sd_config.build_unet(&self.base.paths.transformer, device, 4, false, dtype)?)
        }
    }

    /// Build the UNet from a diffusers-layout single safetensors file with
    /// LoRA wrappers active. Mirrors candle's `build_unet`: open the mmap,
    /// wrap it in a `SimpleBackend`, layer `Sd15LoraBackend` on top, and feed
    /// the resulting `VarBuilder` to `UNet2DConditionModel::new`.
    fn build_unet_diffusers_with_lora(
        &self,
        _sd_config: &stable_diffusion::StableDiffusionConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<stable_diffusion::unet_2d::UNet2DConditionModel> {
        use candle_core::safetensors::MmapedSafetensors;
        use candle_nn::VarBuilder;

        let st = unsafe { MmapedSafetensors::multi(&[&self.base.paths.transformer])? };

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
        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(MmapBackend { st });
        let wrapped = self.wrap_with_loras(inner)?;
        let vb = VarBuilder::from_backend(wrapped, dtype, device.clone());
        Ok(stable_diffusion::unet_2d::UNet2DConditionModel::new(
            vb,
            4,
            4,
            false,
            mold_candle::stable_diffusion::sd15_unet(),
        )?)
    }

    /// Same as `build_unet_single_file` but with the LoRA wrapper layered on
    /// top of `SingleFileBackend`. The SD1.5 `SingleFileBackend` translates
    /// diffusers candle keys onto the mmap'd A1111 checkpoint; the LoRA
    /// wrapper then intercepts the merged tensor and adds the per-layer
    /// delta before it lands on the UNet constructor.
    fn build_unet_single_file_with_lora(
        &self,
        single_file: &std::path::Path,
        remap: &crate::loader::Sd15Remap,
        _sd_config: &stable_diffusion::StableDiffusionConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<stable_diffusion::unet_2d::UNet2DConditionModel> {
        use crate::loader::SingleFileBackend;
        use candle_nn::VarBuilder;

        let backend = SingleFileBackend::from_sd15_unet(single_file, remap)?;
        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(backend);
        let wrapped = self.wrap_with_loras(inner)?;
        let vb = VarBuilder::from_backend(wrapped, dtype, device.clone());
        Ok(stable_diffusion::unet_2d::UNet2DConditionModel::new(
            vb,
            4,
            4,
            false,
            mold_candle::stable_diffusion::sd15_unet(),
        )?)
    }

    /// Wrap an `inner` SimpleBackend with the SD1.5 LoRA backend. Resolves
    /// the `pending_loras` list into `Sd15LoraSpec`s (parsed-LoRA cache hits
    /// keep adapter parsing cheap across requests).
    fn wrap_with_loras(
        &self,
        inner: Box<dyn candle_nn::var_builder::SimpleBackend>,
    ) -> Result<Box<dyn candle_nn::var_builder::SimpleBackend>> {
        let adapters = super::lora::load_lora_adapters(&self.pending_loras, &self.base.progress)?;
        let specs: Vec<super::lora::Sd15LoraSpec<'_>> = adapters
            .iter()
            .zip(self.pending_loras.iter())
            .map(|(adapter, w)| super::lora::Sd15LoraSpec {
                adapter: adapter.as_ref(),
                scale: w.scale,
                path_hash: super::lora::lora_path_hash(&w.path),
            })
            .collect();
        super::lora::wrap_backend_with_lora(inner, &specs, &self.base.progress, None)
    }

    /// VAE load with the same single-file vs diffusers branch as
    /// `build_unet_for_strategy`. Sequential SD1.5 loads the VAE twice
    /// (once for img2img encode, once for post-denoise decode), so this
    /// helper keeps the branch logic in a single place.
    fn build_vae_for_strategy(
        &self,
        sd_config: &stable_diffusion::StableDiffusionConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<stable_diffusion::vae::AutoEncoderKL> {
        if let Some(single_file) = self.single_file_path.as_ref() {
            let remap = Self::load_sd15_remap(single_file)?;
            Self::build_vae_single_file(single_file, &remap, sd_config, device, dtype)
        } else {
            self.build_vae_diffusers(sd_config, device, dtype)
        }
    }

    #[cfg(test)]
    fn load_vae_cpu_tensors(&self) -> Result<Option<Arc<HashMap<String, Tensor>>>> {
        self.load_vae_cpu_tensors_for_path(&self.base.paths.vae)
    }

    fn load_vae_cpu_tensors_for_path(
        &self,
        vae_path: &Path,
    ) -> Result<Option<Arc<HashMap<String, Tensor>>>> {
        let Some(shared_pool) = &self.shared_pool else {
            return Ok(None);
        };
        shared_pool
            .lock()
            .unwrap()
            .load_safetensors_cpu_tensors(std::slice::from_ref(&vae_path))
    }

    fn load_vae_var_builder<'a>(
        &self,
        vae_path: &Path,
        dtype: DType,
        device: &Device,
        component: &str,
    ) -> Result<candle_nn::VarBuilder<'a>> {
        if let Some(tensors) = self.load_vae_cpu_tensors_for_path(vae_path)? {
            return Ok(crate::encoders::park::varbuilder_from_parked(
                tensors.as_ref(),
                dtype,
                device,
            ));
        }

        crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(&vae_path),
            dtype,
            device,
            component,
            &self.base.progress,
        )
    }

    fn build_vae_diffusers(
        &self,
        _sd_config: &stable_diffusion::StableDiffusionConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<stable_diffusion::vae::AutoEncoderKL> {
        let vb = self.load_vae_var_builder(&self.base.paths.vae, dtype, device, "VAE")?;
        Ok(stable_diffusion::vae::AutoEncoderKL::new(
            vb,
            3,
            3,
            mold_candle::stable_diffusion::vae(),
        )?)
    }

    /// Header-parse the single-file checkpoint and build the SD1.5
    /// diffusers→A1111 remap. Cheap (no tensor data is mmap'd) — sequential
    /// reload calls this each time a component is reloaded after dropping.
    fn load_sd15_remap(single_file: &std::path::Path) -> Result<crate::loader::Sd15Remap> {
        use crate::loader::{build_sd15_remap, single_file as single_file_loader};
        use mold_catalog::families::Family;
        let bundle = single_file_loader::load(single_file, Family::Sd15)
            .map_err(|e| anyhow::anyhow!("partition single-file SD1.5 checkpoint: {e}"))?;
        build_sd15_remap(&bundle)
            .map_err(|e| anyhow::anyhow!("build SD1.5 diffusers→A1111 remap: {e}"))
    }

    /// Build a UNet from a Civitai single-file checkpoint via a UNet-scoped
    /// `SingleFileBackend`. Used by both eager `load_components_single_file`
    /// and sequential `generate_sequential`. SD1.5 component keyspaces are
    /// disjoint (no collision risk), but per-scope factories keep parity
    /// with SDXL's collision-avoidance pattern.
    fn build_unet_single_file(
        single_file: &std::path::Path,
        remap: &crate::loader::Sd15Remap,
        _sd_config: &stable_diffusion::StableDiffusionConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<stable_diffusion::unet_2d::UNet2DConditionModel> {
        use crate::loader::SingleFileBackend;
        use candle_nn::VarBuilder;
        let backend = SingleFileBackend::from_sd15_unet(single_file, remap)?;
        let vb = VarBuilder::from_backend(Box::new(backend), dtype, device.clone());
        Ok(stable_diffusion::unet_2d::UNet2DConditionModel::new(
            vb,
            4,
            4,
            false,
            mold_candle::stable_diffusion::sd15_unet(),
        )?)
    }

    /// Build a VAE from a Civitai single-file checkpoint via a VAE-scoped
    /// `SingleFileBackend`.
    fn build_vae_single_file(
        single_file: &std::path::Path,
        remap: &crate::loader::Sd15Remap,
        _sd_config: &stable_diffusion::StableDiffusionConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<stable_diffusion::vae::AutoEncoderKL> {
        use crate::loader::SingleFileBackend;
        use candle_nn::VarBuilder;
        let backend = SingleFileBackend::from_sd15_vae(single_file, remap)?;
        let vb = VarBuilder::from_backend(Box::new(backend), dtype, device.clone());
        Ok(stable_diffusion::vae::AutoEncoderKL::new(
            vb,
            3,
            3,
            mold_candle::stable_diffusion::vae(),
        )?)
    }

    /// Build CLIP-L from a Civitai single-file checkpoint via a CLIP-L-scoped
    /// `SingleFileBackend`. SD1.5 has only one CLIP-L (no CLIP-G).
    fn build_clip_single_file(
        single_file: &std::path::Path,
        remap: &crate::loader::Sd15Remap,
        clip_config: &stable_diffusion::clip::Config,
        clip_device: &Device,
    ) -> Result<stable_diffusion::clip::ClipTextTransformer> {
        use crate::loader::SingleFileBackend;
        use candle_nn::VarBuilder;
        let backend = SingleFileBackend::from_sd15_clip_l(single_file, remap)?;
        let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, clip_device.clone());
        Ok(stable_diffusion::clip::ClipTextTransformer::new(
            vb,
            clip_config,
        )?)
    }

    /// Load all SD1.5 model components (Eager mode).
    ///
    /// Two materialisation paths share this entry point:
    ///
    /// - **Diffusers-layout** (`single_file_path == None`) — calls candle's
    ///   `build_unet` / `build_vae` / `build_clip_transformer` helpers,
    ///   each of which mmaps a separate component file through the
    ///   default `VarBuilder::from_mmaped_safetensors`.
    /// - **Single-file** (`single_file_path == Some(path)`) — header-parses
    ///   the checkpoint, runs `build_sd15_remap` to produce a
    ///   diffusers→A1111 lookup, and feeds candle's per-component
    ///   constructors a `VarBuilder::from_backend(SingleFileBackend)` that
    ///   translates each `vb.get(diffusers_key)` into an mmap'd A1111 read.
    ///   Three backends share the underlying mmap (one per component VB);
    ///   peak memory is identical to the diffusers path.
    ///
    /// On error, `self.base.loaded` remains `None` — all components are
    /// assembled into local variables and only stored on success, so
    /// partial loads cannot leave the engine in an inconsistent state.
    pub fn load(&mut self) -> Result<()> {
        if self.base.loaded.is_some() {
            return Ok(());
        }

        // Sequential mode defers loading to generate_sequential()
        if self.base.load_strategy == LoadStrategy::Sequential {
            return Ok(());
        }

        let (clip_encoder, clip_tokenizer) = self.validate_paths()?;

        tracing::info!(model = %self.base.model_name, "loading SD1.5 model components...");

        let device = crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)?;
        let dtype = if crate::device::is_gpu(&device) {
            DType::F16
        } else {
            DType::F32
        };

        let sd_config = self.sd_config();

        // Tier 1: honor `placement.text_encoders` override (group knob).
        let tier1 = self
            .pending_placement
            .as_ref()
            .map(|p| p.text_encoders.clone())
            .unwrap_or_default();
        let clip_device = crate::device::resolve_device(Some(tier1), || Ok(device.clone()))?;

        // Resolve VAE precision once at load — see LoadedSD15::vae_dtype.
        let vae_dtype = crate::device::resolve_vae_dtype(dtype);
        let (unet, vae, clip) = if let Some(single_file) = self.single_file_path.clone() {
            self.load_components_single_file(
                &single_file,
                &sd_config,
                &device,
                &clip_device,
                dtype,
                vae_dtype,
            )?
        } else {
            self.load_components_diffusers(
                &clip_encoder,
                &sd_config,
                &device,
                &clip_device,
                dtype,
                vae_dtype,
            )?
        };

        let tokenizer = self.load_clip_tokenizer(&clip_tokenizer)?;

        self.base.loaded = Some(LoadedSD15 {
            unet: Some(unet),
            vae,
            clip,
            tokenizer,
            sd_config,
            device,
            clip_device,
            vae_dtype,
            dtype,
        });

        tracing::info!(model = %self.base.model_name, "all SD1.5 components loaded successfully");
        Ok(())
    }

    /// Diffusers-layout component loader (existing pre-2.6 path).
    #[allow(clippy::too_many_arguments)]
    fn load_components_diffusers(
        &mut self,
        clip_encoder: &std::path::Path,
        sd_config: &stable_diffusion::StableDiffusionConfig,
        device: &Device,
        clip_device: &Device,
        dtype: DType,
        vae_dtype: DType,
    ) -> Result<(
        stable_diffusion::unet_2d::UNet2DConditionModel,
        stable_diffusion::vae::AutoEncoderKL,
        stable_diffusion::clip::ClipTextTransformer,
    )> {
        self.base.progress.stage_start("Loading UNet (GPU)");
        let unet_start = Instant::now();
        let unet = sd_config.build_unet(
            &self.base.paths.transformer,
            device,
            4,     // in_channels
            false, // use_flash_attn
            dtype,
        )?;
        self.base
            .progress
            .stage_done("Loading UNet (GPU)", unet_start.elapsed());

        self.base.progress.stage_start("Loading VAE (GPU)");
        let vae_start = Instant::now();
        let vae = self.build_vae_diffusers(sd_config, device, vae_dtype)?;
        self.base
            .progress
            .stage_done("Loading VAE (GPU)", vae_start.elapsed());

        self.base.progress.stage_start("Loading CLIP-L encoder");
        let clip_start = Instant::now();
        let clip = stable_diffusion::build_clip_transformer(
            &sd_config.clip,
            clip_encoder,
            clip_device,
            DType::F32,
        )?;
        self.base
            .progress
            .stage_done("Loading CLIP-L encoder", clip_start.elapsed());

        Ok((unet, vae, clip))
    }

    /// Single-file (Civitai) component loader.
    ///
    /// Header-parses the checkpoint, builds the diffusers→A1111 remap, wraps
    /// it in a `SingleFileBackend` (which translates diffusers `vb.get(key)`
    /// calls into reads from the mmap'd source tensors), then hands three
    /// `VarBuilder::from_backend(SingleFileBackend)` instances to candle's
    /// per-component constructors. UNet, VAE, and CLIP-L all read from the
    /// same single-file mmap — no on-disk shard files needed.
    ///
    /// Mold-owned component factories reproduce the upstream aggregate
    /// configuration without reaching through Candle's private fields.
    #[allow(clippy::too_many_arguments)]
    fn load_components_single_file(
        &mut self,
        single_file: &std::path::Path,
        sd_config: &stable_diffusion::StableDiffusionConfig,
        device: &Device,
        clip_device: &Device,
        dtype: DType,
        vae_dtype: DType,
    ) -> Result<(
        stable_diffusion::unet_2d::UNet2DConditionModel,
        stable_diffusion::vae::AutoEncoderKL,
        stable_diffusion::clip::ClipTextTransformer,
    )> {
        let remap = Self::load_sd15_remap(single_file)?;

        self.base.progress.stage_start("Loading UNet (single-file)");
        let unet_start = Instant::now();
        let unet = Self::build_unet_single_file(single_file, &remap, sd_config, device, dtype)?;
        self.base
            .progress
            .stage_done("Loading UNet (single-file)", unet_start.elapsed());

        self.base.progress.stage_start("Loading VAE (single-file)");
        let vae_start = Instant::now();
        let vae = Self::build_vae_single_file(single_file, &remap, sd_config, device, vae_dtype)?;
        self.base
            .progress
            .stage_done("Loading VAE (single-file)", vae_start.elapsed());

        self.base
            .progress
            .stage_start("Loading CLIP-L (single-file)");
        let clip_start = Instant::now();
        let clip = Self::build_clip_single_file(single_file, &remap, &sd_config.clip, clip_device)?;
        self.base
            .progress
            .stage_done("Loading CLIP-L (single-file)", clip_start.elapsed());

        Ok((unet, vae, clip))
    }

    /// Tokenize a prompt for the CLIP encoder, padding/truncating to max_len tokens.
    fn tokenize(
        tokenizer: &tokenizers::Tokenizer,
        prompt: &str,
        max_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(max_len);
        // Pad with 0s (EOS/PAD token for CLIP)
        while ids.len() < max_len {
            ids.push(0);
        }
        let ids = ids.into_iter().map(|i| i as i64).collect::<Vec<_>>();
        Ok(Tensor::new(ids, device)?.unsqueeze(0)?)
    }

    /// Run the denoising loop (shared between eager and sequential).
    ///
    /// `start_step` allows starting from a later timestep for img2img (0 = full txt2img).
    #[allow(clippy::too_many_arguments)]
    fn denoise_loop(
        &self,
        unet: &stable_diffusion::unet_2d::UNet2DConditionModel,
        text_embeddings: &Tensor,
        sched: Scheduler,
        latents: &mut Tensor,
        guidance: f64,
        cfg_plus: bool,
        steps: u32,
        start_step: usize,
        inpaint_ctx: Option<&crate::img_utils::InpaintContext>,
        controlnet_ctx: Option<&ControlNetContext>,
    ) -> Result<()> {
        let use_cfg = cfg_active(guidance);
        let mut scheduler = crate::scheduler::build_scheduler(
            sched,
            steps as usize,
            PredictionType::Epsilon,
            false,
        )?;
        let timesteps = scheduler.timesteps().to_vec();
        let active_timesteps = &timesteps[start_step..];

        // CFG++ requires the doubled `[uncond, cond]` forward AND a DDIM
        // scheduler — see the SDXL counterpart for the full rationale.
        let cfg_plus_schedule = if cfg_plus && use_cfg && matches!(sched, Scheduler::Ddim) {
            Some(DdimAlphaSchedule::from_default(steps as usize))
        } else {
            if cfg_plus && !use_cfg {
                tracing::warn!(
                    guidance,
                    "cfg_plus requested but cfg_scale ≈ 1.0 — falling back to standard step (no uncond available)"
                );
            } else if cfg_plus {
                tracing::warn!(
                    scheduler = ?sched,
                    "cfg_plus requested but only DDIM is supported on SDXL/SD1.5 — falling back to standard step. Re-run with `--scheduler ddim` to enable CFG++."
                );
            }
            None
        };

        let denoise_label = format!("Denoising ({} steps)", active_timesteps.len());
        self.base.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        for (step_idx, &t) in active_timesteps.iter().enumerate() {
            self.base.progress.checkpoint()?;
            let step_start = std::time::Instant::now();
            let latent_input = if use_cfg {
                Tensor::cat(&[&*latents, &*latents], 0)?
            } else {
                latents.clone()
            };

            let latent_input = scheduler.scale_model_input(latent_input, t)?;

            let noise_pred = if let Some(cn_ctx) = controlnet_ctx {
                let (down_residuals, mid_residual) = cn_ctx.model.forward(
                    &latent_input,
                    t as f64,
                    text_embeddings,
                    &cn_ctx.control_tensor,
                    cn_ctx.scale,
                )?;
                unet.forward_with_additional_residuals(
                    &latent_input,
                    t as f64,
                    text_embeddings,
                    Some(&down_residuals),
                    Some(&mid_residual),
                )?
            } else {
                unet.forward(&latent_input, t as f64, text_embeddings)?
            };

            // Hold onto the raw uncond row when CFG++ is active so we can use
            // it as the renoise direction below.
            let (noise_pred_blended, noise_pred_uncond_opt) = if use_cfg {
                let chunks = noise_pred.chunk(2, 0)?;
                let noise_pred_uncond = chunks[0].clone();
                let noise_pred_cond = &chunks[1];
                let blended =
                    (&noise_pred_uncond + ((noise_pred_cond - &noise_pred_uncond)? * guidance)?)?;
                (blended, Some(noise_pred_uncond))
            } else {
                (noise_pred, None)
            };

            *latents = match (cfg_plus_schedule.as_ref(), noise_pred_uncond_opt.as_ref()) {
                (Some(ddim_sched), Some(eps_uncond)) => {
                    ddim_sched.cfg_plus_step(&*latents, &noise_pred_blended, eps_uncond, t)?
                }
                _ => scheduler.step(&noise_pred_blended, t, &*latents)?,
            };

            // Inpainting: blend preserved regions back at current noise level
            if let Some(ctx) = inpaint_ctx {
                let noised_original =
                    scheduler.add_noise(&ctx.original_latents, ctx.noise.clone(), t)?;
                *latents = crate::img2img::blend_inpaint_latents(&*latents, ctx, &noised_original)?;
            }

            self.base.progress.emit(ProgressEvent::DenoiseStep {
                step: step_idx + 1,
                total: active_timesteps.len(),
                elapsed: step_start.elapsed(),
            });
        }

        self.base.progress.checkpoint()?;
        self.base
            .progress
            .stage_done(&denoise_label, denoise_start.elapsed());
        Ok(())
    }

    /// Prepare img2img latents: VAE encode source image, add noise at the appropriate timestep.
    /// Returns (noised_latents, start_step, encoded_latents, noise) -- extra fields for inpainting.
    ///
    /// `dtype` drives the noise/denoise loop; `vae_dtype` may differ when
    /// `MOLD_VAE_DTYPE` forces fp32 (encoded latents are cast back to `dtype`
    /// before returning so the denoise loop runs at engine precision).
    #[allow(clippy::too_many_arguments)]
    fn prepare_img2img_latents(
        &self,
        vae: &stable_diffusion::vae::AutoEncoderKL,
        source_bytes: &[u8],
        width: u32,
        height: u32,
        strength: f64,
        steps: u32,
        sched: Scheduler,
        seed: u64,
        device: &Device,
        dtype: DType,
        vae_dtype: DType,
    ) -> Result<(Tensor, usize, Tensor, Tensor)> {
        use crate::img_utils::{decode_source_image, NormalizeRange};
        let cache_key = image_size_cache_key(source_bytes, width, height);
        let (encoded, cache_hit) = get_or_insert_cached_tensor(
            &self.source_latent_cache,
            cache_key,
            device,
            dtype,
            || {
                self.base
                    .progress
                    .stage_start("Encoding source image (VAE)");
                let encode_start = Instant::now();

                let source_tensor = decode_source_image(
                    source_bytes,
                    width,
                    height,
                    NormalizeRange::MinusOneToOne,
                    device,
                    vae_dtype,
                )?;

                let encoded = vae.encode(&source_tensor)?;
                let encoded = (encoded.mode()? * VAE_SCALE)?;
                // Cast back to engine dtype so the denoise loop stays at its
                // natural precision when MOLD_VAE_DTYPE upgraded the VAE.
                let encoded = encoded.to_dtype(dtype)?;

                self.base.progress.phase_done(
                    crate::ProgressPhase::Vae,
                    "Encoding source image (VAE)",
                    encode_start.elapsed(),
                );
                Ok(encoded)
            },
        )?;
        if cache_hit {
            self.base.progress.cache_hit("source image latents");
        }

        let start_step = crate::img2img::img2img_start_index(steps as usize, strength);

        // Build scheduler to get timesteps and add noise
        let scheduler = crate::scheduler::build_scheduler(
            sched,
            steps as usize,
            PredictionType::Epsilon,
            false,
        )?;
        let timesteps = scheduler.timesteps().to_vec();

        let latent_h = height as usize / 8;
        let latent_w = width as usize / 8;
        let noise =
            crate::engine::seeded_randn(seed, &[1, 4, latent_h, latent_w], device, DType::F32)?;
        let noise = noise.to_dtype(dtype)?;

        // Add noise at the start timestep
        let noised = if start_step < timesteps.len() {
            scheduler.add_noise(&encoded, noise.clone(), timesteps[start_step])?
        } else {
            encoded.clone()
        };

        tracing::info!(
            start_step,
            total_steps = steps,
            strength,
            "img2img: starting from step {start_step}"
        );

        Ok((noised, start_step, encoded, noise))
    }

    /// Encode prompt with the single CLIP-L encoder.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn encode_prompt(
        &self,
        clip: &stable_diffusion::clip::ClipTextTransformer,
        tokenizer: &tokenizers::Tokenizer,
        prompt: &str,
        negative_prompt: &str,
        max_len: usize,
        device: &Device,
        clip_device: &Device,
        dtype: DType,
        guidance: f64,
    ) -> Result<Tensor> {
        // SD1.5 caches the **concatenated** `(uncond, cond)` tensor when CFG is
        // active, so the cache key must include the negative prompt and the
        // guidance scale. Keying on the positive prompt + guidance alone
        // returned a stale uncond branch when the user changed only the
        // negative prompt — silent wrong output. Mirrors the SD3 / SDXL fix.
        let cache_key = cfg_prompt_cache_key(prompt, negative_prompt, guidance);
        let (text_embeddings, cache_hit) =
            get_or_insert_cached_tensor(&self.prompt_cache, cache_key, device, dtype, || {
                let use_cfg = cfg_active(guidance);

                self.base.progress.stage_start("Encoding prompt (CLIP-L)");
                let encode_start = Instant::now();
                let tokens = Self::tokenize(tokenizer, prompt, max_len, clip_device)?;
                let text_embeddings = clip.forward(&tokens)?;
                self.base.progress.phase_done(
                    crate::ProgressPhase::PromptEncode,
                    "Encoding prompt (CLIP-L)",
                    encode_start.elapsed(),
                );

                let text_embeddings = if use_cfg {
                    let uncond_tokens =
                        Self::tokenize(tokenizer, negative_prompt, max_len, clip_device)?;
                    let uncond_embeddings = clip.forward(&uncond_tokens)?;
                    Tensor::cat(&[&uncond_embeddings, &text_embeddings], 0)?
                } else {
                    text_embeddings
                };

                let text_embeddings = text_embeddings.to_device(device)?;
                Ok(text_embeddings.to_dtype(dtype)?)
            })?;
        if cache_hit {
            self.base.progress.cache_hit("prompt conditioning");
            return Ok(text_embeddings);
        }
        Ok(text_embeddings)
    }

    fn cached_mask(
        &self,
        mask_bytes: &[u8],
        latent_h: usize,
        latent_w: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let key = latent_size_cache_key(mask_bytes, latent_h, latent_w);
        let (mask, cache_hit) =
            get_or_insert_cached_tensor(&self.mask_cache, key, device, dtype, || {
                crate::img_utils::decode_mask_image(mask_bytes, latent_h, latent_w, device, dtype)
            })?;
        if cache_hit {
            self.base.progress.cache_hit("inpaint mask");
            return Ok(mask);
        }
        Ok(mask)
    }

    /// Load ControlNet model and prepare control tensor from the request.
    fn load_controlnet(
        &self,
        req: &GenerateRequest,
        device: &Device,
        dtype: DType,
    ) -> Result<Option<ControlNetContext>> {
        let (control_bytes, control_model_name, scale) =
            match (req.control_image.as_ref(), req.control_model.as_ref()) {
                (Some(bytes), Some(name)) => (bytes, name, req.control_scale),
                _ => return Ok(None),
            };

        self.base.progress.stage_start("Loading ControlNet");
        let cn_start = Instant::now();

        // Resolve ControlNet model weights path
        let config = mold_core::Config::load_or_default();
        let cn_paths =
            mold_core::ModelPaths::resolve(control_model_name, &config).ok_or_else(|| {
                anyhow::anyhow!(
                    "ControlNet model '{}' not found. Pull it with: mold pull {}",
                    control_model_name,
                    control_model_name,
                )
            })?;

        // Use default SD1.5 UNet config for ControlNet architecture
        let unet_config =
            candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModelConfig::default();
        let model = ControlNetModel::load(
            &cn_paths.transformer,
            device,
            dtype,
            unet_config,
            &self.base.progress,
        )?;

        self.base
            .progress
            .stage_done("Loading ControlNet", cn_start.elapsed());

        // Decode and preprocess control image
        self.base
            .progress
            .stage_start("Preprocessing control image");
        let preprocess_start = Instant::now();

        let control_key = image_size_cache_key(control_bytes, req.width, req.height);
        let (control_tensor, cache_hit) = get_or_insert_cached_tensor(
            &self.control_tensor_cache,
            control_key,
            device,
            dtype,
            || {
                crate::img_utils::decode_source_image(
                    control_bytes,
                    req.width,
                    req.height,
                    crate::img_utils::NormalizeRange::ZeroToOne,
                    device,
                    dtype,
                )
            },
        )?;
        if cache_hit {
            self.base.progress.cache_hit("control preprocessing");
        }

        self.base
            .progress
            .stage_done("Preprocessing control image", preprocess_start.elapsed());

        tracing::info!(
            control_model = %control_model_name,
            scale,
            "ControlNet loaded and control image preprocessed"
        );

        Ok(Some(ControlNetContext {
            model,
            control_tensor,
            scale,
        }))
    }

    /// Generate an image using sequential loading strategy.
    ///
    /// Loads components one at a time and drops them when done:
    /// 1. Load CLIP-L → encode → drop CLIP-L
    /// 2. Load UNet → denoise → drop UNet
    /// 3. Load VAE → decode → drop VAE
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let (clip_encoder, clip_tokenizer) = self.validate_paths()?;

        // Check memory budget
        if let Some(warning) = check_memory_budget(&self.base.paths, LoadStrategy::Sequential) {
            self.base.progress.info(&warning);
        }

        let device = crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)?;
        let dtype = if crate::device::is_gpu(&device) {
            DType::F16
        } else {
            DType::F32
        };

        let sd_config = self.sd_config();
        let max_len = sd_config.clip.max_position_embeddings;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;
        let guidance = req.guidance;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            guidance,
            "starting sequential SD1.5 generation"
        );

        self.base
            .progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: Encode prompt (check cache first to skip encoder load) ---
        let neg = req.negative_prompt.as_deref().unwrap_or("");
        let cache_key = cfg_prompt_cache_key(&req.prompt, neg, guidance);
        let text_embeddings = if let Some(tensor) =
            restore_cached_tensor(&self.prompt_cache, &cache_key, &device, dtype)?
        {
            self.base.progress.cache_hit("prompt conditioning");
            tensor
        } else {
            if let Some(status) = memory_status_string() {
                self.base.progress.info(&status);
            }

            let tokenizer = self.load_clip_tokenizer(&clip_tokenizer)?;

            let tier1 = self
                .pending_placement
                .as_ref()
                .map(|p| p.text_encoders.clone())
                .unwrap_or_default();
            let clip_device = crate::device::resolve_device(Some(tier1), || Ok(device.clone()))?;
            // Branch on single_file_path so cv:<id>-style Civitai checkpoints
            // (A1111 naming) go through `SingleFileBackend` just like eager
            // mode does. Without this, candle's diffusers-keyed
            // `build_clip_transformer` errors with
            // "cannot find tensor text_model.embeddings.token_embedding.weight".
            let clip = if let Some(single_file) = self.single_file_path.clone() {
                let remap = Self::load_sd15_remap(&single_file)?;
                self.base
                    .progress
                    .stage_start("Loading CLIP-L (single-file)");
                let clip_start = Instant::now();
                let clip = Self::build_clip_single_file(
                    &single_file,
                    &remap,
                    &sd_config.clip,
                    &clip_device,
                )?;
                self.base
                    .progress
                    .stage_done("Loading CLIP-L (single-file)", clip_start.elapsed());
                clip
            } else {
                self.base.progress.stage_start("Loading CLIP-L encoder");
                let clip_start = Instant::now();
                let clip = stable_diffusion::build_clip_transformer(
                    &sd_config.clip,
                    &clip_encoder,
                    &clip_device,
                    DType::F32,
                )?;
                self.base
                    .progress
                    .stage_done("Loading CLIP-L encoder", clip_start.elapsed());
                clip
            };

            let text_embeddings = self.encode_prompt(
                &clip,
                &tokenizer,
                &req.prompt,
                neg,
                max_len,
                &device,
                &clip_device,
                dtype,
                guidance,
            )?;

            drop(clip);
            self.base.progress.info("Freed CLIP-L encoder");
            tracing::info!("CLIP encoder dropped (sequential mode)");

            text_embeddings
        };

        // --- Phase 2: Load UNet and denoise ---
        let unet_size = std::fs::metadata(&self.base.paths.transformer)
            .map(|m| m.len())
            .unwrap_or(0);
        // SD1.5 runs CFG by default → batch=2 unless guidance ≈ 1 (LCM).
        let unet_batch = if req.guidance > 1.0 { 2 } else { 1 };
        let unet_activation_budget = crate::device::activation_bytes(
            req.width,
            req.height,
            unet_batch,
            crate::device::dtype_bytes(dtype),
            crate::device::ActivationFamily::SdxlUnet,
        );
        preflight_memory_check("UNet", unet_size, unet_activation_budget)?;
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        self.base.progress.stage_start("Loading UNet (GPU)");
        let unet_start = Instant::now();
        let unet = self.build_unet_for_strategy(&sd_config, &device, dtype)?;
        self.base
            .progress
            .stage_done("Loading UNet (GPU)", unet_start.elapsed());

        let sched = req.scheduler.unwrap_or(self.scheduler);
        let is_img2img = req.source_image.is_some();

        // For img2img in sequential mode, we need the VAE for encoding before the UNet
        let (mut latents, start_step, inpaint_ctx) = if let Some(ref source_bytes) =
            req.source_image
        {
            self.base
                .progress
                .info("img2img mode: encoding source image before denoising");

            // Load VAE first for encoding
            self.base.progress.stage_start("Loading VAE (GPU)");
            let vae_start = Instant::now();
            let vae_dtype = crate::device::resolve_vae_dtype(dtype);
            let vae = self.build_vae_for_strategy(&sd_config, &device, vae_dtype)?;
            self.base
                .progress
                .stage_done("Loading VAE (GPU)", vae_start.elapsed());

            let (latents, start_step, encoded, noise) = self.prepare_img2img_latents(
                &vae,
                source_bytes,
                req.width,
                req.height,
                req.strength,
                req.steps,
                sched,
                seed,
                &device,
                dtype,
                vae_dtype,
            )?;

            // Drop VAE to free memory for UNet
            let inpaint_ctx = if let Some(ref mask_bytes) = req.mask_image {
                let mask = self.cached_mask(mask_bytes, height / 8, width / 8, &device, dtype)?;
                Some(crate::img_utils::InpaintContext {
                    original_latents: encoded,
                    mask,
                    noise,
                })
            } else {
                None
            };

            drop(vae);
            self.base
                .progress
                .info("Freed VAE (will reload for decode)");
            device.synchronize()?;

            (latents, start_step, inpaint_ctx)
        } else {
            let latent_h = height / 8;
            let latent_w = width / 8;
            let init_scheduler = crate::scheduler::build_scheduler(
                sched,
                req.steps as usize,
                PredictionType::Epsilon,
                false,
            )?;
            let init_noise_sigma = init_scheduler.init_noise_sigma();
            drop(init_scheduler);
            let latents = (crate::engine::seeded_randn(
                seed,
                &[1, 4, latent_h, latent_w],
                &device,
                DType::F32,
            )? * init_noise_sigma)?;
            (latents.to_dtype(dtype)?, 0, None)
        };

        // Load ControlNet if requested
        let controlnet_ctx = self.load_controlnet(req, &device, dtype)?;

        self.denoise_loop(
            &unet,
            &text_embeddings,
            sched,
            &mut latents,
            guidance,
            resolve_cfg_plus(req),
            req.steps,
            start_step,
            inpaint_ctx.as_ref(),
            controlnet_ctx.as_ref(),
        )?;

        // Drop UNet to free memory for VAE decode
        drop(controlnet_ctx);
        drop(inpaint_ctx);
        drop(unet);
        drop(text_embeddings);
        device.synchronize()?;
        self.base.progress.info("Freed UNet");
        tracing::info!("UNet dropped (sequential mode)");

        // --- Phase 3: Load VAE and decode ---
        // (img2img already loaded+dropped VAE above; reload for decode)
        let vae_load_label = if is_img2img {
            "Reloading VAE (GPU)"
        } else {
            "Loading VAE (GPU)"
        };
        self.base.progress.stage_start(vae_load_label);
        let vae_start = Instant::now();
        let vae_dtype = crate::device::resolve_vae_dtype(dtype);
        let vae = self.build_vae_for_strategy(&sd_config, &device, vae_dtype)?;
        self.base
            .progress
            .stage_done(vae_load_label, vae_start.elapsed());

        self.base.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();

        let latents = (latents / VAE_SCALE)?;
        let latents_for_vae = latents.to_dtype(vae_dtype)?;
        let device_for_sync = device.clone();
        let img = crate::vae_tiling::decode_with_oom_fallback(
            &latents_for_vae,
            |t| vae.decode(t).map_err(Into::into),
            || {
                if let Err(e) = device_for_sync.synchronize() {
                    tracing::warn!(
                        "SD1.5 (sequential) device.synchronize() after VAE OOM failed: {e}"
                    );
                }
            },
        )?;

        let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
        let img = (img * 255.)?.to_dtype(DType::U8)?;
        let img = img.squeeze(0)?;

        self.base.progress.phase_done(
            crate::ProgressPhase::Vae,
            "VAE decode",
            vae_decode_start.elapsed(),
        );

        let output_metadata = build_output_metadata(req, seed, Some(sched));
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
            "sequential SD1.5 generation complete"
        );

        Ok(GenerateResponse {
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

impl SD15Engine {
    fn generate_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        // Sequential mode: load-use-drop each component
        if self.base.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Eager mode: if the requested LoRA stack differs from what's merged
        // into the loaded UNet, drop it now so `reload_unet_if_needed` rebuilds
        // it with the new wrapper.
        let requested_stack = lora_stack_fingerprint(&self.pending_loras);
        if requested_stack != self.active_lora_fingerprint {
            if let Some(loaded) = self.base.loaded.as_mut() {
                if loaded.unet.is_some() {
                    loaded.unet = None;
                    loaded.device.synchronize()?;
                    tracing::info!("SD1.5 UNet dropped (LoRA stack changed)");
                }
            }
            self.active_lora_fingerprint = requested_stack;
        }

        // Eager mode: reload UNet if dropped after previous VAE decode
        self.reload_unet_if_needed()?;

        let loaded = self
            .base
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;
        let guidance = req.guidance;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            guidance,
            scheduler = %self.scheduler,
            "starting SD1.5 generation"
        );

        // 1. Encode prompt with CLIP-L
        let max_len = loaded.sd_config.clip.max_position_embeddings;
        let neg = req.negative_prompt.as_deref().unwrap_or("");
        let text_embeddings = self.encode_prompt(
            &loaded.clip,
            &loaded.tokenizer,
            &req.prompt,
            neg,
            max_len,
            &loaded.device,
            &loaded.clip_device,
            loaded.dtype,
            guidance,
        )?;

        // 2. Build scheduler and create initial latents
        let sched = req.scheduler.unwrap_or(self.scheduler);

        let (mut latents, start_step, inpaint_ctx) =
            if let Some(ref source_bytes) = req.source_image {
                let (latents, start_step, encoded, noise) = self.prepare_img2img_latents(
                    &loaded.vae,
                    source_bytes,
                    req.width,
                    req.height,
                    req.strength,
                    req.steps,
                    sched,
                    seed,
                    &loaded.device,
                    loaded.dtype,
                    loaded.vae_dtype,
                )?;
                let inpaint_ctx = if let Some(ref mask_bytes) = req.mask_image {
                    let mask = self.cached_mask(
                        mask_bytes,
                        height / 8,
                        width / 8,
                        &loaded.device,
                        loaded.dtype,
                    )?;
                    Some(crate::img_utils::InpaintContext {
                        original_latents: encoded,
                        mask,
                        noise,
                    })
                } else {
                    None
                };
                (latents, start_step, inpaint_ctx)
            } else {
                let latent_h = height / 8;
                let latent_w = width / 8;
                let init_scheduler = crate::scheduler::build_scheduler(
                    sched,
                    req.steps as usize,
                    PredictionType::Epsilon,
                    false,
                )?;
                let init_noise_sigma = init_scheduler.init_noise_sigma();
                drop(init_scheduler);
                let latents = (crate::engine::seeded_randn(
                    seed,
                    &[1, 4, latent_h, latent_w],
                    &loaded.device,
                    DType::F32,
                )? * init_noise_sigma)?;
                (latents.to_dtype(loaded.dtype)?, 0, None)
            };

        // 3. Load ControlNet if requested
        let controlnet_ctx = self.load_controlnet(req, &loaded.device, loaded.dtype)?;

        // 4. Denoising loop
        let unet = loaded
            .unet
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("UNet not loaded"))?;
        self.denoise_loop(
            unet,
            &text_embeddings,
            sched,
            &mut latents,
            guidance,
            resolve_cfg_plus(req),
            req.steps,
            start_step,
            inpaint_ctx.as_ref(),
            controlnet_ctx.as_ref(),
        )?;

        // Drop UNet before VAE decode to free VRAM for conv2d intermediates.
        drop(controlnet_ctx);
        drop(inpaint_ctx);
        // End the immutable borrow of `loaded` so we can take a mutable one.
        let _ = loaded;
        let loaded = self.base.loaded.as_mut().unwrap();
        loaded.unet = None;
        loaded.device.synchronize()?;
        tracing::info!("UNet dropped to free VRAM for VAE decode");
        // Re-borrow as immutable for VAE decode + output encoding.
        let _ = loaded;
        let loaded = self.base.loaded.as_ref().unwrap();

        // 5. VAE decode
        self.base.progress.stage_start("VAE decode");
        let vae_start = Instant::now();

        let latents = (latents / VAE_SCALE)?;
        let latents_for_vae = latents.to_dtype(loaded.vae_dtype)?;
        let vae = &loaded.vae;
        let device_for_sync = loaded.device.clone();
        let img = crate::vae_tiling::decode_with_oom_fallback(
            &latents_for_vae,
            |t| vae.decode(t).map_err(Into::into),
            || {
                if let Err(e) = device_for_sync.synchronize() {
                    tracing::warn!(
                        "SD1.5 (parallel) device.synchronize() after VAE OOM failed: {e}"
                    );
                }
            },
        )?;

        let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
        let img = (img * 255.)?.to_dtype(DType::U8)?;
        let img = img.squeeze(0)?;

        self.base
            .progress
            .phase_done(crate::ProgressPhase::Vae, "VAE decode", vae_start.elapsed());

        // 5. Encode to image format
        let output_metadata = build_output_metadata(req, seed, Some(sched));
        let image_bytes = encode_image(
            &img,
            req.resolved_output_format(),
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "SD1.5 generation complete");

        Ok(GenerateResponse {
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

impl InferenceEngine for SD15Engine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.base.progress.checkpoint()?;
        self.pending_placement = req.placement.clone();
        self.pending_loras = super::lora::effective_sd15_loras(req);
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
        SD15Engine::load(self)
    }

    fn load_for_request(&mut self, req: &GenerateRequest) -> Result<()> {
        self.pending_placement = req.placement.clone();
        let result = SD15Engine::load(self);
        self.pending_placement = None;
        result
    }

    fn unload(&mut self) {
        self.base.unload();
        clear_cache(&self.prompt_cache);
        clear_cache(&self.source_latent_cache);
        clear_cache(&self.mask_cache);
        clear_cache(&self.control_tensor_cache);
        self.active_lora_fingerprint.clear();
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
        crate::batch_execution_capability_for_family("sd15")
            .expect("production SD1.5 batch capability must be registered")
    }

    fn model_paths(&self) -> Option<&mold_core::ModelPaths> {
        Some(&self.base.paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::InferenceEngine;
    use crate::shared_pool::SharedPool;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use tokenizers::models::bpe::BPE;

    /// Synthesise a minimal SD1.5-shaped single-file safetensors with one
    /// representative key per component bucket. Tensor data is one zero
    /// F32 — `loader::single_file::load` is header-only and the
    /// constructor doesn't materialise weights.
    fn synth_sd15_single_file(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mold-sd15-from-sf-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        let keys: &[&str] = &[
            // UNet
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight",
            // VAE
            "first_stage_model.encoder.down.0.block.0.norm1.weight",
            "first_stage_model.quant_conv.weight",
            // CLIP-L
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "cond_stage_model.transformer.text_model.final_layer_norm.weight",
        ];

        let f32_zero = 0.0f32.to_le_bytes().to_vec();
        let buffers: Vec<Vec<u8>> = keys.iter().map(|_| f32_zero.clone()).collect();
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (key, buf) in keys.iter().zip(buffers.iter()) {
            tensors.insert(
                (*key).to_string(),
                TensorView::new(SafeDtype::F32, vec![1], buf).unwrap(),
            );
        }
        serialize_to_file(&tensors, &None, &path).unwrap();
        path
    }

    #[test]
    fn from_single_file_constructs_for_synthetic_sd15_checkpoint() {
        let single_file = synth_sd15_single_file("ok");
        // The companion tokenizer is pulled separately and is
        // not validated by the constructor — its path is just stored.
        let tokenizer_path = std::env::temp_dir().join("mold-sd15-tok-stub.json");

        let engine = SD15Engine::from_single_file(
            "dreamshaper-8".to_string(),
            single_file.clone(),
            tokenizer_path,
            Scheduler::default(),
            LoadStrategy::Eager,
            0,
            None,
        )
        .expect("constructor must accept a valid SD1.5 single-file layout");

        assert_eq!(engine.model_name(), "dreamshaper-8");
        assert_eq!(
            engine.single_file_path.as_deref(),
            Some(single_file.as_path()),
            "single-file path must be stashed for the future load() branch",
        );
        assert!(
            !engine.is_loaded(),
            "constructor must not eagerly materialise model weights",
        );

        let _ = std::fs::remove_file(single_file);
    }

    #[test]
    fn sd15_loads_clip_tokenizer_through_shared_pool() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer_path = dir.path().join("clip-tokenizer.json");
        tokenizers::Tokenizer::new(BPE::default())
            .save(&tokenizer_path, false)
            .unwrap();
        let weights_path = dir.path().join("weights.safetensors");
        std::fs::write(&weights_path, b"stub").unwrap();

        let shared_pool = Arc::new(Mutex::new(SharedPool::new()));
        let pooled = shared_pool
            .lock()
            .unwrap()
            .load_tokenizer(&tokenizer_path)
            .unwrap();

        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: weights_path.clone(),
            transformer_shards: Vec::new(),
            vae: weights_path.clone(),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: Some(weights_path),
            t5_tokenizer: None,
            clip_tokenizer: Some(tokenizer_path.clone()),
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };
        let engine = SD15Engine::new(
            "sd15-test".to_string(),
            paths,
            Scheduler::default(),
            LoadStrategy::Eager,
            0,
            Some(shared_pool),
        );

        let loaded = engine.load_clip_tokenizer(&tokenizer_path).unwrap();

        assert!(Arc::ptr_eq(&pooled, &loaded));
    }

    #[test]
    fn sd15_loads_vae_tensors_through_shared_pool() {
        let dir = tempfile::tempdir().unwrap();
        let vae_path = dir.path().join("vae.safetensors");
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

        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: dir.path().join("unet.safetensors"),
            transformer_shards: Vec::new(),
            vae: vae_path.clone(),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: Some(dir.path().join("clip.safetensors")),
            t5_tokenizer: None,
            clip_tokenizer: Some(dir.path().join("clip-tokenizer.json")),
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };
        let engine = SD15Engine::new(
            "sd15-test".to_string(),
            paths,
            Scheduler::default(),
            LoadStrategy::Eager,
            0,
            Some(shared_pool),
        );

        let loaded = engine.load_vae_cpu_tensors().unwrap().unwrap();

        assert!(Arc::ptr_eq(&pooled, &loaded));
    }

    #[test]
    fn load_branches_to_single_file_path_and_invokes_candle_constructors() {
        // Smoke test: `load()` must dispatch to the single-file
        // branch, hand a `VarBuilder::from_backend(SingleFileBackend)` to
        // candle's `UNet2DConditionModel::new`, and surface an error
        // sourced from the SingleFileBackend's tensor lookup (because the
        // synthetic checkpoint doesn't carry the full SD1.5 tensor set).
        // This confirms the wiring without paying the real-shape model
        // construction cost. The full real-shape `load().unwrap()` smoke
        // runs in 2.10's <gpu-host> UAT against a real DreamShaper
        // checkpoint.
        let single_file = synth_sd15_single_file("load-branch");
        let tokenizer_stub = std::env::temp_dir().join(format!(
            "mold-sd15-tok-stub-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::write(&tokenizer_stub, b"").unwrap();

        let mut engine = SD15Engine::from_single_file(
            "dreamshaper-8".to_string(),
            single_file.clone(),
            tokenizer_stub.clone(),
            Scheduler::Ddim,
            LoadStrategy::Eager,
            0,
            None,
        )
        .expect("constructor");

        std::env::set_var("MOLD_DEVICE", "cpu");
        let err = SD15Engine::load(&mut engine).expect_err(
            "synthetic single-file checkpoint can't satisfy SD1.5's full tensor set; \
             dispatch should land in the candle constructor and fail there",
        );
        std::env::remove_var("MOLD_DEVICE");

        let msg = err.to_string();
        // Proof we got past the dispatch: the error originates from the
        // single-file remap / SingleFileBackend layer, not from the
        // diffusers loader (which would say "transformer/diffusion_pytorch_model.safetensors")
        // or from the dispatch itself (which has no error surface — it
        // just calls `load_components_single_file`).
        assert!(
            msg.contains("single-file") || msg.contains("rename rule"),
            "expected error from the single-file load layer, got: {msg}",
        );

        let _ = std::fs::remove_file(single_file);
        let _ = std::fs::remove_file(tokenizer_stub);
    }

    /// Sequential single-file dispatch must hand CLIP-L construction to
    /// `SingleFileBackend` (which translates diffusers `text_model.X` keys
    /// through `Sd15Remap` into A1111 reads), NOT to candle's diffusers-keyed
    /// `build_clip_transformer`. Same contract as the SDXL sibling.
    ///
    /// Locks the SD1.5 half of bug 1 from
    /// `tasks/catalog-run-bridge-option-c-handoff.md`.
    #[test]
    fn build_clip_single_file_dispatches_through_backend_not_diffusers_loader() {
        let single_file = synth_sd15_single_file("seq-clip-dispatch");
        let remap = SD15Engine::load_sd15_remap(&single_file).expect("remap");

        let result = SD15Engine::build_clip_single_file(
            &single_file,
            &remap,
            &stable_diffusion::clip::Config::v1_5(),
            &Device::Cpu,
        );

        let err = result.expect_err("synthetic CLIP-L is incomplete");
        let msg = err.to_string();
        assert!(
            !msg.contains("cannot find tensor text_model"),
            "expected failure from SingleFileBackend ('no rename rule for diffusers key …'), \
             not from candle's diffusers `from_mmaped_safetensors`. Sequential dispatch \
             is still routing through `build_clip_transformer`. Got: {msg}",
        );

        let _ = std::fs::remove_file(single_file);
    }

    /// UNet sequential dispatch parity. SD1.5 sibling of the SDXL UNet
    /// dispatch test.
    #[test]
    fn build_unet_single_file_dispatches_through_backend_not_diffusers_loader() {
        let single_file = synth_sd15_single_file("seq-unet-dispatch");
        let remap = SD15Engine::load_sd15_remap(&single_file).expect("remap");

        let result = SD15Engine::build_unet_single_file(
            &single_file,
            &remap,
            &stable_diffusion::StableDiffusionConfig::v1_5(None, None, None),
            &Device::Cpu,
            DType::F32,
        );

        let err = result.expect_err("synthetic UNet is incomplete");
        let msg = err.to_string();
        assert!(
            !msg.contains("cannot find tensor conv_in"),
            "expected failure from SingleFileBackend, not diffusers loader. Got: {msg}",
        );

        let _ = std::fs::remove_file(single_file);
    }

    /// Real-shape SD1.5 `load()` smoke — full UNet + VAE + CLIP-L
    /// construction against a synthetic checkpoint. Demoted to
    /// `#[ignore]` per the 2.6 handoff: ~3.4 GB of f32 zeros at
    /// SD1.5-correct shapes is over the test budget, and the construction
    /// path also can't run today because of the candle-mold private-field
    /// blocker (see `load_components_single_file`). Kept as documentation
    /// of the eventual UAT contract — 2.10's <gpu-host> UAT runs the
    /// real-checkpoint variant.
    #[test]
    #[ignore]
    fn from_single_file_real_shape_load_smoke() {
        // Implementation deferred to the candle-fork-accessor follow-up.
        // 2.10 UAT replaces this with a real DreamShaper 8 / Realistic
        // Vision V5 checkpoint pull + load + 4-step generation.
    }

    #[test]
    fn from_single_file_rejects_missing_file() {
        let bogus = std::env::temp_dir().join(format!(
            "mold-sd15-from-sf-missing-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        let result = SD15Engine::from_single_file(
            "missing".to_string(),
            bogus,
            std::env::temp_dir().join("mold-sd15-tok-stub.json"),
            Scheduler::default(),
            LoadStrategy::Eager,
            0,
            None,
        );

        assert!(
            result.is_err(),
            "constructor must surface a missing-file error before deeper parsing",
        );
    }

    // The SD1.5 denoise loop and prompt encoder both gate the unconditional
    // pass on `cfg_active(guidance)`. These tests pin the predicate that
    // those branches use so a regression to `guidance > 1.0` (which would
    // break LCM / Lightning at exactly cfg=1.0) is caught here.

    #[test]
    fn test_cfg_disabled_at_guidance_1_0() {
        assert!(!cfg_active(1.0));
    }

    #[test]
    fn test_cfg_disabled_just_below_1_0() {
        assert!(!cfg_active(1.0 - 1e-5));
    }

    #[test]
    fn test_cfg_enabled_at_guidance_1_5() {
        assert!(cfg_active(1.5));
    }

    #[test]
    fn test_cfg_enabled_at_guidance_7_5() {
        assert!(cfg_active(7.5));
    }

    /// Regression test for the SD1.5 prompt-cache key bug: keying only on the
    /// positive prompt + guidance (as the original code did) returns stale
    /// `(uncond_old, cond)` when the user changes just the negative prompt.
    /// Mirrors the SD3 / SDXL regression tests in their respective pipelines.
    #[test]
    fn sd15_prompt_cache_distinguishes_negative_prompt_changes() {
        use crate::cache::{cfg_prompt_cache_key, store_cached_tensor};

        let cache: Mutex<LruCache<CfgPromptCacheKey, CachedTensor>> =
            Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY));
        let device = Device::Cpu;
        let dtype = DType::F32;
        let embeddings = candle_core::Tensor::zeros((1, 4), dtype, &device).unwrap();

        let key_a = cfg_prompt_cache_key("a cat", "blurry", 7.0);
        store_cached_tensor(&cache, key_a.clone(), &embeddings).unwrap();

        // Same positive + same guidance, different negative → MUST miss.
        let key_b = cfg_prompt_cache_key("a cat", "low quality", 7.0);
        let restored = restore_cached_tensor(&cache, &key_b, &device, dtype).unwrap();
        assert!(
            restored.is_none(),
            "different negative prompt must miss the cache (silent-wrong-output bug)",
        );

        // Same key as the insert → MUST hit.
        let restored = restore_cached_tensor(&cache, &key_a, &device, dtype).unwrap();
        assert!(
            restored.is_some(),
            "identical (pos, neg, guidance) must hit",
        );
    }

    /// `lora_stack_fingerprint` must produce equal fingerprints for the same
    /// `(path, scale)` pair and distinct ones when either changes — this is
    /// the predicate that drives UNet drop on LoRA stack change in eager
    /// mode. Same shape as the SDXL test of the same name.
    #[test]
    fn lora_stack_fingerprint_equality_drives_unet_drop() {
        let a = mold_core::LoraWeight {
            path: "/a.safetensors".to_string(),
            scale: 0.8,

            expert: None,
        };
        let b = mold_core::LoraWeight {
            path: "/b.safetensors".to_string(),
            scale: 0.4,

            expert: None,
        };
        let same_a = mold_core::LoraWeight {
            path: "/a.safetensors".to_string(),
            scale: 0.8,

            expert: None,
        };

        // Identical contents → identical fingerprint.
        assert_eq!(
            lora_stack_fingerprint(&[a.clone(), b.clone()]),
            lora_stack_fingerprint(&[same_a.clone(), b.clone()])
        );

        // Same path but different scale → different fingerprint.
        let scaled = mold_core::LoraWeight {
            path: "/a.safetensors".to_string(),
            scale: 0.81,

            expert: None,
        };
        assert_ne!(
            lora_stack_fingerprint(std::slice::from_ref(&a)),
            lora_stack_fingerprint(std::slice::from_ref(&scaled))
        );

        // Reordering yields a different fingerprint (stack order matters).
        assert_ne!(
            lora_stack_fingerprint(&[a.clone(), b.clone()]),
            lora_stack_fingerprint(&[b, a])
        );
    }
}
