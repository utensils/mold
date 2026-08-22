use anyhow::{bail, Result};
use mold_core::{Config, ModelPaths, Scheduler};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::engine::{InferenceEngine, LoadStrategy};
use crate::flux::FluxEngine;
use crate::flux2::Flux2Engine;
use crate::ltx2::Ltx2Engine;
use crate::ltx_video::LtxVideoEngine;
use crate::qwen_image::QwenImageEngine;
use crate::sd15::SD15Engine;
use crate::sd3::SD3Engine;
use crate::sdxl::SDXLEngine;
use crate::shared_pool::SharedPool;
use crate::wan::pipeline::WanEngine;
use crate::wuerstchen::WuerstchenEngine;
use crate::zimage::ZImageEngine;

/// Whether the frozen factory can construct a family today.
///
/// Contract-only families are explicit so static metadata work cannot be
/// mistaken for a runnable engine registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactoryFamilyAvailability {
    Runnable,
    ContractOnly,
}

pub fn factory_family_availability(family: &str) -> Option<FactoryFamilyAvailability> {
    if let Some(contract) = mold_core::minimax_h3::capability_contract_for_model(family) {
        Some(
            if contract.generation.runtime_available
                && crate::production_family_capability_for_family(family).is_some()
            {
                FactoryFamilyAvailability::Runnable
            } else {
                FactoryFamilyAvailability::ContractOnly
            },
        )
    } else if crate::production_family_capability_for_family(family).is_some() {
        Some(FactoryFamilyAvailability::Runnable)
    } else {
        None
    }
}

/// Immutable inputs that may change the concrete engine constructed for a
/// model. Scheduler-owned execution plans capture this once and pass it to
/// [`create_engine_with_frozen_config`], so worker dispatch never consults a
/// newer `Config` or process environment after admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenEngineConfig {
    pub family: String,
    /// Trusted root for concrete model artifacts. The activation policy strips
    /// this storage prefix before classifying path identity, so a Mold home
    /// named after a gated UAT target does not taint unrelated models.
    pub artifact_root: PathBuf,
    pub is_schnell: Option<bool>,
    pub is_turbo: Option<bool>,
    pub scheduler: Option<Scheduler>,
    pub t5_variant: Option<String>,
    pub qwen3_variant: Option<String>,
    pub qwen2_variant: Option<String>,
    pub qwen2_text_encoder_mode: Option<String>,
    pub ltx2_gemma_variant: Option<String>,
    pub umt5_variant: Option<String>,
    /// Exact encoder artifacts materialized before scheduler admission. Empty
    /// fields mean the legacy non-planned factory path; the planned factory
    /// verifies every populated path is already local before constructing an
    /// engine, so a leased worker cannot trigger a network download.
    pub selected_t5_path: Option<PathBuf>,
    pub selected_qwen3_paths: Vec<PathBuf>,
    pub selected_qwen2_path: Option<PathBuf>,
    pub selected_gemma_paths: Vec<PathBuf>,
    /// Exact Wan UMT5 encoder file admission materialized, when it chose a
    /// GGUF variant. `None` means the manifest's FP16 encoder.
    pub selected_umt5_path: Option<PathBuf>,
    /// Exact PuLID identity-conditioning assets admission materialized for a
    /// request that asks for face identity with a non-zero effective weight.
    ///
    /// `None` is the whole story for every other request: no identity fields,
    /// or `id_weight` 0, plans no assets, downloads nothing, and adds no
    /// memory demand. Populated, these four paths are verified local below
    /// exactly like the selected encoder artifacts.
    pub identity_assets: Option<mold_core::pulid_assets::PulidPaths>,
    /// Exact contract-only MiniMax H3 admission/factory authority. This is
    /// `None` for every runnable family and cannot activate H3 by itself.
    pub h3_factory_authority: Option<crate::FrozenH3FactoryAuthority>,
    pub runtime_environment: crate::runtime_env::FrozenRuntimeEnvironment,
    pub attention_backend: crate::attention::AttentionBackend,
    pub attention_chunk: crate::attention::AttentionChunkPolicy,
    pub vae_tiling: crate::vae_tiling::TiledMode,
    pub vae_dtype: crate::device::VaeDtypePolicy,
}

impl FrozenEngineConfig {
    pub fn resolve(model_name: &str, config: &Config) -> Self {
        let model_cfg = config.resolved_model_config(model_name);
        let runtime_environment = crate::runtime_env::snapshot();
        Self {
            family: resolve_family(model_name, config),
            artifact_root: config.resolved_models_dir(),
            is_schnell: model_cfg.is_schnell,
            is_turbo: model_cfg.is_turbo,
            scheduler: model_cfg.scheduler,
            t5_variant: runtime_environment
                .value("MOLD_T5_VARIANT")
                .map(str::to_string)
                .or_else(|| config.t5_variant.clone()),
            qwen3_variant: (!mold_core::validation::is_flux2_dev_model(model_name))
                .then(|| {
                    runtime_environment
                        .value("MOLD_QWEN3_VARIANT")
                        .map(str::to_string)
                        .or_else(|| config.qwen3_variant.clone())
                })
                .flatten(),
            qwen2_variant: runtime_environment
                .value("MOLD_QWEN2_VARIANT")
                .map(str::to_string),
            qwen2_text_encoder_mode: runtime_environment
                .value("MOLD_QWEN2_TEXT_ENCODER_MODE")
                .map(str::to_string),
            ltx2_gemma_variant: runtime_environment
                .value("MOLD_LTX2_GEMMA_VARIANT")
                .map(str::to_string),
            umt5_variant: runtime_environment
                .value("MOLD_UMT5_VARIANT")
                .map(str::to_string)
                .or_else(|| config.umt5_variant.clone()),
            selected_t5_path: None,
            selected_qwen3_paths: Vec::new(),
            selected_qwen2_path: None,
            selected_gemma_paths: Vec::new(),
            selected_umt5_path: None,
            identity_assets: None,
            h3_factory_authority: None,
            runtime_environment,
            attention_backend: crate::attention::AttentionBackend::resolve(),
            attention_chunk: crate::attention::resolved_chunk_policy(),
            vae_tiling: crate::vae_tiling::resolve_mode(),
            vae_dtype: crate::device::resolved_vae_dtype_policy(),
        }
    }
}

/// Determine the model family from config or manifest, defaulting to "flux".
fn resolve_family(model_name: &str, config: &Config) -> String {
    let model_cfg = config.resolved_model_config(model_name);
    if let Some(family) = model_cfg.family {
        return family;
    }
    // Default to flux for backward compatibility
    "flux".to_string()
}

/// Detect a Civitai single-file SD1.5 / SDXL checkpoint from a `ModelPaths`.
///
/// The Civitai single-file convention is one `.safetensors` containing
/// UNet + VAE + CLIP-L (+ CLIP-G for SDXL); diffusers layouts ship each
/// component as a separate file under its own subdir. The two are
/// distinguishable purely by `paths.transformer == paths.vae` plus the
/// `.safetensors` extension — a duck-type that avoids a new
/// `single_file: Option<PathBuf>` field threaded through every caller.
///
/// Used by the SD1.5 + SDXL match arms to dispatch to `from_single_file`
/// instead of the diffusers-layout `new` constructor.
fn is_single_file(paths: &ModelPaths) -> bool {
    paths.transformer == paths.vae
        && paths
            .transformer
            .extension()
            .is_some_and(|ext| ext == "safetensors")
}

/// Civitai / ComfyUI Flux.2 fine-tunes ship the transformer as one
/// BFL-native single-file `.safetensors` (every key prefixed
/// `model.diffusion_model.`) and rely on a separate `flux2-vae`
/// companion for the VAE — so the SD-style `paths.transformer ==
/// paths.vae` heuristic doesn't apply. Header-peek the transformer
/// instead. Returns `true` for `BflNative`, `BflNativeRoot`, and
/// `Nvfp4` — all three route through `SingleFileBackend` (NVFP4
/// exposes packed FP4, FP8 block scales, and tensor scales as streaming
/// subkeys; the BFL variants pass tensors through directly). Returns `false` for sharded
/// transformers and for any non-`safetensors` path so HF diffusers
/// layouts and GGUF quantised weights stay on the existing load paths.
fn is_flux2_bfl_native_single_file(paths: &ModelPaths) -> bool {
    if !paths.transformer_shards.is_empty() {
        return false;
    }
    let path = &paths.transformer;
    let is_safetensors = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
    if !is_safetensors {
        return false;
    }
    matches!(
        crate::flux2::detect_format(path),
        Ok(crate::flux2::Flux2SingleFileFormat::BflNative)
            | Ok(crate::flux2::Flux2SingleFileFormat::BflNativeRoot)
            | Ok(crate::flux2::Flux2SingleFileFormat::Nvfp4)
    )
}

/// LTX-2 catalog fine-tunes can pair one native transformer checkpoint with
/// a separate VAE, so equality between the transformer and VAE paths cannot
/// identify the single-file transformer layout. Use its header validator.
fn is_ltx2_native_single_file(paths: &ModelPaths) -> bool {
    paths.transformer_shards.is_empty()
        && crate::ltx2::single_file::load(&paths.transformer).is_ok()
}

fn require_prepared_cache_entry(
    path: &PathBuf,
    repo: &str,
    filename: &str,
    subdir: &str,
) -> Result<()> {
    let cached = mold_core::download::cached_file_path(repo, filename, Some(subdir));
    if cached.as_ref() != Some(path) {
        bail!(
            "prepared encoder artifact '{}' is not the current local cache entry \
             for {repo}/{filename}; refusing post-lease dependency resolution",
            path.display()
        );
    }
    Ok(())
}

fn validate_prepared_quantized_dependencies(frozen: &FrozenEngineConfig) -> Result<()> {
    if let (Some(path), Some(tag)) = (&frozen.selected_t5_path, frozen.t5_variant.as_deref()) {
        if tag != "fp16" {
            let variant = mold_core::manifest::find_t5_variant(tag)
                .ok_or_else(|| anyhow::anyhow!("unknown prepared T5 variant '{tag}'"))?;
            require_prepared_cache_entry(
                path,
                variant.hf_repo,
                variant.hf_filename,
                "shared/t5-gguf",
            )?;
        }
    }
    if let Some(path) = frozen.selected_qwen3_paths.first() {
        if frozen.qwen3_variant.as_deref() != Some("bf16") {
            let tag = frozen
                .qwen3_variant
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("prepared Qwen3 path has no frozen variant"))?;
            let candidates = [
                (
                    mold_core::manifest::find_qwen3_variant(tag),
                    "shared/qwen3-gguf",
                ),
                (
                    mold_core::manifest::find_qwen3_8b_variant(tag),
                    "shared/qwen3-8b-gguf",
                ),
            ];
            let matched = candidates.into_iter().any(|(variant, subdir)| {
                variant.is_some_and(|variant| {
                    mold_core::download::cached_file_path(
                        variant.hf_repo,
                        variant.hf_filename,
                        Some(subdir),
                    )
                    .as_ref()
                        == Some(path)
                })
            });
            if !matched {
                bail!(
                    "prepared Qwen3 artifact '{}' is not a current local cache entry",
                    path.display()
                );
            }
        }
    }
    if let Some(path) = &frozen.selected_qwen2_path {
        let tag = frozen
            .qwen2_variant
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("prepared Qwen2 path has no frozen variant"))?;
        let variant = mold_core::manifest::find_qwen2_vl_variant(tag)
            .ok_or_else(|| anyhow::anyhow!("unknown prepared Qwen2 variant '{tag}'"))?;
        require_prepared_cache_entry(
            path,
            variant.hf_repo,
            variant.hf_filename,
            "shared/qwen2-vl-gguf",
        )?;
    }
    if let Some(path) = &frozen.selected_umt5_path {
        let tag = frozen
            .umt5_variant
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("prepared UMT5 path has no frozen variant"))?;
        // `fp16` is the manifest's own encoder, not a shared-cache download,
        // so admission never freezes a path for it.
        let variant = mold_core::manifest::find_umt5_variant(tag)
            .ok_or_else(|| anyhow::anyhow!("unknown prepared UMT5 variant '{tag}'"))?;
        require_prepared_cache_entry(
            path,
            variant.hf_repo,
            variant.hf_filename,
            "shared/wan/umt5-gguf",
        )?;
    }
    Ok(())
}

fn validate_prepared_gemma_dependencies(
    paths: &ModelPaths,
    frozen: &FrozenEngineConfig,
) -> Result<()> {
    if frozen.selected_gemma_paths.is_empty() {
        return Ok(());
    }
    let root = paths
        .text_encoder_files
        .first()
        .and_then(|path| path.parent())
        .ok_or_else(|| anyhow::anyhow!("prepared Gemma assets have no concrete root"))?;
    let tag = frozen
        .ltx2_gemma_variant
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("prepared Gemma paths have no frozen variant"))?;
    let mut discovered = std::fs::read_dir(root)
        .map_err(|error| anyhow::anyhow!("failed to inspect prepared Gemma root: {error}"))?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            let selected = match tag {
                "bf16" => {
                    name == "model.safetensors"
                        || (name.starts_with("model-") && name.ends_with(".safetensors"))
                }
                "q4" => name.ends_with(".gguf"),
                _ => false,
            };
            (selected && path.is_file()).then_some(path)
        })
        .collect::<Vec<_>>();
    discovered.sort();
    if discovered != frozen.selected_gemma_paths {
        bail!(
            "prepared Gemma {tag} artifact set changed after admission; refusing live variant discovery"
        );
    }
    Ok(())
}

/// Create an inference engine for the given model, auto-detecting the family.
///
/// Returns the appropriate engine (FluxEngine, SD15Engine, SDXLEngine, or ZImageEngine)
/// based on the model's family from config or manifest.
///
/// `load_strategy` controls whether components are loaded eagerly (server) or
/// sequentially (CLI one-shot) to reduce peak memory.
pub fn create_engine(
    model_name: String,
    paths: ModelPaths,
    config: &Config,
    load_strategy: LoadStrategy,
    gpu_ordinal: usize,
    offload: bool,
) -> Result<Box<dyn InferenceEngine>> {
    create_engine_with_pool(
        model_name,
        paths,
        config,
        load_strategy,
        gpu_ordinal,
        offload,
        None,
    )
}

/// Create an inference engine with an optional shared tokenizer pool.
///
/// When `shared_pool` is provided, engines can cache and reuse tokenizers across
/// model switches (e.g. all FLUX variants share the same T5/CLIP tokenizers).
pub fn create_engine_with_pool(
    model_name: String,
    paths: ModelPaths,
    config: &Config,
    load_strategy: LoadStrategy,
    gpu_ordinal: usize,
    offload: bool,
    shared_pool: Option<Arc<Mutex<SharedPool>>>,
) -> Result<Box<dyn InferenceEngine>> {
    let mut frozen = FrozenEngineConfig::resolve(&model_name, config);
    // The forced-local path has no scheduler admission to materialize the
    // PuLID bundle, so resolve it here. `FrozenEngineConfig::resolve` itself
    // deliberately stays `None`: the server compares a live `resolve()` against
    // a frozen plan (`execution_plan::current_engine_config`), and populating
    // it there would make every admitted job's frozen config disagree with the
    // live one the moment the bundle is installed.
    //
    // `pulid_paths_for` returns `Some` only for a complete, on-disk bundle, so
    // the "every populated frozen path must already be local" check in
    // `create_engine_with_frozen_config` holds by construction. It is inert for
    // a request that does not condition on a face — the engine loads nothing
    // until one does.
    //
    // The bundle follows the MODEL, because the adapter does: a FLUX engine
    // resolving the SDXL bundle would freeze a checkpoint whose `pulid_ca.*`
    // prefix does not exist. `identity_family` is the same authority admission
    // uses, so the forced-local and served paths choose identically.
    if frozen.identity_assets.is_none() {
        frozen.identity_assets = mold_core::identity::identity_family(&model_name)
            .and_then(|family| mold_core::pulid_assets::pulid_paths_for(config, family));
    }
    create_engine_with_frozen_config(
        model_name,
        paths,
        &frozen,
        load_strategy,
        gpu_ordinal,
        offload,
        shared_pool,
    )
}

fn boxed_inference_engine(engine: impl InferenceEngine + 'static) -> Box<dyn InferenceEngine> {
    Box::new(engine)
}

/// Create an engine from scheduler-frozen construction inputs.
///
/// Unlike [`create_engine_with_pool`], this path performs no live config or
/// environment resolution. It is the only factory entry point valid after a
/// scheduler lease has been granted.
pub fn create_engine_with_frozen_config(
    model_name: String,
    paths: ModelPaths,
    frozen: &FrozenEngineConfig,
    load_strategy: LoadStrategy,
    gpu_ordinal: usize,
    offload: bool,
    shared_pool: Option<Arc<Mutex<SharedPool>>>,
) -> Result<Box<dyn InferenceEngine>> {
    create_engine_with_frozen_config_and_h3_factory(
        model_name,
        paths,
        frozen,
        load_strategy,
        gpu_ordinal,
        offload,
        shared_pool,
        |_| bail!(crate::minimax_h3::engine::H3_REAL_LOADER_DISABLED),
    )
}

#[allow(clippy::too_many_arguments)]
fn create_engine_with_frozen_config_and_h3_factory<F>(
    model_name: String,
    paths: ModelPaths,
    frozen: &FrozenEngineConfig,
    load_strategy: LoadStrategy,
    gpu_ordinal: usize,
    offload: bool,
    shared_pool: Option<Arc<Mutex<SharedPool>>>,
    h3_factory: F,
) -> Result<Box<dyn InferenceEngine>>
where
    F: FnOnce(
        crate::minimax_h3::engine::H3RuntimeLoadRequest<'_>,
    ) -> Result<Box<dyn InferenceEngine>>,
{
    // The compliance gate is intentionally the first operation. No path,
    // runtime, device, authority, or future loader inspection may precede it.
    mold_core::require_model_activation(&model_name, Some(&frozen.family))?;
    if mold_core::minimax_h3::is_family(&frozen.family) {
        let authority = frozen.h3_factory_authority.as_ref().ok_or_else(|| {
            anyhow::anyhow!("MiniMax H3 factory dispatch requires an exact frozen server authority")
        })?;
        // The static runtime/registry gate deliberately precedes even path
        // classification and local-existence checks. Legal authorization alone
        // can therefore never make a contract-only build touch H3 artifacts.
        authority.validate_for_dispatch(
            &model_name,
            &frozen.family,
            gpu_ordinal,
            offload,
            frozen.attention_backend,
            frozen.attention_chunk,
        )?;
    }
    for path in paths.all_file_paths() {
        mold_core::require_model_artifact_activation(
            path,
            Some(&frozen.artifact_root),
            Some(&frozen.family),
        )?;
    }
    let current_attention = crate::attention::AttentionBackend::resolve();
    let current_chunk = crate::attention::resolved_chunk_policy();
    let current_vae_tiling = crate::vae_tiling::resolve_mode();
    let current_vae_dtype = crate::device::resolved_vae_dtype_policy();
    let current_runtime_environment = crate::runtime_env::snapshot();
    if frozen.attention_backend != current_attention
        || frozen.attention_chunk != current_chunk
        || frozen.vae_tiling != current_vae_tiling
        || frozen.vae_dtype != current_vae_dtype
        || frozen.runtime_environment != current_runtime_environment
    {
        bail!(
            "frozen engine runtime policy does not match the process-frozen \
             attention/chunk/VAE authority"
        );
    }
    // Identity assets join the encoder artifacts here rather than in a second
    // loop: the rule is one rule — a populated frozen path must already be
    // local, so a leased worker can never trigger a network download.
    let identity_paths = frozen.identity_assets.iter().flat_map(|assets| {
        [
            &assets.adapter,
            &assets.vision_encoder_source,
            &assets.face_detector,
            &assets.face_recognizer,
            &assets.face_parser_source,
        ]
    });
    for path in frozen
        .selected_t5_path
        .iter()
        .chain(frozen.selected_qwen3_paths.iter())
        .chain(frozen.selected_qwen2_path.iter())
        .chain(frozen.selected_gemma_paths.iter())
        .chain(frozen.selected_umt5_path.iter())
        .chain(identity_paths)
    {
        if !path.is_file() {
            bail!(
                "prepared engine artifact '{}' is not locally available; \
                 refusing post-lease dependency resolution",
                path.display()
            );
        }
    }
    validate_prepared_quantized_dependencies(frozen)?;
    validate_prepared_gemma_dependencies(&paths, frozen)?;
    let engine_result: Result<Box<dyn InferenceEngine>> = match frozen.family.as_str() {
        "flux" => {
            Ok(boxed_inference_engine(FluxEngine::new(
                model_name,
                paths,
                frozen.is_schnell,
                frozen.t5_variant.clone(),
                load_strategy,
                gpu_ordinal,
                offload,
                shared_pool,
                frozen.identity_assets.clone(),
            )))
        }
        "sd15" | "sd1.5" | "stable-diffusion-1.5" => {
            let scheduler = frozen.scheduler.unwrap_or(Scheduler::Ddim);
            if is_single_file(&paths) {
                // Companion-pulled CLIP-L tokenizer. Civitai
                // single-file checkpoints never bundle the tokenizer.
                let clip_tokenizer = paths
                    .clip_tokenizer
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!(
                        "single-file SD1.5 dispatch requires a companion-pulled clip_tokenizer"
                    ))?;
                let single_file = paths.transformer.clone();
                Ok(boxed_inference_engine(SD15Engine::from_single_file(
                    model_name,
                    single_file,
                    clip_tokenizer,
                    scheduler,
                    load_strategy,
                    gpu_ordinal,
                    shared_pool,
                )?))
            } else {
                Ok(boxed_inference_engine(SD15Engine::new(
                    model_name,
                    paths,
                    scheduler,
                    load_strategy,
                    gpu_ordinal,
                    shared_pool,
                )))
            }
        }
        "sdxl" => {
            let is_turbo = frozen
                .is_turbo
                .unwrap_or_else(|| model_name.contains("turbo"));
            let scheduler = frozen.scheduler.unwrap_or(if is_turbo {
                Scheduler::EulerAncestral
            } else {
                Scheduler::Ddim
            });
            if is_single_file(&paths) {
                // Companion-pulled CLIP-L + CLIP-G tokenizers.
                let clip_l_tokenizer = paths.clip_tokenizer.clone().ok_or_else(|| {
                    anyhow::anyhow!(
                        "single-file SDXL dispatch requires a companion-pulled clip_tokenizer(CLIP-L)"
                    )
                })?;
                let clip_g_tokenizer = paths.clip_tokenizer_2.clone().ok_or_else(|| {
                    anyhow::anyhow!(
                        "single-file SDXL dispatch requires a companion-pulled clip_tokenizer_2(CLIP-G)"
                    )
                })?;
                let single_file = paths.transformer.clone();
                Ok(boxed_inference_engine(SDXLEngine::from_single_file(
                    model_name,
                    single_file,
                    clip_l_tokenizer,
                    clip_g_tokenizer,
                    scheduler,
                    is_turbo,
                    load_strategy,
                    gpu_ordinal,
                    shared_pool,
                    frozen.identity_assets.clone(),
                )?))
            } else {
                Ok(boxed_inference_engine(SDXLEngine::new(
                    model_name,
                    paths,
                    scheduler,
                    is_turbo,
                    load_strategy,
                    gpu_ordinal,
                    shared_pool,
                    frozen.identity_assets.clone(),
                )))
            }
        }
        "sd3" | "sd3.5" | "stable-diffusion-3" | "stable-diffusion-3.5" => {
            let is_turbo = frozen
                .is_turbo
                .unwrap_or_else(|| model_name.contains("turbo"));
            let is_medium = model_name.contains("medium");
            Ok(boxed_inference_engine(SD3Engine::new(
                model_name,
                paths,
                is_turbo,
                is_medium,
                frozen.t5_variant.clone(),
                load_strategy,
                gpu_ordinal,
                offload,
                shared_pool,
            )))
        }
        "z-image" => {
            Ok(boxed_inference_engine(ZImageEngine::new(
                model_name,
                paths,
                frozen.qwen3_variant.clone(),
                load_strategy,
                gpu_ordinal,
                offload,
                shared_pool,
            )))
        }
        "flux2" | "flux.2" | "flux2-klein" => {
            if is_flux2_bfl_native_single_file(&paths) {
                // Civitai / ComfyUI fine-tunes ship as one BFL-native
                // safetensors. The VAE + Qwen3 text encoder come from the
                // `flux2-vae` and `flux2-te*` companions wired by the
                // catalog bridge — copy them through unchanged.
                Ok(boxed_inference_engine(Flux2Engine::from_single_file(
                    model_name,
                    paths.transformer.clone(),
                    paths.vae.clone(),
                    paths.text_encoder_files.clone(),
                    paths.text_tokenizer.clone(),
                    frozen.qwen3_variant.clone(),
                    load_strategy,
                    gpu_ordinal,
                    offload,
                    shared_pool,
                )?))
            } else {
                Ok(boxed_inference_engine(Flux2Engine::new(
                    model_name,
                    paths,
                    frozen.qwen3_variant.clone(),
                    load_strategy,
                    gpu_ordinal,
                    offload,
                    shared_pool,
                )))
            }
        }
        "qwen-image" | "qwen_image" => Ok(boxed_inference_engine(QwenImageEngine::new_with_preferences(
            model_name,
            paths,
            load_strategy,
            gpu_ordinal,
            offload,
            shared_pool,
            frozen.qwen2_variant.clone(),
            frozen.qwen2_text_encoder_mode.clone(),
        ))),
        "qwen-image-edit" => Ok(boxed_inference_engine(QwenImageEngine::new_with_preferences(
            model_name,
            paths,
            load_strategy,
            gpu_ordinal,
            offload,
            shared_pool,
            frozen.qwen2_variant.clone(),
            frozen.qwen2_text_encoder_mode.clone(),
        ))),
        "ltx-video" | "ltx_video" => {
            if is_single_file(&paths) {
                // Civitai single-file dispatch. `paths.vae` is
                // either the same checkpoint (combined transformer+VAE) or
                // the ltx-video-vae companion set by `populate_companion_paths`.
                let vae_path = if paths.vae != paths.transformer {
                    Some(paths.vae.clone())
                } else {
                    None
                };
                Ok(boxed_inference_engine(LtxVideoEngine::from_single_file(
                    model_name,
                    paths.transformer.clone(),
                    vae_path,
                    paths.t5_encoder.clone(),
                    paths.t5_tokenizer.clone(),
                    frozen.t5_variant.clone(),
                    load_strategy,
                    gpu_ordinal,
                    shared_pool,
                )?))
            } else {
                Ok(boxed_inference_engine(LtxVideoEngine::new(
                    model_name,
                    paths,
                    frozen.t5_variant.clone(),
                    load_strategy,
                    gpu_ordinal,
                    shared_pool,
                )))
            }
        }
        "ltx2" | "ltx-2" | "ltx2.3" => {
            if is_ltx2_native_single_file(&paths) {
                // Civitai single-file dispatch. Combined checkpoints use
                // `vae.*`; transformer-only checkpoints use `paths.vae`.
                // Pass the full companion graph so both the Gemma encoder
                // and version-matched VAE reach the runtime and chain stages.
                let checkpoint = paths.transformer.clone();
                Ok(boxed_inference_engine(Ltx2Engine::from_single_file_with_gemma_variant(
                    model_name,
                    checkpoint,
                    paths,
                    load_strategy,
                    gpu_ordinal,
                    frozen.ltx2_gemma_variant.clone(),
                )?))
            } else {
                Ok(boxed_inference_engine(Ltx2Engine::new_with_gemma_variant(
                    model_name,
                    paths,
                    load_strategy,
                    gpu_ordinal,
                    frozen.ltx2_gemma_variant.clone(),
                )))
            }
        }
        // Wan resolves its geometry from the checkpoint's own tensor shapes at
        // load time, so one arm serves 2.1 and 2.2 and any future width.
        "wan" => Ok(boxed_inference_engine(
            WanEngine::new(model_name, paths, load_strategy, gpu_ordinal, shared_pool)
                .with_umt5_variant(frozen.umt5_variant.clone())
                .with_selected_umt5_path(frozen.selected_umt5_path.clone()),
        )),
        family if mold_core::minimax_h3::is_family(family) => {
            let authority = frozen.h3_factory_authority.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "MiniMax H3 factory dispatch requires an exact frozen server authority"
                )
            })?;
            authority.validate_for_dispatch(
                &model_name,
                family,
                gpu_ordinal,
                offload,
                frozen.attention_backend,
                frozen.attention_chunk,
            )?;
            h3_factory(crate::minimax_h3::engine::H3RuntimeLoadRequest {
                model_name: &model_name,
                paths: &paths,
                authority,
                load_strategy,
                gpu_ordinal,
                block_offload: offload,
            })
        }
        "wuerstchen" | "wuerstchen-v2" => Ok(boxed_inference_engine(WuerstchenEngine::new(
            model_name,
            paths,
            load_strategy,
            gpu_ordinal,
            shared_pool,
        ))),
        other => bail!(
            "unknown model family '{}' for model '{}'. Supported: flux, flux2, ltx-video, ltx2, sd15, sd3, sdxl, z-image, qwen-image, qwen-image-edit, wan, wuerstchen",
            other,
            model_name
        ),
    };
    let engine = engine_result?;
    crate::validate_runtime_batch_capability(&frozen.family, engine.batch_execution_capability())?;
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn h3_factory_authority(
        frozen: &FrozenEngineConfig,
        model: &str,
    ) -> crate::FrozenH3FactoryAuthority {
        crate::FrozenH3FactoryAuthority::new_contract_only(crate::H3FactoryAuthorityInput {
            model: model.into(),
            device_id: "gpu-0".into(),
            device_ordinal: 0,
            compute_capability: Some((8, 9)),
            execution_fingerprint: sha('a'),
            conditioner_placement: crate::H3FactoryConditionerPlacement::HostCpuThenDrop,
            qwen_parameter_bytes: 2048,
            qwen_host_resident_parameter_bytes: 2048,
            qwen_device_resident_parameter_bytes: 0,
            qwen_activation_workspace_bytes: 1024,
            qwen_maximum_tensor_staging_bytes: 512,
            qwen_retained_raw_header_bytes: 64,
            qwen_output_text_rows: 1,
            qwen_vision_rows: 0,
            condition_visual_rows: 0,
            resident_block_count: 8,
            prefetch_depth: 1,
            attention_backend: frozen.attention_backend,
            attention_chunk: frozen.attention_chunk,
            attention_kernel_identity: "synthetic-lossless-packed-attention-v1".into(),
            attention_qualification_sha256: sha('b'),
            attention_full_noncausal: true,
            attention_lossless: true,
            attention_head_count: 56,
            attention_head_dim: 128,
            block_offload: true,
            attention_runtime: None,
            prepared_attempt: None,
            execution_budget_echo: None,
            quantization: crate::H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256: sha('c'),
                qwen_policy_sha256: sha('d'),
                pruned_adaln_table_sha256: sha('e'),
                turbo_adapter: None,
            },
            components: [
                crate::H3FactoryComponentRole::Conditioner,
                crate::H3FactoryComponentRole::Transformer,
                crate::H3FactoryComponentRole::VisualVae,
                crate::H3FactoryComponentRole::AudioVae,
            ]
            .into_iter()
            .enumerate()
            .map(|(index, role)| {
                crate::H3FactoryComponentAuthority::new(
                    role,
                    sha(char::from(b'1' + index as u8)),
                    sha(char::from(b'5' + index as u8)),
                )
                .unwrap()
            })
            .collect(),
        })
        .unwrap()
    }

    fn dummy_paths() -> ModelPaths {
        ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: PathBuf::from("/tmp/transformer"),
            transformer_shards: vec![],
            vae: PathBuf::from("/tmp/vae"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: Some(PathBuf::from("/tmp/t5")),
            clip_encoder: Some(PathBuf::from("/tmp/clip")),
            t5_tokenizer: Some(PathBuf::from("/tmp/t5_tok")),
            clip_tokenizer: Some(PathBuf::from("/tmp/clip_tok")),
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![],
            text_tokenizer: None,
            decoder: None,
        }
    }

    #[test]
    fn every_advertised_production_family_and_alias_constructs_through_frozen_factory() {
        for entry in crate::production_batch_capabilities() {
            for family in std::iter::once(entry.family).chain(entry.aliases.iter().copied()) {
                let mut frozen = FrozenEngineConfig::resolve(family, &Config::default());
                frozen.family = family.to_string();
                let engine = create_engine_with_frozen_config(
                    family.to_string(),
                    dummy_paths(),
                    &frozen,
                    LoadStrategy::Sequential,
                    0,
                    false,
                    None,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "advertised production family/alias {family:?} did not construct: {error}"
                    )
                });
                assert_eq!(
                    engine.batch_execution_capability(),
                    entry.execution,
                    "{family}"
                );
            }
        }
    }

    #[test]
    fn frozen_factory_rejects_runtime_policy_claim_it_cannot_consume() {
        let config = Config::default();
        let mut frozen = FrozenEngineConfig::resolve("flux-dev:q4", &config);
        frozen.attention_backend = match frozen.attention_backend {
            crate::attention::AttentionBackend::Math => crate::attention::AttentionBackend::Flash,
            crate::attention::AttentionBackend::Flash => crate::attention::AttentionBackend::Math,
        };
        let error = create_engine_with_frozen_config(
            "flux-dev:q4".into(),
            dummy_paths(),
            &frozen,
            LoadStrategy::Sequential,
            0,
            false,
            None,
        )
        .err()
        .expect("mismatched runtime policy must fail closed");
        assert!(error
            .to_string()
            .contains("process-frozen attention/chunk/VAE authority"));
    }

    #[test]
    fn frozen_factory_rejects_compliance_gated_family_before_runtime_or_device_work() {
        let mut frozen = FrozenEngineConfig::resolve("private-checkpoint", &Config::default());
        frozen.family = "minimax-h3".to_string();

        let error = create_engine_with_frozen_config(
            "private-checkpoint".into(),
            dummy_paths(),
            &frozen,
            LoadStrategy::Sequential,
            usize::MAX,
            false,
            None,
        )
        .err()
        .expect("compliance-gated family must fail closed");

        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[test]
    fn frozen_factory_never_calls_h3_loader_before_the_runtime_gate() {
        #[cfg(feature = "h3")]
        let models = vec![mold_core::minimax_h3::REF2VA_COMFY];
        #[cfg(not(feature = "h3"))]
        let models = vec![
            mold_core::minimax_h3::FL2VA_COMFY,
            mold_core::minimax_h3::REF2VA_COMFY,
        ];
        for model in models {
            let mut frozen = FrozenEngineConfig::resolve(model, &Config::default());
            frozen.family = mold_core::minimax_h3::FAMILY.to_string();
            frozen.h3_factory_authority = Some(h3_factory_authority(&frozen, model));
            let called = Arc::new(AtomicBool::new(false));
            let loader_called = Arc::clone(&called);

            let error = create_engine_with_frozen_config_and_h3_factory(
                model.into(),
                dummy_paths(),
                &frozen,
                LoadStrategy::Sequential,
                0,
                true,
                None,
                move |_| {
                    loader_called.store(true, Ordering::SeqCst);
                    unreachable!("the MiniMax H3 legal gate must precede the loader")
                },
            )
            .err()
            .expect("contract-only H3 runtime must fail closed");

            assert!(error.to_string().contains(
                "public capability or production factory registry remains runtime unavailable"
            ));
            assert!(!called.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn h3_aliases_are_factory_contract_only_not_production_capabilities() {
        for alias in mold_core::minimax_h3::FAMILY_ALIASES {
            assert_eq!(
                factory_family_availability(alias),
                Some(if cfg!(feature = "h3") {
                    FactoryFamilyAvailability::Runnable
                } else {
                    FactoryFamilyAvailability::ContractOnly
                })
            );
            assert_eq!(
                crate::production_family_capability_for_family(alias).is_some(),
                cfg!(feature = "h3")
            );
        }
        assert_eq!(
            factory_family_availability("flux"),
            Some(FactoryFamilyAvailability::Runnable)
        );
    }

    #[test]
    fn frozen_factory_rejects_restricted_artifact_path_before_runtime_or_device_work() {
        let frozen = FrozenEngineConfig::resolve("flux-dev:q4", &Config::default());
        let mut paths = dummy_paths();
        paths.transformer = PathBuf::from("/models/MiniMax-H3/transformer.safetensors");

        let error = create_engine_with_frozen_config(
            "ordinary-checkpoint".into(),
            paths,
            &frozen,
            LoadStrategy::Sequential,
            usize::MAX,
            false,
            None,
        )
        .err()
        .expect("restricted artifact path must fail closed");

        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[test]
    fn frozen_factory_ignores_h3_named_root_but_rejects_nested_h3_artifact() {
        let artifact_root = PathBuf::from("/Volumes/ExternalStorage/mold-uat/minimax-h3/models");
        let mut frozen = FrozenEngineConfig::resolve("flux-dev:q4", &Config::default());
        frozen.artifact_root = artifact_root.clone();
        frozen.attention_backend = match crate::attention::AttentionBackend::resolve() {
            crate::attention::AttentionBackend::Math => crate::attention::AttentionBackend::Flash,
            crate::attention::AttentionBackend::Flash => crate::attention::AttentionBackend::Math,
        };
        let mut ordinary = dummy_paths();
        ordinary.transformer = artifact_root.join("flux-dev/transformer.safetensors");
        ordinary.vae = artifact_root.join("flux-vae/ae.safetensors");

        let error = create_engine_with_frozen_config(
            "ordinary-checkpoint".into(),
            ordinary,
            &frozen,
            LoadStrategy::Sequential,
            usize::MAX,
            false,
            None,
        )
        .err()
        .expect("the deliberate runtime mismatch should follow activation");
        assert!(
            error
                .to_string()
                .contains("process-frozen attention/chunk/VAE authority"),
            "the storage root name must not taint ordinary artifacts: {error:#}"
        );

        let mut nested_h3 = dummy_paths();
        nested_h3.transformer = artifact_root.join("MiniMax-H3/transformer.safetensors");
        let error = create_engine_with_frozen_config(
            "ordinary-checkpoint".into(),
            nested_h3,
            &frozen,
            LoadStrategy::Sequential,
            usize::MAX,
            false,
            None,
        )
        .err()
        .expect("a nested H3 artifact must fail closed before runtime checks");
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[test]
    fn prepared_gemma_rejects_new_live_variant_candidate() {
        let root = tempfile::tempdir().unwrap();
        let tokenizer = root.path().join("tokenizer.json");
        let shard = root.path().join("model-00001-of-00001.safetensors");
        std::fs::write(&tokenizer, b"tokenizer").unwrap();
        std::fs::write(&shard, b"weights").unwrap();
        let mut paths = dummy_paths();
        paths.text_encoder_files = vec![tokenizer];
        let mut frozen = FrozenEngineConfig::resolve("ltx-2-19b-distilled:fp8", &Config::default());
        frozen.ltx2_gemma_variant = Some("bf16".into());
        frozen.selected_gemma_paths = vec![shard];
        validate_prepared_gemma_dependencies(&paths, &frozen).unwrap();

        std::fs::write(
            root.path().join("model-00002-of-00002.safetensors"),
            b"changed",
        )
        .unwrap();
        let error = validate_prepared_gemma_dependencies(&paths, &frozen).unwrap_err();
        assert!(error.to_string().contains("changed after admission"));
    }

    #[test]
    fn prepared_qwen3_bf16_missing_path_fails_before_live_variant_fallback() {
        let root = tempfile::tempdir().unwrap();
        let missing = root.path().join("planned-qwen3-bf16.safetensors");
        let mut paths = dummy_paths();
        paths.text_encoder_files = vec![missing.clone()];
        let mut frozen = FrozenEngineConfig::resolve("z-image:bf16", &Config::default());
        frozen.family = "z-image".into();
        frozen.qwen3_variant = Some("bf16".into());
        frozen.selected_qwen3_paths = vec![missing];

        let error = create_engine_with_frozen_config(
            "z-image:bf16".into(),
            paths,
            &frozen,
            LoadStrategy::Sequential,
            0,
            false,
            None,
        )
        .err()
        .expect("a missing planned BF16 path must fail before runtime auto-resolution");
        assert!(error
            .to_string()
            .contains("refusing post-lease dependency resolution"));
    }

    /// Every frozen PuLID asset — the parser included — must already be local
    /// at post-lease construction; a bundle that lost its fifth file between
    /// admission and dispatch fails closed exactly like the other four.
    #[test]
    fn prepared_identity_bundle_missing_parser_fails_before_engine_construction() {
        let root = tempfile::tempdir().unwrap();
        let present = |name: &str| {
            let path = root.path().join(name);
            std::fs::write(&path, b"stub").unwrap();
            path
        };
        let missing_parser = root.path().join("parsing_bisenet.pth");
        let mut frozen = FrozenEngineConfig::resolve("z-image:bf16", &Config::default());
        frozen.family = "z-image".into();
        frozen.identity_assets = Some(mold_core::pulid_assets::PulidPaths {
            family: mold_core::identity::IdentityFamily::Flux,
            adapter: present("pulid_flux_v0.9.1.safetensors"),
            vision_encoder_source: present("EVA02_CLIP_L_336_psz14_s6B.pt"),
            face_detector: present("scrfd_10g_bnkps.onnx"),
            face_recognizer: present("glintr100.onnx"),
            face_parser_source: missing_parser.clone(),
        });

        let error = create_engine_with_frozen_config(
            "z-image:bf16".into(),
            dummy_paths(),
            &frozen,
            LoadStrategy::Sequential,
            0,
            false,
            None,
        )
        .err()
        .expect("a missing frozen face parser must fail before engine construction");
        let rendered = error.to_string();
        assert!(
            rendered.contains("refusing post-lease dependency resolution"),
            "{rendered}"
        );
        assert!(
            rendered.contains(&missing_parser.display().to_string()),
            "{rendered}"
        );
    }

    #[test]
    fn flux2_dev_ignores_global_qwen3_variant_authority() {
        let config = Config {
            qwen3_variant: Some("q4".into()),
            ..Config::default()
        };
        let frozen = FrozenEngineConfig::resolve("flux2-dev:bf16", &config);
        assert_eq!(frozen.qwen3_variant, None);
        assert!(frozen.selected_qwen3_paths.is_empty());
    }

    #[test]
    fn resolve_family_from_manifest_sd15() {
        let config = Config::default();
        assert_eq!(resolve_family("sd15:fp16", &config), "sd15");
        assert_eq!(resolve_family("dreamshaper-v8:fp16", &config), "sd15");
        assert_eq!(resolve_family("realistic-vision-v5:fp16", &config), "sd15");
    }

    #[test]
    fn resolve_family_from_manifest_sdxl() {
        let config = Config::default();
        assert_eq!(resolve_family("sdxl-base:fp16", &config), "sdxl");
    }

    #[test]
    fn resolve_family_unknown_defaults_to_flux() {
        let config = Config::default();
        assert_eq!(resolve_family("totally-unknown-model", &config), "flux");
    }

    #[test]
    fn resolve_family_config_overrides_manifest() {
        let mut config = Config::default();
        config.models.insert(
            "custom-model".to_string(),
            mold_core::config::ModelConfig {
                family: Some("sd15".to_string()),
                ..Default::default()
            },
        );
        assert_eq!(resolve_family("custom-model", &config), "sd15");
    }

    #[test]
    fn every_factory_family_runtime_contract_matches_the_static_registry() {
        for family in [
            "flux",
            "flux2",
            "sd15",
            "sdxl",
            "sd3",
            "z-image",
            "qwen-image",
            "qwen-image-edit",
            "ltx-video",
            "ltx2",
            "wan",
            "wuerstchen",
        ] {
            let mut config = Config::default();
            config.models.insert(
                format!("test-{family}"),
                mold_core::config::ModelConfig {
                    family: Some(family.to_string()),
                    ..Default::default()
                },
            );
            let engine = create_engine(
                format!("test-{family}"),
                dummy_paths(),
                &config,
                LoadStrategy::Sequential,
                0,
                false,
            )
            .unwrap();
            let expected = crate::batch_execution_capability_for_family(family).unwrap();
            assert_eq!(engine.batch_execution_capability(), expected, "{family}");
        }
    }

    #[test]
    fn create_engine_sd15() {
        let config = Config::default();
        let engine = create_engine(
            "sd15:fp16".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            0,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "sd15:fp16");
    }

    #[test]
    fn create_engine_sd15_family_alias() {
        // Config-based family alias "sd1.5" should work
        let mut config = Config::default();
        config.models.insert(
            "my-model".to_string(),
            mold_core::config::ModelConfig {
                family: Some("sd1.5".to_string()),
                ..Default::default()
            },
        );
        let engine = create_engine(
            "my-model".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            0,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-model");
    }

    #[test]
    fn resolve_family_from_manifest_sd3() {
        let config = Config::default();
        assert_eq!(resolve_family("sd3.5-large:q8", &config), "sd3");
        assert_eq!(resolve_family("sd3.5-medium:q8", &config), "sd3");
        assert_eq!(resolve_family("sd3.5-large-turbo:q8", &config), "sd3");
    }

    #[test]
    fn resolve_family_from_manifest_flux2() {
        let config = Config::default();
        assert_eq!(resolve_family("flux2-klein:bf16", &config), "flux2");
    }

    #[test]
    fn create_engine_flux2() {
        let config = Config::default();
        let engine = create_engine(
            "flux2-klein:bf16".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            0,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "flux2-klein:bf16");
    }

    #[test]
    fn unknown_family_error_lists_ltx_families() {
        let mut config = Config::default();
        config.models.insert(
            "my-model".to_string(),
            mold_core::config::ModelConfig {
                family: Some("mystery".to_string()),
                ..Default::default()
            },
        );
        let err = create_engine(
            "my-model".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            0,
            false,
        )
        .err()
        .expect("unknown family should fail");
        let message = err.to_string();
        assert!(message.contains("ltx-video"));
        assert!(message.contains("ltx2"));
        assert!(message.contains("wan"));
    }

    #[test]
    fn create_engine_flux2_family_alias() {
        // Config-based family alias "flux.2" should work
        let mut config = Config::default();
        config.models.insert(
            "my-flux2".to_string(),
            mold_core::config::ModelConfig {
                family: Some("flux.2".to_string()),
                ..Default::default()
            },
        );
        let engine = create_engine(
            "my-flux2".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            0,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-flux2");
    }

    #[test]
    fn resolve_family_from_manifest_qwen_image() {
        let config = Config::default();
        assert_eq!(resolve_family("qwen-image:bf16", &config), "qwen-image");
    }

    #[test]
    fn resolve_family_from_manifest_qwen_image_edit() {
        let config = Config::default();
        assert_eq!(
            resolve_family("qwen-image-edit-2511:q4", &config),
            "qwen-image-edit"
        );
    }

    #[test]
    fn create_engine_qwen_image() {
        let mut config = Config::default();
        config.models.insert(
            "my-qwen-image".to_string(),
            mold_core::config::ModelConfig {
                family: Some("qwen-image".to_string()),
                ..Default::default()
            },
        );
        let engine = create_engine(
            "my-qwen-image".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            0,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-qwen-image");
    }

    #[test]
    fn create_engine_unknown_family_fails() {
        let mut config = Config::default();
        config.models.insert(
            "bad-model".to_string(),
            mold_core::config::ModelConfig {
                family: Some("nosuchfamily".to_string()),
                ..Default::default()
            },
        );
        let result = create_engine(
            "bad-model".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            0,
            false,
        );
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(err.contains("nosuchfamily"));
        assert!(err.contains("qwen-image-edit"));
    }

    #[test]
    fn create_engine_qwen_image_edit_routes_to_qwen_engine() {
        let result = create_engine(
            "qwen-image-edit-2511:q4".to_string(),
            dummy_paths(),
            &Config::default(),
            LoadStrategy::Sequential,
            0,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn resolve_family_from_manifest_wuerstchen() {
        let config = Config::default();
        assert_eq!(resolve_family("wuerstchen-v2:fp16", &config), "wuerstchen");
    }

    #[test]
    fn create_engine_wuerstchen() {
        let mut config = Config::default();
        config.models.insert(
            "my-wuerstchen".to_string(),
            mold_core::config::ModelConfig {
                family: Some("wuerstchen".to_string()),
                ..Default::default()
            },
        );
        let engine = create_engine(
            "my-wuerstchen".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            0,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-wuerstchen");
    }

    #[test]
    fn create_engine_wuerstchen_family_alias() {
        let mut config = Config::default();
        config.models.insert(
            "my-wurst".to_string(),
            mold_core::config::ModelConfig {
                family: Some("wuerstchen-v2".to_string()),
                ..Default::default()
            },
        );
        let engine = create_engine(
            "my-wurst".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            0,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-wurst");
    }

    // ----- Single-file routing -----

    fn single_file_paths(checkpoint: &std::path::Path) -> ModelPaths {
        // The factory routing decision keys off transformer == vae +
        // `.safetensors` extension — every other path is a stand-in for
        // companion-pulled tokenizer / encoder assets the factory does
        // not consult when picking the constructor.
        ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: checkpoint.to_path_buf(),
            transformer_shards: vec![],
            vae: checkpoint.to_path_buf(),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: Some(PathBuf::from("/tmp/will-be-clobbered-clip-l")),
            t5_tokenizer: None,
            clip_tokenizer: Some(PathBuf::from("/tmp/clip-l-tokenizer.json")),
            clip_encoder_2: Some(PathBuf::from("/tmp/will-be-clobbered-clip-g")),
            clip_tokenizer_2: Some(PathBuf::from("/tmp/clip-g-tokenizer.json")),
            text_encoder_files: vec![],
            text_tokenizer: None,
            decoder: None,
        }
    }

    #[test]
    fn is_single_file_true_when_transformer_eq_vae_safetensors() {
        let path = PathBuf::from("/tmp/some-civitai-checkpoint.safetensors");
        let paths = single_file_paths(&path);
        assert!(
            super::is_single_file(&paths),
            "transformer == vae + .safetensors must trigger single-file dispatch",
        );
    }

    #[test]
    fn is_single_file_false_when_transformer_neq_vae() {
        // Diffusers-layout shards have transformer + vae as distinct files.
        let mut paths = single_file_paths(&PathBuf::from("/tmp/x.safetensors"));
        paths.vae = PathBuf::from("/tmp/different-vae.safetensors");
        assert!(
            !super::is_single_file(&paths),
            "distinct transformer + vae paths must route to the diffusers `new` constructor",
        );
    }

    #[test]
    fn is_single_file_false_when_extension_not_safetensors() {
        // Diffusers shards with a `.bin` (legacy) or no-extension shard
        // must not be misclassified just because transformer == vae would
        // otherwise match (catches future ModelPaths defaults that point
        // both at the same placeholder dir).
        let mut paths = single_file_paths(&PathBuf::from("/tmp/x.bin"));
        paths.transformer = PathBuf::from("/tmp/x.bin");
        paths.vae = PathBuf::from("/tmp/x.bin");
        assert!(
            !super::is_single_file(&paths),
            "non-`.safetensors` extension must never route to single-file",
        );
    }

    #[test]
    fn ltx2_native_single_file_dispatch_survives_external_vae_path() {
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;

        let temp = tempfile::tempdir().unwrap();
        let checkpoint = temp.path().join("ltx23_transformer.safetensors");
        let zero = 0.0f32.to_le_bytes();
        let mut tensors = HashMap::new();
        tensors.insert(
            "transformer_blocks.0.attn1.to_q.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1], &zero).unwrap(),
        );
        serialize_to_file(&tensors, &None, &checkpoint).unwrap();

        let mut paths = single_file_paths(&checkpoint);
        paths.vae = temp.path().join("external_vae.safetensors");
        assert!(super::is_ltx2_native_single_file(&paths));
        assert!(!super::is_single_file(&paths));
    }

    /// Synthesise a minimal SD1.5-shaped single-file safetensors so the
    /// factory's `from_single_file` dispatch survives header parsing +
    /// `build_sd15_remap` validation. Tensor data is one zero F32 per key
    /// — no model weights are materialised.
    fn synth_sd15_for_factory(name: &str) -> PathBuf {
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;

        let path = std::env::temp_dir().join(format!(
            "mold-factory-sd15-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        let keys: &[&str] = &[
            "model.diffusion_model.input_blocks.0.0.weight",
            "first_stage_model.encoder.down.0.block.0.norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
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

    /// SDXL synthetic checkpoint — adds CLIP-G slabs (incl. the fused QKV
    /// row stack) so `build_sdxl_remap` exercises the `FusedSlice` path at
    /// dispatch time.
    fn synth_sdxl_for_factory(name: &str) -> PathBuf {
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;

        let path = std::env::temp_dir().join(format!(
            "mold-factory-sdxl-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        let keys: &[&str] = &[
            "model.diffusion_model.input_blocks.0.0.weight",
            "first_stage_model.encoder.down.0.block.0.norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
            "conditioner.embedders.1.model.text_projection",
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
    fn factory_sd15_single_file_dispatches_to_from_single_file() {
        let single_file = synth_sd15_for_factory("dispatch");
        let paths = single_file_paths(&single_file);

        let engine = create_engine(
            "sd15:fp16".to_string(),
            paths,
            &Config::default(),
            LoadStrategy::Sequential,
            0,
            false,
        )
        .expect("single-file dispatch must succeed against a valid SD1.5 checkpoint");

        assert_eq!(engine.model_name(), "sd15:fp16");

        // `from_single_file` rebuilds `ModelPaths` so every component
        // resolves through the single file; the original
        // `clip_encoder = /tmp/will-be-clobbered-clip-l` we passed in
        // must be replaced by the checkpoint path. If the factory had
        // dispatched to `SD15Engine::new`, the input `clip_encoder`
        // would have survived verbatim.
        let resolved = engine
            .model_paths()
            .expect("SD15 engine must expose its ModelPaths via the trait");
        assert_eq!(
            resolved.clip_encoder.as_deref(),
            Some(single_file.as_path()),
            "single-file dispatch must materialise clip_encoder from the checkpoint",
        );

        let _ = std::fs::remove_file(single_file);
    }

    #[test]
    fn factory_sd15_diffusers_layout_routes_to_new_constructor() {
        // Distinct transformer + vae paths — the diffusers-layout signal.
        // `SD15Engine::new` does not validate path existence, so this
        // succeeds even with non-existent paths. If the factory had
        // misrouted to `from_single_file`, the missing-file check would
        // surface a `single-file checkpoint not found` error.
        let mut paths = dummy_paths();
        paths.transformer =
            PathBuf::from("/tmp/diffusers/unet/diffusion_pytorch_model.safetensors");
        paths.vae = PathBuf::from("/tmp/diffusers/vae/diffusion_pytorch_model.safetensors");

        let engine = create_engine(
            "sd15:fp16".to_string(),
            paths,
            &Config::default(),
            LoadStrategy::Sequential,
            0,
            false,
        )
        .expect("diffusers-layout SD1.5 must route to `new` and not validate paths");

        // The original `clip_encoder` path must survive untouched —
        // proof we did **not** take the single-file branch.
        let resolved = engine.model_paths().expect("SD15 must expose paths");
        assert_eq!(
            resolved.clip_encoder.as_deref(),
            Some(std::path::Path::new("/tmp/clip")),
        );
    }

    #[test]
    fn factory_sd15_single_file_missing_checkpoint_surfaces_error() {
        // transformer == vae + `.safetensors`, but the file does not
        // exist. The single-file path errors at the existence check; the
        // diffusers `new` path would instead succeed (no I/O at construct
        // time). This test pins down which branch ran.
        let bogus = std::env::temp_dir().join(format!(
            "mold-factory-sd15-missing-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let paths = single_file_paths(&bogus);

        let result = create_engine(
            "sd15:fp16".to_string(),
            paths,
            &Config::default(),
            LoadStrategy::Sequential,
            0,
            false,
        );

        let err = result
            .err()
            .expect("missing single-file checkpoint must surface as constructor error");
        assert!(
            err.to_string().contains("single-file checkpoint not found"),
            "expected single-file error, got: {err}",
        );
    }

    #[test]
    fn factory_sdxl_single_file_dispatches_to_from_single_file() {
        let single_file = synth_sdxl_for_factory("dispatch");
        let paths = single_file_paths(&single_file);

        let engine = create_engine(
            "sdxl-base:fp16".to_string(),
            paths,
            &Config::default(),
            LoadStrategy::Sequential,
            0,
            false,
        )
        .expect("single-file dispatch must succeed against a valid SDXL checkpoint");

        let resolved = engine
            .model_paths()
            .expect("SDXL engine must expose its ModelPaths via the trait");
        // SDXL adds the CLIP-G identity check on top of CLIP-L; both must
        // resolve to the single-file path after dispatch.
        assert_eq!(
            resolved.clip_encoder.as_deref(),
            Some(single_file.as_path()),
            "single-file dispatch must materialise clip_encoder (CLIP-L) from the checkpoint",
        );
        assert_eq!(
            resolved.clip_encoder_2.as_deref(),
            Some(single_file.as_path()),
            "single-file dispatch must materialise clip_encoder_2 (CLIP-G) from the checkpoint",
        );

        let _ = std::fs::remove_file(single_file);
    }

    #[test]
    fn factory_sdxl_single_file_threads_is_turbo_via_model_config() {
        // is_turbo is plumbed through model_cfg → from_single_file.
        // Field-level verification of the threaded value lives in
        // `sdxl/pipeline.rs::tests` where direct field access works; at
        // the factory boundary we only assert that an `is_turbo = true`
        // model config dispatches successfully (which proves the
        // constructor signature accepts the threaded value).
        let single_file = synth_sdxl_for_factory("turbo");
        let paths = single_file_paths(&single_file);

        let mut config = Config::default();
        config.models.insert(
            "sdxl-turbo:fp16".to_string(),
            mold_core::config::ModelConfig {
                family: Some("sdxl".to_string()),
                is_turbo: Some(true),
                ..Default::default()
            },
        );

        let engine = create_engine(
            "sdxl-turbo:fp16".to_string(),
            paths,
            &config,
            LoadStrategy::Sequential,
            0,
            false,
        )
        .expect("is_turbo = true must thread through to the single-file constructor");
        assert_eq!(engine.model_name(), "sdxl-turbo:fp16");

        let _ = std::fs::remove_file(single_file);
    }
}
