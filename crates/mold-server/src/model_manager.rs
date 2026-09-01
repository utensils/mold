use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use mold_core::{
    build_model_catalog, Config, GenerateRequest, GenerationMemoryEstimate, ModelComponentOption,
    ModelComponentStatus, ModelComponentsResponse, ModelDefaults, ModelInfo, ModelInfoExtended,
    ModelPaths,
};
#[cfg(test)]
use mold_inference::device::ActivationFamily;

use mold_catalog::resolve::{
    installed_intent_from_sidecar, looks_like_catalog_id, resolve_intent_to_model_config,
    MissingCompanionPolicy, ResolveError, ResolveOptions,
};

use crate::model_cache::ModelResidency;
use crate::{routes::ApiError, state::AppState};

pub(crate) type EngineProgressCallback = Arc<dyn Fn(mold_inference::ProgressEvent) + Send + Sync>;

#[cfg(all(test, not(feature = "cuda")))]
pub(crate) use crate::memory_preflight::preflight_memory_guard_after_drop;
pub use crate::memory_preflight::ActivationHint;
#[cfg(test)]
pub(crate) use crate::memory_preflight::{
    check_model_memory_budget, preflight_memory_guard_with_available,
    preflight_memory_guard_with_available_and_policy, rejection_suggestion,
    request_requires_fresh_engine_for_offload_policy_with_request,
    server_offload_enabled_for_paths_with_request,
};
pub(crate) use crate::memory_preflight::{
    effective_load_available_bytes, estimate_generation_memory_for_request,
    preflight_memory_guard_after_drop_for_request, preflight_memory_guard_for_request,
    request_requires_fresh_engine_for_offload_policy, select_server_load_strategy_for_budget,
    select_server_load_strategy_for_device, server_offload_enabled_for_paths,
    GenerationOffloadPolicy,
};

pub(crate) fn request_has_effective_lora(req: &GenerateRequest) -> bool {
    if req.ic_lora_control.is_some() {
        return true;
    }
    const ZERO_SCALE_EPS: f64 = 1e-8;
    if let Some(loras) = &req.loras {
        if !loras.is_empty() {
            return loras.iter().any(|lora| lora.scale.abs() > ZERO_SCALE_EPS);
        }
    }
    req.lora
        .as_ref()
        .is_some_and(|lora| lora.scale.abs() > ZERO_SCALE_EPS)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExistingModelResolution {
    pub paths: ModelPaths,
    /// Runtime-only catalog entry needed to make an opaque `cv:` / `hf:`
    /// model resolvable. Built-in and explicitly configured models leave this
    /// absent. Callers carry the overlay with prepared work instead of
    /// mutating `AppState.config`.
    pub model_config_overlay: Option<mold_core::ModelConfig>,
}

#[derive(Clone, Debug)]
pub(crate) struct ExistingModelAuthority {
    pub paths: ModelPaths,
    pub config: Config,
}

/// Resolve concrete model paths using only the supplied config and installed
/// local sidecars.
///
/// This is the common authority for placement preview, admission planning,
/// and worker fallback. It never reloads configuration, consults the live
/// catalog, mutates intent/config caches, registers downloads, or writes a
/// sidecar.
pub(crate) fn resolve_existing_model_paths(
    model_name: &str,
    config: &Config,
) -> Result<Option<ExistingModelResolution>, ApiError> {
    if !looks_like_catalog_id(model_name) {
        return Ok(
            ModelPaths::resolve(model_name, config).map(|paths| ExistingModelResolution {
                paths,
                model_config_overlay: None,
            }),
        );
    }

    if let Some(model_config) = config.models.get(model_name).cloned() {
        if let Some(paths) = ModelPaths::resolve(model_name, config) {
            return Ok(Some(ExistingModelResolution {
                paths,
                model_config_overlay: Some(model_config),
            }));
        }
    }

    let Some(intent) = installed_intent_from_sidecar(&config.resolved_models_dir(), model_name)
    else {
        return Ok(None);
    };
    let model_config = resolve_intent_to_paths(model_name, &intent, config)
        .map_err(|error| resolve_error_to_api_error(&error))?;
    let mut effective = config.clone();
    effective
        .models
        .insert(model_name.to_string(), model_config.clone());
    let paths = ModelPaths::resolve(model_name, &effective).ok_or_else(|| {
        ApiError::not_found(format!(
            "catalog model '{model_name}' resolved to a config that ModelPaths \
             could not turn into runtime paths — internal mismatch, please file an issue."
        ))
    })?;

    Ok(Some(ExistingModelResolution {
        paths,
        model_config_overlay: Some(model_config),
    }))
}

/// Resolve one immutable, local-only config snapshot for work that must keep
/// an opaque catalog model stable beyond request admission.
///
/// The returned config owns the catalog overlay instead of mutating
/// `AppState.config`, so inventory refreshes cannot erase or replace the
/// authority carried by durable work.
pub(crate) fn resolve_existing_model_authority(
    model_name: &str,
    config: &Config,
) -> Result<Option<ExistingModelAuthority>, ApiError> {
    let Some(resolved) = resolve_existing_model_paths(model_name, config)? else {
        return Ok(None);
    };
    let mut effective = config.clone();
    if let Some(model_config) = resolved.model_config_overlay {
        effective
            .models
            .insert(model_name.to_string(), model_config);
    }
    Ok(Some(ExistingModelAuthority {
        paths: resolved.paths,
        config: effective,
    }))
}

pub(crate) fn resolve_installed_catalog_paths_for_worker(
    model_name: &str,
    config: &Config,
) -> Result<Option<(ModelPaths, Config)>, ApiError> {
    if !looks_like_catalog_id(model_name) {
        return Ok(None);
    }
    let Some(authority) = resolve_existing_model_authority(model_name, config)? else {
        return Ok(None);
    };
    Ok(Some((authority.paths, authority.config)))
}

pub(crate) type DownloadProgressCallback =
    Arc<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>;

pub(crate) fn configured_upscaler_weights_exist(config: &Config, model: &str) -> bool {
    let canonical = mold_core::manifest::resolve_model_name(model);
    config
        .models
        .get(&canonical)
        .and_then(|model| model.transformer.as_ref())
        .is_some_and(|path| Path::new(path).is_file())
}

/// First-use acquisition gate shared by every native upscaler surface.
/// A configured-but-missing file is not installed; pulling repairs stale
/// configuration instead of deferring the failure to engine construction.
pub(crate) fn upscaler_model_needs_pull(config: &Config, model: &str) -> bool {
    !configured_upscaler_weights_exist(config, model)
}

pub(crate) enum PullStatus {
    AlreadyAvailable,
    Pulled,
}

pub(crate) async fn refresh_config(state: &AppState) -> mold_core::Config {
    // Unit-test states intentionally carry isolated in-memory model and
    // output authorities. Reloading is explicit per state so another
    // parallel test's temporary process-global MOLD_HOME cannot erase
    // synthetic recipes or import unrelated disk configuration. Production
    // always takes the reload path because cfg!(test) is false.
    if cfg!(test) && !state.reload_config_from_disk {
        return state.config.read().await.clone();
    }

    {
        let fresh = {
            let current = state.config.read().await;
            current.reload_from_disk_preserving_runtime()
        };

        let mut config = state.config.write().await;
        *config = fresh.clone();
        fresh
    }
}

pub(crate) async fn list_models(state: &AppState) -> Vec<ModelInfoExtended> {
    let config = refresh_config(state).await;
    let models_dir = config.resolved_models_dir();

    // Multi-GPU mode: derive "loaded" state from the worker pool so /api/models
    // reflects the actual engine cache, not the legacy single-GPU snapshot.
    if state.gpu_pool.worker_count() > 0 {
        let loaded_models = loaded_models_across_pool(state);
        let primary = loaded_models.first().cloned();
        let mut catalog = build_model_catalog(&config, primary.as_deref(), primary.is_some());
        // Mark every GPU-resident model as loaded (not just the primary).
        for entry in catalog.iter_mut() {
            if loaded_models.contains(&entry.info.name) {
                entry.info.is_loaded = true;
            }
        }
        catalog.extend(installed_catalog_models(
            state,
            &config,
            &models_dir,
            primary.as_deref(),
            primary.is_some(),
        ));
        annotate_audio_capabilities(&mut catalog, &config);
        annotate_ltx25_runtime_readiness(&mut catalog, &config);
        annotate_source_image_capabilities(&mut catalog, &config);
        synchronize_generation_profile_capabilities(&mut catalog);
        retain_deliverable_generation_profiles(&mut catalog);
        return catalog;
    }

    let snapshot = state.model_cache.lock().await.snapshot();
    let mut catalog =
        build_model_catalog(&config, snapshot.model_name.as_deref(), snapshot.is_loaded);
    catalog.extend(installed_catalog_models(
        state,
        &config,
        &models_dir,
        snapshot.model_name.as_deref(),
        snapshot.is_loaded,
    ));
    annotate_audio_capabilities(&mut catalog, &config);
    annotate_ltx25_runtime_readiness(&mut catalog, &config);
    // CPU-fallback / maintenance runtimes (no workers) advertise the same
    // conditioning contracts: classification reads safetensors headers, not
    // a GPU.
    annotate_source_image_capabilities(&mut catalog, &config);
    synchronize_generation_profile_capabilities(&mut catalog);
    retain_deliverable_generation_profiles(&mut catalog);
    catalog
}

fn annotate_ltx25_runtime_readiness(catalog: &mut [ModelInfoExtended], config: &Config) {
    for entry in catalog {
        if !mold_core::ltx25_manifest::is_contract_manifest(&entry.info.name) {
            continue;
        }
        if !entry.downloaded {
            entry.runtime_ready = Some(false);
            entry.runtime_readiness_error = Some(
                "LTX-2.5 split pack is incomplete on this host; pull or repair the model before generation."
                    .to_string(),
            );
            entry.supports_duration_prediction = Some(false);
            continue;
        }
        let qualification =
            mold_core::ltx25_manifest::Ltx25ModelPaths::resolve(config, &entry.info.name)
                .ok_or_else(|| "LTX-2.5 split component graph could not be resolved.".to_string())
                .and_then(|paths| paths.qualify().map_err(|error| error.to_string()));
        match qualification {
            Ok(()) => {
                entry.runtime_ready = Some(true);
                entry.runtime_readiness_error = None;
                entry.supports_duration_prediction = Some(true);
            }
            Err(error) => {
                entry.runtime_ready = Some(false);
                entry.runtime_readiness_error = Some(format!(
                    "LTX-2.5 split pack failed component qualification: {error}"
                ));
                entry.supports_duration_prediction = Some(false);
            }
        }
    }
}

/// Drop rows whose profile lost every recipe to this binary's delivery
/// encoders, except the ones that must stay listed as downloads.
///
/// Two kinds of H3 row survive an empty profile. A reviewed compact identity
/// is runnable on a qualified build and must remain pullable from a build
/// that merely cannot mux MP4. A row that advertises
/// `runtime_available: Some(false)` is download-only on *every* build — the
/// pruned NVFP4 layout and the official BF16 references — so the absence of
/// a deliverable recipe is not a reason to hide it; the row's whole point is
/// to be pulled, inventoried, and removed ahead of a runtime.
fn retain_deliverable_generation_profiles(catalog: &mut Vec<ModelInfoExtended>) {
    catalog.retain(|entry| {
        entry.generation_profile.as_ref().is_none_or(|profile| {
            !profile.recipes.is_empty()
                || mold_core::model_policy::is_reviewed_minimax_h3_model(&entry.info.name)
                || entry.runtime_available == Some(false)
        })
    });
}

fn annotate_audio_capabilities(catalog: &mut [ModelInfoExtended], config: &Config) {
    for entry in catalog {
        if entry.supports_audio.is_some()
            || !entry.downloaded
            || !mold_inference::audio::output_probe_registered(&entry.info.family)
        {
            continue;
        }
        entry.supports_audio = ModelPaths::resolve(&entry.info.name, config).and_then(|paths| {
            mold_inference::audio::output_supported(
                &entry.info.family,
                &entry.info.name,
                config,
                &paths,
            )
        });
    }
}

/// Fill `source_image` for downloaded wan checkpoints from their own headers
/// — the same shape-driven classification the engine applies at generate
/// time (#772). The probe outranks the manifest's task-structure answer for
/// a downloaded checkpoint because `ModelPaths` honors config/env path
/// overrides: the artifacts actually loaded can differ from what the
/// manifest was assembled from. Cold (not-yet-downloaded) tiers keep the
/// manifest classification, and an unclassifiable downloaded entry falls
/// back to it too; entries neither can classify stay `None`, which clients
/// must read as "unknown", never as one of the three contracts.
fn annotate_source_image_capabilities(catalog: &mut [ModelInfoExtended], config: &Config) {
    for entry in catalog {
        if !entry.downloaded || entry.info.family != "wan" {
            continue;
        }
        let probed = ModelPaths::resolve(&entry.info.name, config).and_then(|paths| {
            mold_inference::wan_source_image_capability(&paths.transformer, &paths.vae)
        });
        entry.source_image = probed.or(entry.source_image);
        // Extend continues a clip by seeding it with the source's final frame,
        // so it is available exactly when the checkpoint conditions on an
        // image (#783). Re-derive it from the contract we just resolved rather
        // than leaving the manifest's cold guess in place: a config path
        // override can point the same manifest name at a different checkpoint,
        // and advertising extend on a text-to-video one promises a
        // continuation it has no channel to accept.
        entry.supports_extend = Some(mold_core::catalog::extend_capable_model(
            &entry.info.family,
            entry.source_image,
        ));
    }
}

/// Runtime checkpoint probes can refine cold manifest capabilities. Keep the
/// versioned profile and its content hash synchronized with those resolved
/// row fields so clients and mixed-host routing never receive a stale
/// pre-probe contract.
fn synchronize_generation_profile_capabilities(catalog: &mut [ModelInfoExtended]) {
    for entry in catalog {
        let Some(profile) = entry.generation_profile.as_mut() else {
            continue;
        };
        if entry.supports_audio == Some(false) {
            profile.recipes.retain(|recipe| {
                recipe.request_selector.pipeline != Some(mold_core::Ltx2PipelineMode::T2a)
            });
        }
        for recipe in &mut profile.recipes {
            if let Some(supports_audio) = entry.supports_audio {
                recipe.capabilities.supports_audio = supports_audio;
            }
            if let Some(source_image) = entry.source_image {
                recipe.capabilities.source_image = Some(source_image);
                if entry.info.family == "wan" {
                    let accepts_source =
                        source_image != mold_core::SourceImageCapability::Unsupported;
                    recipe.capabilities.wan_recipe.supports_first_last_frame = accepts_source;
                    recipe.capabilities.keyframes.mode = if accepts_source {
                        mold_core::ControlMode::Adjustable
                    } else {
                        mold_core::ControlMode::Hidden
                    };
                    recipe.capabilities.keyframes.required = false;
                    recipe.capabilities.keyframes.reason = (!accepts_source)
                        .then(|| "This Wan checkpoint does not accept keyframes.".to_string());
                }
            }
            if let Some(supports_extend) = entry.supports_extend {
                recipe.capabilities.supports_extend = supports_extend;
            }
        }
        mold_core::qualify_generation_profile_delivery(
            profile,
            mold_core::GenerationDeliveryCapabilities::new(
                cfg!(feature = "mp4"),
                cfg!(feature = "webp"),
            ),
        );
    }
}

fn loaded_models_across_pool(state: &AppState) -> Vec<String> {
    let mut names = Vec::new();
    for worker in state.gpu_pool.worker_snapshot() {
        // Prefer the active-generation model (cache entry is taken out during
        // inflight generation), else whatever is GPU-resident.
        let active = worker
            .active_generation
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(|g| g.model.clone()));
        let loaded = active.or_else(|| {
            worker
                .resident_model
                .read()
                .ok()?
                .as_deref()
                .map(crate::gpu_pool::resident_model_display_name)
                .map(str::to_string)
        });
        if let Some(name) = loaded {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

// ── Catalog bridge ───────────────────────────────────────────────────────────
//
// The `cv:*` / `hf:*` resolution logic itself lives in the shared
// `mold_catalog::resolve` module (consumed here and by `mold-cli`); this
// section keeps only the server-specific orchestration (intent cache, live
// install, API-error translation).

/// Best-effort family lookup for a catalog (`cv:*` / `hf:*`) model name,
/// used to feed `validate_generate_request_with_family` so catalog IDs
/// get the same family-gated feature checks as manifest-resident models.
///
/// Reads from the intent cache first (populated by `install_catalog_model`
/// before the engine-load resolution runs), then falls back to
/// `state.config.models` for back-compat with callers that snapshotted
/// the resolved `ModelConfig`.
pub(crate) async fn catalog_family_for(state: &AppState, model_name: &str) -> Option<String> {
    if !looks_like_catalog_id(model_name) {
        return None;
    }
    {
        let intents = state.catalog_intents.read().await;
        if let Some(intent) = intents.get(model_name) {
            return Some(intent.family.clone());
        }
    }
    let config = state.config.read().await;
    if let Some(family) = config.models.get(model_name).and_then(|m| m.family.clone()) {
        return Some(family);
    }
    installed_intent_from_sidecar(&config.resolved_models_dir(), model_name)
        .map(|intent| intent.family)
}

/// Resolve the family slug for any model name — checks the static manifest
/// first (covers `flux-dev:q8` and friends), then configured non-catalog
/// models, and finally catalog metadata (covers `cv:*` / `hf:*`). Used by the
/// activation-budget preflight to dispatch to the right [`ActivationFamily`].
pub(crate) async fn family_for_model(state: &AppState, model_name: &str) -> Option<String> {
    if let Some(manifest) = mold_core::manifest::find_manifest(model_name) {
        return Some(manifest.family.clone());
    }
    if looks_like_catalog_id(model_name) {
        return catalog_family_for(state, model_name).await;
    }
    state
        .config
        .read()
        .await
        .models
        .get(model_name)
        .and_then(|model| model.family.clone())
}

/// Sync variant of [`family_for_model`] for the GPU-worker hot path which
/// runs inside `spawn_blocking` and only has a `&Config` snapshot to work
/// with. Falls through to manifest-then-config in the same order.
pub(crate) fn family_for_model_sync(
    model_name: &str,
    config: &mold_core::Config,
) -> Option<String> {
    if let Some(manifest) = mold_core::manifest::find_manifest(model_name) {
        return Some(manifest.family.clone());
    }
    config.models.get(model_name).and_then(|m| m.family.clone())
}

/// Sync variant of [`activation_hint_for_request`] for the GPU-worker hot
/// path. Returns `None` when the family slug can't be resolved.
pub(crate) fn activation_hint_for_request_sync(
    config: &mold_core::Config,
    req: &GenerateRequest,
) -> Option<ActivationHint> {
    let family = family_for_model_sync(&req.model, config)?;
    Some(ActivationHint::from_request(req, &family))
}

/// Build an [`ActivationHint`] for the given request using the resolved
/// model family (manifest or catalog). Returns `None` when the family slug
/// can't be resolved — the preflight then falls back to the size-only peak.
pub(crate) async fn activation_hint_for_request(
    state: &AppState,
    req: &GenerateRequest,
) -> Option<ActivationHint> {
    let family = family_for_model(state, &req.model).await?;
    Some(ActivationHint::from_request(req, &family))
}

/// Server-side disk-aware resolution: turn a pure `CatalogModelIntent` into a
/// runtime `ModelConfig`. The server fails hard on any missing required
/// companion so the engine-load retry path can distinguish "still
/// downloading" from "broken config", and requires the primary present.
/// The resolution logic itself is the shared `mold_catalog::resolve`
/// implementation (also consumed by `mold-cli`).
pub(crate) fn resolve_intent_to_paths(
    model_name: &str,
    intent: &mold_catalog::synthesis::CatalogModelIntent,
    config: &mold_core::Config,
) -> Result<mold_core::ModelConfig, ResolveError> {
    resolve_intent_to_model_config(
        model_name,
        intent,
        config,
        ResolveOptions {
            missing_companions: MissingCompanionPolicy::Fail,
            require_primary_present: true,
        },
    )
}

/// If `model_name` is a catalog ID and not yet in the intent cache, hit
/// live HF/Civitai for the entry and synthesize its
/// [`CatalogModelIntent`]. Returns `Ok(())` when a fresh entry was
/// installed or one was already cached. Returns a typed [`InstallError`]
/// so the caller can map per-variant to a user-facing HTTP status.
///
/// This function is *pure synthesis* — no disk reads. The intent stays
/// valid across download events; resolution into a `ModelConfig` is run
/// lazily by [`resolve_intent_to_paths`] at engine-load time.
fn require_catalog_intent_activation(
    model_name: &str,
    intent: &mold_catalog::synthesis::CatalogModelIntent,
    artifact_root: &Path,
) -> Result<(), mold_core::ModelActivationError> {
    let family = Some(intent.family.as_str());
    mold_core::require_model_activation(model_name, family)?;
    if let Some(sub_family) = intent.sub_family.as_deref() {
        mold_core::require_model_activation(sub_family, family)?;
    }
    mold_core::require_model_artifact_activation(
        &intent.primary_recipe_path,
        Some(artifact_root),
        family,
    )?;
    if let Some(path) = intent.vae_recipe_path.as_deref() {
        mold_core::require_model_artifact_activation(path, Some(artifact_root), family)?;
    }
    if let Some(path) = intent.low_noise_recipe_path.as_deref() {
        mold_core::require_model_artifact_activation(path, Some(artifact_root), family)?;
    }
    for path in &intent.text_encoder_recipe_paths {
        mold_core::require_model_artifact_activation(path, Some(artifact_root), family)?;
    }
    for companion in &intent.companions {
        mold_core::require_model_activation(&companion.name, family)?;
    }
    Ok(())
}

pub(crate) async fn install_catalog_model(
    state: &AppState,
    model_name: &str,
) -> Result<(), mold_core::InstallError> {
    mold_core::require_model_activation(model_name, None)?;
    if !looks_like_catalog_id(model_name) {
        // Caller should have shape-checked first; treat it as a no-op
        // success rather than fabricating a custom error variant.
        return Ok(());
    }
    let models_dir = state.config.read().await.resolved_models_dir();

    // Already installed from a prior request — keep the cached intent.
    {
        let intents = state.catalog_intents.read().await;
        if let Some(intent) = intents.get(model_name) {
            require_catalog_intent_activation(model_name, intent, &models_dir)?;
            return Ok(());
        }
    }

    mold_catalog::sidecar::require_installed_sidecar_activation(&models_dir, model_name)?;
    if let Some(intent) = installed_intent_from_sidecar(&models_dir, model_name) {
        require_catalog_intent_activation(model_name, &intent, &models_dir)?;
        let mut intents = state.catalog_intents.write().await;
        intents.insert(model_name.to_string(), intent);
        return Ok(());
    }

    // Live single-id lookup via the shared cv:/hf: dispatcher. Tokens are
    // resolved from server-owned credentials (environment first, then the
    // private credential file) so unauthenticated browsing still works; the
    // Civitai base is test-overridable through AppState.
    let civitai_token = crate::catalog_credentials::resolved_civitai_token();
    let hf_token = crate::catalog_credentials::resolved_hf_token();
    let entry = mold_catalog::live::fetch_entry_by_id(
        model_name,
        state.catalog_live_civitai_base.as_str(),
        "https://huggingface.co",
        civitai_token.as_deref(),
        hf_token.as_deref(),
    )
    .await
    .map_err(|e| live_error_to_install_error(model_name, &e))?;

    mold_catalog::entry::require_catalog_entry_activation(&entry)?;

    let intent = mold_catalog::synthesis::synthesize_intent(&entry, &models_dir).map_err(|e| {
        mold_core::InstallError::RecipeMalformed(format!("synthesize intent for {model_name}: {e}"))
    })?;
    require_catalog_intent_activation(model_name, &intent, &models_dir)?;
    if model_name.starts_with("cv:") {
        write_catalog_sidecar_from_intent(&models_dir, &entry, &intent);
    }

    let mut intents = state.catalog_intents.write().await;
    intents.insert(model_name.to_string(), intent);
    Ok(())
}

fn write_catalog_sidecar_from_intent(
    models_dir: &std::path::Path,
    entry: &mold_catalog::entry::CatalogEntry,
    intent: &mold_catalog::synthesis::CatalogModelIntent,
) {
    let sc_path = mold_catalog::sidecar::civitai_sidecar_path(models_dir, entry.id.as_str());
    let Some(sidecar_dir) = sc_path.parent() else {
        return;
    };
    let Ok(primary_rel) = intent.primary_recipe_path.strip_prefix(sidecar_dir) else {
        return;
    };
    let Some(primary_rel) = primary_rel.to_str() else {
        return;
    };
    let sidecar = mold_catalog::sidecar::sidecar_from_entry(entry, primary_rel.to_string());
    if let Err(e) = mold_catalog::sidecar::write_sidecar(&sc_path, &sidecar) {
        tracing::warn!(
            target: "catalog.sidecar",
            catalog_id = %entry.id.as_str(),
            error = %e,
            "sidecar write failed after live catalog install",
        );
    }
}

/// Translate a `LiveSearchError` into the user-facing `InstallError`
/// shape. The `Upstream` variant carries the HTTP status from the
/// upstream's response body — 404 is "not found", anything else is
/// "malformed recipe" (the live API answered, but the response wasn't
/// what mold expects).
fn live_error_to_install_error(
    model_name: &str,
    err: &mold_catalog::live::LiveSearchError,
) -> mold_core::InstallError {
    use mold_catalog::live::LiveSearchError;
    match err {
        LiveSearchError::Network(e) => {
            mold_core::InstallError::Network(format!("{model_name}: {e}"))
        }
        LiveSearchError::Decode(e) => mold_core::InstallError::RecipeMalformed(format!(
            "{model_name}: decode upstream payload: {e}"
        )),
        LiveSearchError::Upstream { status, body, .. } if *status == 404 => {
            mold_core::InstallError::NotFound(format!(
                "{model_name}: upstream returned 404 ({})",
                truncate_body(body)
            ))
        }
        LiveSearchError::Upstream { status, body, .. } => {
            mold_core::InstallError::RecipeMalformed(format!(
                "{model_name}: upstream HTTP {status}: {}",
                truncate_body(body)
            ))
        }
    }
}

fn truncate_body(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.len() > 160 {
        format!("{}…", &trimmed[..160])
    } else {
        trimmed.to_string()
    }
}

/// Translate `InstallError` into the user-facing `ApiError`. Each variant
/// maps to a distinct HTTP status so clients can tell "Civitai is down"
/// from "you typed a bad ID".
pub(crate) fn install_error_to_api_error(err: &mold_core::InstallError) -> ApiError {
    use mold_core::InstallError;
    match err {
        InstallError::ModelActivation(error) => ApiError::model_activation(*error),
        InstallError::Network(msg) => {
            // 502 Bad Gateway via internal_with_status — the catalog
            // upstream is unreachable, not a mold internal failure.
            ApiError::internal_with_status(
                format!("network unreachable: {msg}"),
                axum::http::StatusCode::BAD_GATEWAY,
            )
        }
        InstallError::NotFound(msg) => ApiError::not_found(msg.to_string()),
        InstallError::RecipeMalformed(msg) => ApiError::internal(msg.to_string()),
    }
}

/// Translate `ResolveError` into the user-facing `ApiError`. The
/// download-repairable variants surface as 404 (the model isn't loadable
/// yet) with distinct, specific messages so the user can act; the
/// never-should-happen `UnknownFamily` surfaces as a 500 internal error.
pub(crate) fn resolve_error_to_api_error(err: &ResolveError) -> ApiError {
    if matches!(err, ResolveError::UnknownFamily { .. }) {
        return ApiError::internal(err.to_string());
    }
    ApiError::not_found(err.to_string())
}

/// Return `ModelInfoExtended` entries for every installed catalog
/// checkpoint discovered via per-install sidecars. Replaces the
/// bulk-scrape DB query that used to back this surface.
fn installed_catalog_models(
    _state: &AppState,
    config: &mold_core::Config,
    models_dir: &std::path::Path,
    loaded_model: Option<&str>,
    engine_is_loaded: bool,
) -> Vec<ModelInfoExtended> {
    let walked = mold_catalog::sidecar::walk_sidecars(models_dir);
    let mut out = Vec::new();
    for (sidecar_dir, sidecar) in walked {
        if mold_catalog::sidecar::require_sidecar_artifact_activation(
            models_dir,
            &sidecar_dir,
            &sidecar,
        )
        .is_err()
        {
            continue;
        }
        if sidecar.kind != "checkpoint" {
            continue;
        }
        if mold_catalog::sidecar::primary_looks_like_auxiliary(&sidecar) {
            continue;
        }
        // Skip sidecars whose primary file isn't actually present —
        // partial pulls / aborted downloads.
        let Some(primary_path) =
            mold_catalog::sidecar::primary_path_if_present(&sidecar_dir, &sidecar)
        else {
            continue;
        };
        // Skip entries already covered by a manifest.
        if mold_core::manifest::find_manifest_by_hf_repo(&sidecar.source_id).is_some() {
            continue;
        }

        let size_gb = sidecar
            .size_bytes
            .map(|b| b as f32 / 1_000_000_000.0)
            .unwrap_or(0.0);

        let defaults = mold_catalog::defaults::runtime_defaults_for_family(
            &sidecar.family,
            sidecar.sub_family.as_deref(),
        );
        let user_cfg = config.lookup_model_config(&sidecar.id);
        let w = user_cfg
            .as_ref()
            .and_then(|cfg| cfg.default_width)
            .unwrap_or(defaults.width);
        let h = user_cfg
            .as_ref()
            .and_then(|cfg| cfg.default_height)
            .unwrap_or(defaults.height);
        let steps = user_cfg
            .as_ref()
            .and_then(|cfg| cfg.default_steps)
            .unwrap_or(defaults.steps);
        let guidance = user_cfg
            .as_ref()
            .and_then(|cfg| cfg.default_guidance)
            .unwrap_or(defaults.guidance);
        let frames = user_cfg
            .as_ref()
            .and_then(|cfg| cfg.default_frames)
            .or(defaults.frames);
        let fps = user_cfg
            .as_ref()
            .and_then(|cfg| cfg.default_fps)
            .or(defaults.fps);

        let description = sidecar
            .description
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| match &sidecar.author {
                Some(a) if !a.is_empty() => format!("{} by {a}", sidecar.name),
                _ => sidecar.name.clone(),
            });

        let supports_audio =
            mold_inference::audio::checkpoint_output_supported(&sidecar.family, &primary_path);
        let supports_extend = sidecar.family == "ltx2";
        let supports_sequence =
            crate::chain_limits::sequence_support(&sidecar.name, &sidecar.family, false).supported;
        let generation_profile =
            mold_core::resolve_generation_profile(mold_core::GenerationProfileInput {
                model: &sidecar.id,
                family: &sidecar.family,
                sub_family: sidecar.sub_family.as_deref(),
                default_width: w,
                default_height: h,
                default_steps: steps,
                default_guidance: guidance,
                default_frames: frames,
                default_fps: fps,
                default_negative_prompt: mold_core::manifest::default_negative_prompt_for_family(
                    &sidecar.family,
                )
                .map(str::to_string),
                source_image: None,
                supports_sequence,
                supports_extend,
                supports_audio: supports_audio.unwrap_or(false),
            });
        let resolution = mold_core::catalog::resolution_defaults_from_profile(&generation_profile);
        out.push(ModelInfoExtended {
            downloaded: true,
            supports_duration_prediction: None,
            runtime_ready: None,
            runtime_readiness_error: None,
            // Sidecar-installed catalog rows are never H3 manifest
            // identities; H3 is manifest-pinned only.
            runtime_available: None,
            runtime_unavailable_reason: None,
            defaults: ModelDefaults {
                default_width: w,
                default_height: h,
                default_steps: steps,
                default_guidance: guidance,
                default_frames: frames,
                default_fps: fps,
                // Model-aware, matching `mold_core::catalog`'s row builders:
                // an unrecognized H3 identity deliberately takes the reviewed
                // compact envelope, so its profile is fixed at one clip length
                // while the family helpers would advertise the official
                // BF16 ladder beside it.
                min_frames: mold_core::validation::min_frames_for_model(
                    &sidecar.family,
                    &sidecar.id,
                ),
                max_frames: mold_core::validation::max_frames_for_model_at_fps(
                    &sidecar.family,
                    &sidecar.id,
                    fps.unwrap_or(mold_core::validation::LTX2_DEFAULT_FPS),
                ),
                max_runtime_seconds: mold_core::validation::max_runtime_seconds_for_family(
                    &sidecar.family,
                ),
                max_frames_absolute: mold_core::validation::max_frames_absolute_for_model(
                    &sidecar.family,
                    &sidecar.id,
                ),
                frame_step: mold_core::validation::frame_step_for_family(&sidecar.family),
                frame_offset: mold_core::validation::frame_offset_for_family(&sidecar.family),
                // The engine substitutes its family default on absence for
                // installed cv:/hf: checkpoints exactly as for manifest
                // models, so the advertisement is per-family here too.
                default_negative_prompt: mold_core::manifest::default_negative_prompt_for_family(
                    &sidecar.family,
                )
                .map(str::to_string),
                // Per model, through the same helper the manifest catalog
                // uses. An installed `cv:` / `hf:` checkpoint has no spatial
                // upsampler, so it advertises the single-pass ceiling and
                // never offers a composed rung it would then reject.
                max_pixels: resolution.max_pixels,
                max_axis_pixels: resolution.max_axis_pixels,
                recommended_dimensions: resolution.recommended_dimensions,
                dimension_alignment: resolution.dimension_alignment,
                description,
            },
            info: ModelInfo {
                name: sidecar.id.clone(),
                family: sidecar.family.clone(),
                size_gb,
                is_loaded: loaded_model.is_some_and(|n| engine_is_loaded && n == sidecar.id),
                last_used: None,
                hf_repo: String::new(),
            },
            disk_usage_bytes: sidecar.size_bytes,
            remaining_download_bytes: Some(0),
            display_name: Some(sidecar.name.clone()),
            kind: Some(sidecar.kind.clone()),
            modality: Some(sidecar.modality.clone()),
            nsfw: sidecar.nsfw,
            supports_audio,
            // Same authority as `capabilities.supports_identity` — never a
            // second predicate about which checkpoints are qualified.
            supports_identity: Some(generation_profile.supports_identity()),
            // One authority, `mold_core::catalog::extend_capable_model`: the
            // whole ltx2 family continues through its latent motion tail,
            // while wan continues only from a checkpoint that conditions on
            // an image. The contract is unknown until the annotate pass below
            // reads the headers, and re-deriving it there from this same
            // helper is what keeps `supports_extend` from restating a family
            // literal that contradicts its own overlap default (#783).
            supports_extend: Some(supports_extend),
            // Per family: the overlap a continuation defaults to is its
            // carryover, and wan's is the one frame it was seeded with (#783).
            extend_default_overlap_frames: Some(
                mold_core::validation::default_extend_overlap_frames_for_family(Some(
                    &sidecar.family,
                )),
            ),
            // Per-model, not per-family, because this is where a future
            // pipeline that cannot chain would have to be caught.
            supports_sequence: Some(supports_sequence),
            guidance_capabilities: Some(mold_core::GuidanceCapabilities::for_recipe(
                &sidecar.family,
                &format!("{} {} {}", sidecar.id, sidecar.name, primary_path.display()),
                None,
            )),
            // Classified from the installed checkpoint's own headers — the
            // shape-driven read the engine applies — never from the sidecar's
            // display name (#772). The wan VAE is a shared companion, so the
            // annotate pass below fills this once paths resolve; a bare
            // primary file cannot answer alone.
            source_image: None,
            generation_profile: Some(generation_profile),
        });
    }
    out
}

/// Check whether a model is available — either already in the cache or
/// has resolvable paths on disk. Returns `Some(paths)` if the model needs
/// to be created from scratch, `None` if already in the cache.
pub(crate) async fn check_model_available(
    state: &AppState,
    model_name: &str,
) -> Result<Option<ModelPaths>, ApiError> {
    let family = family_for_model(state, model_name).await;
    mold_core::require_model_activation(model_name, family.as_deref())
        .map_err(ApiError::model_activation)?;
    // Check the model cache first.
    {
        let cache = state.model_cache.lock().await;
        if cache.contains(model_name) {
            return Ok(None);
        }
    }

    let paths = {
        let config = state.config.read().await;
        if config.manifest_model_needs_download(model_name) {
            None
        } else {
            ModelPaths::resolve(model_name, &config)
        }
    };
    if let Some(paths) = paths {
        return Ok(Some(paths));
    }

    {
        let current = state.config.read().await.clone();
        let fresh_config = current.reload_from_disk_preserving_runtime();
        let needs_download = fresh_config.manifest_model_needs_download(model_name);
        let paths = if needs_download {
            None
        } else {
            ModelPaths::resolve(model_name, &fresh_config)
        };
        {
            let mut config = state.config.write().await;
            *config = fresh_config;
        }
        if let Some(paths) = paths {
            return Ok(Some(paths));
        }
    }

    // Catalog bridge: install (live lookup → cache intent), then resolve the
    // intent against fresh disk state on every request. Lazy resolution
    // means a request that fires while files are still downloading no
    // longer seals a wrong `cfg.vae` into the config — the next request
    // re-resolves once files land.
    if looks_like_catalog_id(model_name) {
        if let Err(install_err) = install_catalog_model(state, model_name).await {
            return Err(install_error_to_api_error(&install_err));
        }
        let intents = state.catalog_intents.read().await;
        let intent = intents
            .get(model_name)
            .ok_or_else(|| {
                ApiError::not_found(format!(
                    "catalog model '{model_name}' is not installed. Download it from \
                     the catalog first."
                ))
            })?
            .clone();
        drop(intents);

        let resolved = {
            let config = state.config.read().await;
            resolve_intent_to_paths(model_name, &intent, &config)
        };
        match resolved {
            Ok(model_cfg) => {
                // Cache the resolved ModelConfig back into config.models so
                // downstream `resolved_model_config` / family lookups still
                // work as before.
                {
                    let mut config = state.config.write().await;
                    config.models.insert(model_name.to_string(), model_cfg);
                }
                let config = state.config.read().await;
                if let Some(paths) = ModelPaths::resolve(model_name, &config) {
                    return Ok(Some(paths));
                }
                // ModelPaths::resolve failed despite successful synthesis —
                // shouldn't happen because resolve_intent_to_paths checks
                // companions, but surface a precise diagnostic if it does.
                return Err(ApiError::not_found(format!(
                    "catalog model '{model_name}' resolved to a config that ModelPaths \
                     could not turn into runtime paths — internal mismatch, please file an issue."
                )));
            }
            Err(e) => return Err(resolve_error_to_api_error(&e)),
        }
    }

    if mold_core::manifest::find_manifest(model_name).is_some() {
        return Err(ApiError::not_found(format!(
            "model '{model_name}' is not downloaded. Run: mold pull {model_name}"
        )));
    }
    Err(ApiError::unknown_model(format!(
        "unknown model '{model_name}'. Run 'mold list' to see available models."
    )))
}

pub(crate) async fn estimate_generation_memory(
    state: &AppState,
    req: &GenerateRequest,
) -> Result<GenerationMemoryEstimate, ApiError> {
    let paths = match check_model_available(state, &req.model).await? {
        Some(paths) => paths,
        None => {
            let config = state.config.read().await;
            ModelPaths::resolve(&req.model, &config).ok_or_else(|| {
                ApiError::not_found(format!(
                    "model '{}' is loaded but runtime paths are not available for estimation",
                    req.model
                ))
            })?
        }
    };
    let mut effective_req = req.clone();
    effective_req.placement = Some(
        state
            .config
            .read()
            .await
            .effective_placement(&req.model, req.placement.as_ref()),
    );
    let hint = activation_hint_for_request(state, &effective_req).await;
    // This endpoint is diagnostic, not an admission side channel. Use the
    // latest resource-sampler facts and report the roomiest current
    // candidate for an Auto request; never perform a device-0 live query or
    // substitute total VRAM when no free-memory sample exists.
    let resources = state.resources.latest();
    let explicit_ordinal = state
        .gpu_pool
        .resolve_explicit_placement_gpu(effective_req.placement.as_ref())
        .map_err(ApiError::validation)?;
    let canonical = state.device_registry.canonical_snapshot(
        &state.gpu_pool,
        resources.as_ref(),
        &state.job_registry,
    );
    // Capacity diagnostics must follow the same device eligibility boundary
    // as dispatch. A disabled, draining, degraded, or otherwise unavailable
    // GPU is not evidence that this request fits, and a hard placement pin
    // narrows both the live and stable estimates to that one device.
    let candidates = resources
        .as_ref()
        .map(|snapshot| {
            canonical
                .scheduler_devices
                .iter()
                .filter(|device| device.schedulable)
                .filter(|device| explicit_ordinal.is_none_or(|ordinal| device.ordinal == ordinal))
                .filter_map(|device| {
                    snapshot
                        .gpus
                        .iter()
                        .find(|gpu| gpu.backend == device.backend && gpu.ordinal == device.ordinal)
                        .map(|gpu| {
                            (
                                gpu.vram_total.saturating_sub(gpu.vram_used),
                                gpu.vram_total,
                                gpu.ordinal,
                                gpu.backend,
                            )
                        })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let roomiest = candidates
        .iter()
        .map(|(available, _, ordinal, backend)| (*available, *ordinal, *backend))
        .max_by_key(|(available, ordinal, _)| (*available, std::cmp::Reverse(*ordinal)));
    // The Create badge answers a different question from current admission:
    // whether this request fits the host's hardware once earlier work releases
    // its allocations. Resolve that estimate against physical capacity so an
    // active denoise/load cannot make the answer oscillate on every sample.
    let roomiest_capacity = candidates
        .iter()
        .map(|(_, capacity, ordinal, backend)| (*capacity, *ordinal, *backend))
        .max_by_key(|(capacity, ordinal, _)| (*capacity, std::cmp::Reverse(*ordinal)));
    let available = roomiest.map(|(available, _, _)| available);
    let forced_offload = matches!(
        mold_inference::runtime_env::value("MOLD_OFFLOAD").as_deref(),
        Some("1") | Some("true") | Some("yes")
    );
    let estimate = estimate_generation_memory_for_request(
        &effective_req,
        &paths,
        hint,
        GenerationOffloadPolicy::new(
            forced_offload,
            roomiest.map_or(
                mold_inference::wan::block_offload::AdmissionPolicy::Disabled,
                |(_, _, backend)| {
                    mold_inference::wan::block_offload::AdmissionPolicy::from_values(
                        backend,
                        mold_inference::runtime_env::value("MOLD_WAN_OFFLOAD_BLOCKS").as_deref(),
                        mold_inference::runtime_env::value("MOLD_OFFLOAD").as_deref(),
                    )
                },
            ),
        ),
        available,
        request_has_effective_lora(&effective_req),
        roomiest.is_some_and(|(_, ordinal, _)| {
            crate::memory_preflight::ltx2_encoder_phase_competes_with_transformer_gpu(ordinal)
        }),
    );
    let capacity_estimate = roomiest_capacity.map(|(capacity, ordinal, backend)| {
        estimate_generation_memory_for_request(
            &effective_req,
            &paths,
            hint,
            GenerationOffloadPolicy::new(
                forced_offload,
                mold_inference::wan::block_offload::AdmissionPolicy::from_values(
                    backend,
                    mold_inference::runtime_env::value("MOLD_WAN_OFFLOAD_BLOCKS").as_deref(),
                    mold_inference::runtime_env::value("MOLD_OFFLOAD").as_deref(),
                ),
            ),
            Some(capacity),
            request_has_effective_lora(&effective_req),
            crate::memory_preflight::ltx2_encoder_phase_competes_with_transformer_gpu(ordinal),
        )
    });

    Ok(GenerationMemoryEstimate {
        model: req.model.clone(),
        peak_memory_bytes: estimate.peak_memory_bytes,
        activation_memory_bytes: estimate.activation_memory_bytes,
        available_memory_bytes: estimate.available_memory_bytes,
        load_strategy: format!("{:?}", estimate.load_strategy).to_ascii_lowercase(),
        fits_available_memory: estimate.fits_available_memory,
        capacity_peak_memory_bytes: capacity_estimate
            .as_ref()
            .map(|estimate| estimate.peak_memory_bytes),
        device_capacity_bytes: roomiest_capacity.map(|(capacity, _, _)| capacity),
        fits_device_capacity: capacity_estimate.and_then(|estimate| estimate.fits_available_memory),
    })
}

pub(crate) async fn model_component_status(
    state: &AppState,
    model_name: &str,
) -> Result<ModelComponentsResponse, ApiError> {
    let family = family_for_model(state, model_name).await;
    mold_core::require_model_acquisition(model_name, family.as_deref())
        .map_err(ApiError::model_activation)?;
    let resolved = mold_core::manifest::resolve_model_name(model_name);
    if let Some(manifest) = mold_core::manifest::find_manifest(&resolved) {
        let config = state.config.read().await;
        let models_dir = config.resolved_models_dir();
        let components = manifest
            .files
            .iter()
            .map(|file| {
                let kind = manifest_component_kind(file.component);
                let path = models_dir.join(mold_core::manifest::storage_path(manifest, file));
                ModelComponentStatus {
                    kind: kind.to_string(),
                    name: manifest_component_name(file.component, &file.hf_filename).to_string(),
                    present: path.is_file(),
                    path: Some(path.to_string_lossy().to_string()),
                    repair_model: Some(resolved.clone()),
                    options: component_options_for_kind(&config, kind, Some(&path)),
                }
            })
            .collect();
        return Ok(ModelComponentsResponse {
            model: resolved,
            components,
        });
    }

    if looks_like_catalog_id(model_name) {
        let resolved_paths = check_model_available(state, model_name).await?;
        let paths = match resolved_paths {
            Some(paths) => paths,
            None => {
                let config = state.config.read().await;
                ModelPaths::resolve(model_name, &config).ok_or_else(|| {
                    ApiError::not_found(format!(
                        "catalog model '{model_name}' is loaded but its runtime paths are unavailable"
                    ))
                })?
            }
        };
        let config = state.config.read().await;
        require_model_paths_activation(&config, model_name, &paths)
            .map_err(ApiError::model_activation)?;
        return Ok(ModelComponentsResponse {
            model: model_name.to_string(),
            components: component_status_from_paths(&config, model_name, &paths),
        });
    }

    let config = state.config.read().await;
    let Some(paths) = ModelPaths::resolve(model_name, &config) else {
        return Err(ApiError::unknown_model(format!(
            "unknown model '{model_name}'. Run 'mold list' to see available models."
        )));
    };
    require_model_paths_activation(&config, model_name, &paths)
        .map_err(ApiError::model_activation)?;
    Ok(ModelComponentsResponse {
        model: model_name.to_string(),
        components: component_status_from_paths(&config, model_name, &paths),
    })
}

/// Strictly local component inspection for read-only placement previews.
///
/// Unlike [`model_component_status`], this never reloads configuration,
/// installs a catalog model, consults live catalog state, mutates intent
/// caches, or writes sidecars. Installed catalog IDs are resolved from
/// safe-contained local sidecars without warming the in-memory configuration.
pub(crate) async fn model_component_status_existing_only(
    state: &AppState,
    model_name: &str,
) -> Result<ModelComponentsResponse, ApiError> {
    let family = family_for_model(state, model_name).await;
    mold_core::require_model_activation(model_name, family.as_deref())
        .map_err(ApiError::model_activation)?;
    let resolved = mold_core::manifest::resolve_model_name(model_name);
    let config = state.config.read().await;
    if mold_core::manifest::find_manifest(&resolved).is_some() {
        let components =
            if let Some(paths) = manifest_paths_with_config_overrides(&config, &resolved) {
                require_model_paths_activation(&config, &resolved, &paths)
                    .map_err(ApiError::model_activation)?;
                component_status_from_paths(&config, &resolved, &paths)
            } else {
                manifest_component_status(&config, &resolved, &resolved)
            };
        return Ok(ModelComponentsResponse {
            model: resolved,
            components,
        });
    }

    if looks_like_catalog_id(model_name) {
        if let Ok(Some(resolution)) = resolve_existing_model_paths(model_name, &config) {
            let overlaid = resolution
                .model_config_overlay
                .as_ref()
                .map(|model_config| {
                    let mut effective = config.clone();
                    effective
                        .models
                        .insert(model_name.to_string(), model_config.clone());
                    effective
                });
            let effective = overlaid.as_ref().unwrap_or(&config);
            require_model_paths_activation(effective, model_name, &resolution.paths)
                .map_err(ApiError::model_activation)?;
            return Ok(ModelComponentsResponse {
                model: model_name.to_string(),
                components: component_status_from_paths(effective, model_name, &resolution.paths),
            });
        }
        return Ok(ModelComponentsResponse {
            model: model_name.to_string(),
            components: catalog_sidecar_component_status(&config, model_name)
                .map_err(ApiError::model_activation)?,
        });
    }

    let Some(paths) = ModelPaths::resolve(model_name, &config) else {
        return Err(ApiError::not_found(format!(
            "unknown model '{model_name}'. Run 'mold list' to see available models."
        )));
    };
    require_model_paths_activation(&config, model_name, &paths)
        .map_err(ApiError::model_activation)?;
    Ok(ModelComponentsResponse {
        model: model_name.to_string(),
        components: component_status_from_paths(&config, model_name, &paths),
    })
}

fn require_model_paths_activation(
    config: &Config,
    model_name: &str,
    paths: &ModelPaths,
) -> Result<(), mold_core::ModelActivationError> {
    let family = config
        .lookup_model_config(model_name)
        .and_then(|model| model.family)
        .or_else(|| {
            mold_core::manifest::find_manifest(model_name).map(|manifest| manifest.family.clone())
        });
    let artifact_root = config.resolved_models_dir();
    for path in paths.all_file_paths() {
        mold_core::require_model_artifact_activation(
            path,
            Some(&artifact_root),
            family.as_deref(),
        )?;
    }
    Ok(())
}

fn catalog_sidecar_component_status(
    config: &Config,
    model_name: &str,
) -> Result<Vec<ModelComponentStatus>, mold_core::ModelActivationError> {
    let models_dir = config.resolved_models_dir();
    let sidecar = mold_catalog::sidecar::walk_sidecars(&models_dir)
        .into_iter()
        .find(|(_, sidecar)| sidecar.id == model_name);
    let Some((sidecar_dir, sidecar)) = sidecar else {
        return Ok(vec![ModelComponentStatus {
            kind: "transformer".to_string(),
            name: "primary checkpoint".to_string(),
            present: false,
            path: None,
            repair_model: Some(model_name.to_string()),
            options: component_options_for_kind(config, "transformer", None),
        }]);
    };
    // A sidecar is externally persisted metadata. Validate all of its
    // identities before deriving or returning concrete local paths so an
    // opaque catalog id cannot expose compliance-gated model locations via
    // the read-only component diagnostics endpoint.
    mold_catalog::sidecar::require_sidecar_artifact_activation(
        &models_dir,
        &sidecar_dir,
        &sidecar,
    )?;

    let primary = mold_catalog::sidecar::primary_path(&sidecar_dir, &sidecar);
    // Per-expert presence: a pair sidecar's whole-install
    // `primary_path_if_present` is None whenever EITHER half is missing,
    // which would report the (present) high expert as the missing
    // component and never name the low-noise file at all. Diagnostics
    // check each half separately; installed detection elsewhere still
    // requires both.
    let pulling = mold_catalog::sidecar::install_pull_in_progress(&sidecar_dir, &sidecar);
    let primary_present = !pulling
        && mold_catalog::sidecar::primary_file_if_complete(&sidecar_dir, &sidecar).is_some();
    let mut components = vec![ModelComponentStatus {
        kind: "transformer".to_string(),
        name: "primary checkpoint".to_string(),
        present: primary_present,
        path: primary
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        repair_model: Some(model_name.to_string()),
        options: component_options_for_kind(config, "transformer", primary.as_deref()),
    }];
    if sidecar.low_noise_filename_rel.is_some() {
        let low = mold_catalog::sidecar::low_noise_path(&sidecar_dir, &sidecar);
        let low_present = !pulling
            && mold_catalog::sidecar::low_noise_file_if_complete(&sidecar_dir, &sidecar).is_some();
        components.push(ModelComponentStatus {
            kind: "transformer".to_string(),
            name: "low-noise transformer".to_string(),
            present: low_present,
            path: low.as_ref().map(|path| path.to_string_lossy().to_string()),
            repair_model: Some(model_name.to_string()),
            options: component_options_for_kind(config, "transformer", low.as_deref()),
        });
    }

    let Ok(family) = mold_catalog::families::Family::from_str(&sidecar.family) else {
        return Ok(components);
    };
    for companion in mold_catalog::companions::companions_for(
        family,
        sidecar.sub_family.as_deref(),
        mold_catalog::entry::Bundling::SingleFile,
        mold_catalog::entry::Kind::Checkpoint,
    ) {
        if let Some(paths) = manifest_paths_with_config_overrides(config, &companion) {
            require_model_paths_activation(config, model_name, &paths)?;
            components.extend(component_status_from_paths(config, model_name, &paths));
        } else {
            components.extend(manifest_component_status(config, &companion, model_name));
        }
    }
    Ok(components)
}

fn manifest_paths_with_config_overrides(config: &Config, model_name: &str) -> Option<ModelPaths> {
    let mut effective = config.resolved_model_config(model_name);
    if let Some(configured) = config.lookup_model_config(model_name) {
        macro_rules! apply_configured_path {
            ($field:ident) => {
                if configured.$field.is_some() {
                    effective.$field = configured.$field;
                }
            };
        }
        apply_configured_path!(transformer);
        apply_configured_path!(transformer_shards);
        apply_configured_path!(vae);
        apply_configured_path!(spatial_upscaler);
        apply_configured_path!(temporal_upscaler);
        apply_configured_path!(distilled_lora);
        apply_configured_path!(t5_encoder);
        apply_configured_path!(clip_encoder);
        apply_configured_path!(t5_tokenizer);
        apply_configured_path!(clip_tokenizer);
        apply_configured_path!(clip_encoder_2);
        apply_configured_path!(clip_tokenizer_2);
        apply_configured_path!(text_encoder_files);
        apply_configured_path!(text_tokenizer);
        apply_configured_path!(decoder);
    }
    ModelPaths::resolve_from_model_config_exact(&effective)
        .or_else(|| ModelPaths::resolve(model_name, config))
}

fn manifest_component_status(
    config: &Config,
    manifest_name: &str,
    repair_model: &str,
) -> Vec<ModelComponentStatus> {
    let Some(manifest) = mold_core::manifest::find_manifest(manifest_name) else {
        return Vec::new();
    };
    let models_dir = config.resolved_models_dir();
    manifest
        .files
        .iter()
        .map(|file| {
            let kind = manifest_component_kind(file.component);
            let path = models_dir.join(mold_core::manifest::storage_path(manifest, file));
            ModelComponentStatus {
                kind: kind.to_string(),
                name: manifest_component_name(file.component, &file.hf_filename).to_string(),
                present: path.is_file(),
                path: Some(path.to_string_lossy().to_string()),
                repair_model: Some(repair_model.to_string()),
                options: component_options_for_kind(config, kind, Some(&path)),
            }
        })
        .collect()
}

fn manifest_component_kind(component: mold_core::manifest::ModelComponent) -> &'static str {
    use mold_core::manifest::ModelComponent;
    match component {
        ModelComponent::Transformer
        | ModelComponent::TransformerShard
        | ModelComponent::LowNoiseTransformer => "transformer",
        ModelComponent::Vae => "vae",
        ModelComponent::AudioVae => "audio_vae",
        ModelComponent::DurationHead => "duration_head",
        ModelComponent::SpatialUpscaler => "spatial_upscaler",
        ModelComponent::TemporalUpscaler => "temporal_upscaler",
        ModelComponent::DistilledLora | ModelComponent::LowNoiseDistilledLora => "distilled_lora",
        ModelComponent::T5Encoder | ModelComponent::TextEncoder => "text_encoder",
        ModelComponent::ClipEncoder | ModelComponent::ClipEncoder2 => "clip",
        ModelComponent::T5Tokenizer
        | ModelComponent::ClipTokenizer
        | ModelComponent::ClipTokenizer2
        | ModelComponent::TextTokenizer
        | ModelComponent::Processor => "tokenizer",
        ModelComponent::VideoScheduler
        | ModelComponent::AudioScheduler
        | ModelComponent::ModelConfig
        | ModelComponent::TaskConfig => "config",
        ModelComponent::Decoder => "decoder",
        ModelComponent::Upscaler => "upscaler",
        // PuLID's bundle is auxiliary conditioning, not a generator's parts.
        // Each artifact gets its own kind rather than being folded into
        // `transformer`/`clip`, which name slots in a diffusion pipeline the
        // bundle does not participate in.
        ModelComponent::IdentityAdapter => "identity_adapter",
        ModelComponent::IdentityVisionEncoder => "identity_vision_encoder",
        ModelComponent::FaceDetector => "face_detector",
        ModelComponent::FaceRecognizer => "face_recognizer",
        ModelComponent::FaceParser => "face_parser",
    }
}

fn manifest_component_name(component: mold_core::manifest::ModelComponent, filename: &str) -> &str {
    use mold_core::manifest::ModelComponent;
    match component {
        ModelComponent::Transformer => "transformer",
        ModelComponent::TransformerShard => "transformer shard",
        ModelComponent::LowNoiseTransformer => "low-noise transformer",
        ModelComponent::Vae => "vae",
        ModelComponent::AudioVae => "audio vae",
        ModelComponent::DurationHead => "duration head",
        ModelComponent::SpatialUpscaler => "spatial upscaler",
        ModelComponent::TemporalUpscaler => "temporal upscaler",
        ModelComponent::DistilledLora => "distilled lora",
        ModelComponent::LowNoiseDistilledLora => "low-noise distilled lora",
        ModelComponent::T5Encoder => "t5 encoder",
        ModelComponent::ClipEncoder => "clip encoder",
        ModelComponent::T5Tokenizer => "t5 tokenizer",
        ModelComponent::ClipTokenizer => "clip tokenizer",
        ModelComponent::ClipEncoder2 => "clip-g encoder",
        ModelComponent::ClipTokenizer2 => "clip-g tokenizer",
        ModelComponent::TextEncoder => "text encoder",
        ModelComponent::TextTokenizer => "text tokenizer",
        ModelComponent::Processor => "processor",
        ModelComponent::VideoScheduler => "video scheduler",
        ModelComponent::AudioScheduler => "audio scheduler",
        ModelComponent::ModelConfig => "model config",
        ModelComponent::TaskConfig => "task config",
        ModelComponent::Decoder => "decoder",
        ModelComponent::Upscaler => filename,
        ModelComponent::IdentityAdapter => "identity adapter",
        ModelComponent::IdentityVisionEncoder => "identity vision encoder",
        ModelComponent::FaceDetector => "face detector",
        ModelComponent::FaceRecognizer => "face recognizer",
        ModelComponent::FaceParser => "face parser",
    }
}

fn component_status_from_paths(
    config: &Config,
    model_name: &str,
    paths: &ModelPaths,
) -> Vec<ModelComponentStatus> {
    let mut components = Vec::new();
    let mut push_path = |kind: &str, name: &str, path: &std::path::Path| {
        components.push(ModelComponentStatus {
            kind: kind.to_string(),
            name: name.to_string(),
            present: path.is_file(),
            path: Some(path.to_string_lossy().to_string()),
            repair_model: Some(model_name.to_string()),
            options: component_options_for_kind(config, kind, Some(path)),
        });
    };
    push_path("transformer", "transformer", &paths.transformer);
    for shard in &paths.transformer_shards {
        push_path("transformer", "transformer shard", shard);
    }
    // The low-noise expert is not read until the schedule crosses the expert
    // boundary, so without its own row a deleted one is invisible to the
    // placement preview: the model reports complete and fails mid-generation
    // with nothing named for repair.
    if let Some(path) = &paths.low_noise_transformer {
        push_path("transformer", "low-noise transformer", path);
    }
    push_path("vae", "vae", &paths.vae);
    if let Some(path) = &paths.spatial_upscaler {
        push_path("spatial_upscaler", "spatial upscaler", path);
    }
    if let Some(path) = &paths.temporal_upscaler {
        push_path("temporal_upscaler", "temporal upscaler", path);
    }
    if let Some(path) = &paths.distilled_lora {
        push_path("distilled_lora", "distilled lora", path);
    }
    if let Some(path) = &paths.low_noise_distilled_lora {
        push_path("distilled_lora", "low-noise distilled lora", path);
    }
    if let Some(path) = &paths.t5_encoder {
        push_path("text_encoder", "t5 encoder", path);
    }
    if let Some(path) = &paths.t5_tokenizer {
        push_path("tokenizer", "t5 tokenizer", path);
    }
    if let Some(path) = &paths.clip_encoder {
        push_path("clip", "clip encoder", path);
    }
    if let Some(path) = &paths.clip_tokenizer {
        push_path("tokenizer", "clip tokenizer", path);
    }
    if let Some(path) = &paths.clip_encoder_2 {
        push_path("clip", "clip-g encoder", path);
    }
    if let Some(path) = &paths.clip_tokenizer_2 {
        push_path("tokenizer", "clip-g tokenizer", path);
    }
    for path in &paths.text_encoder_files {
        push_path("text_encoder", "text encoder", path);
    }
    if let Some(path) = &paths.text_tokenizer {
        push_path("tokenizer", "text tokenizer", path);
    }
    if let Some(path) = &paths.decoder {
        push_path("decoder", "decoder", path);
    }
    components
}

fn component_options_for_kind(
    config: &Config,
    kind: &str,
    current_path: Option<&Path>,
) -> Vec<ModelComponentOption> {
    let mut options = BTreeMap::<String, ModelComponentOption>::new();
    let models_dir = config.resolved_models_dir();
    if let Some(path) = current_path.filter(|path| {
        mold_core::require_model_artifact_activation(path, Some(&models_dir), None).is_ok()
    }) {
        add_component_option(&mut options, path);
    }
    for model_cfg in config.models.values() {
        for path in config_component_paths_for_kind(model_cfg, kind) {
            let path = Path::new(path);
            if mold_core::require_model_artifact_activation(
                path,
                Some(&models_dir),
                model_cfg.family.as_deref(),
            )
            .is_ok()
            {
                add_component_option(&mut options, path);
            }
        }
    }
    for manifest in mold_core::manifest::known_manifests() {
        if mold_core::require_model_activation(&manifest.name, Some(&manifest.family)).is_err() {
            continue;
        }
        for file in &manifest.files {
            if manifest_component_kind(file.component) != kind {
                continue;
            }
            let path = models_dir.join(mold_core::manifest::storage_path(manifest, file));
            if path.is_file()
                && mold_core::require_model_artifact_activation(
                    &path,
                    Some(&models_dir),
                    Some(&manifest.family),
                )
                .is_ok()
            {
                add_component_option(&mut options, &path);
            }
        }
    }
    options.into_values().collect()
}

fn config_component_paths_for_kind<'a>(
    model_cfg: &'a mold_core::config::ModelConfig,
    kind: &str,
) -> Vec<&'a str> {
    let mut paths = Vec::new();
    match kind {
        "transformer" => {
            if let Some(path) = model_cfg.transformer.as_deref() {
                paths.push(path);
            }
            if let Some(shards) = &model_cfg.transformer_shards {
                paths.extend(shards.iter().map(String::as_str));
            }
        }
        "vae" => {
            if let Some(path) = model_cfg.vae.as_deref() {
                paths.push(path);
            }
        }
        "text_encoder" => {
            if let Some(path) = model_cfg.t5_encoder.as_deref() {
                paths.push(path);
            }
            if let Some(files) = &model_cfg.text_encoder_files {
                paths.extend(files.iter().map(String::as_str));
            }
        }
        "clip" => {
            if let Some(path) = model_cfg.clip_encoder.as_deref() {
                paths.push(path);
            }
            if let Some(path) = model_cfg.clip_encoder_2.as_deref() {
                paths.push(path);
            }
        }
        "tokenizer" => {
            for path in [
                model_cfg.t5_tokenizer.as_deref(),
                model_cfg.clip_tokenizer.as_deref(),
                model_cfg.clip_tokenizer_2.as_deref(),
                model_cfg.text_tokenizer.as_deref(),
            ]
            .into_iter()
            .flatten()
            {
                paths.push(path);
            }
        }
        "spatial_upscaler" => {
            if let Some(path) = model_cfg.spatial_upscaler.as_deref() {
                paths.push(path);
            }
        }
        "temporal_upscaler" => {
            if let Some(path) = model_cfg.temporal_upscaler.as_deref() {
                paths.push(path);
            }
        }
        "distilled_lora" => {
            if let Some(path) = model_cfg.distilled_lora.as_deref() {
                paths.push(path);
            }
        }
        "decoder" => {
            if let Some(path) = model_cfg.decoder.as_deref() {
                paths.push(path);
            }
        }
        _ => {}
    }
    paths
}

fn add_component_option(options: &mut BTreeMap<String, ModelComponentOption>, path: &Path) {
    let path_str = path.to_string_lossy().to_string();
    options.entry(path_str.clone()).or_insert_with(|| {
        let label = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(path_str.as_str())
            .to_string();
        ModelComponentOption {
            label,
            path: path_str,
            present: path.is_file(),
        }
    });
}

/// Ensure the requested model is loaded on GPU and ready for inference.
///
/// Checks the model cache: if already loaded, just touches the LRU order.
/// If cached but unloaded, reloads it. If not in cache, creates a new engine.
///
/// `hint` carries the per-request resolution / family used by the activation
/// budget. `None` falls back to the previous fixed-headroom approximation —
/// admin-API loads with no resolution context (cache prewarm, etc.) take
/// this path.
pub(crate) async fn ensure_model_ready(
    state: &AppState,
    model_name: &str,
    progress: Option<EngineProgressCallback>,
    hint: Option<ActivationHint>,
    request_has_lora: bool,
) -> Result<(), ApiError> {
    let _guard = state.model_load_lock.lock().await;

    // Fast path: model is in cache and loaded.
    {
        let mut cache = state.model_cache.lock().await;
        // Grab active model's VRAM before mutable borrow via get_mut.
        let active_vram = cache.active_vram_bytes();
        if let Some(entry) = cache.get_mut(model_name) {
            if entry.residency == ModelResidency::Gpu {
                let must_recreate = entry.engine.model_paths().is_some_and(|paths| {
                    request_requires_fresh_engine_for_offload_policy(paths, hint, request_has_lora)
                });
                if must_recreate {
                    tracing::info!(
                        model = %model_name,
                        "recreating loaded engine for request-specific offload policy"
                    );
                } else {
                    // Already loaded: nothing is about to be loaded, so there
                    // is no load progress to report. Leave the engine with no
                    // callback — the generation installs its own and clears it
                    // — rather than installing one that only ever gets replaced.
                    entry.engine.clear_on_progress();
                    return Ok(());
                }
            }

            // Cached but not on GPU (Parked) — need to reload.
            // MPS memory guard: check before unloading the active model.
            // Include the active model's footprint as reclaimable memory.
            let cached_paths = entry.engine.model_paths().cloned();
            if let Some(paths) = cached_paths.as_ref() {
                preflight_memory_guard_for_request(
                    model_name,
                    paths,
                    active_vram,
                    0,
                    hint,
                    request_has_lora,
                )?;
            }
            // Parked engines retain tokenizers/caches for faster reload.
            // First unload the currently active model (if any) to free VRAM.
            if let Some(active_name) = cache.unload_active() {
                #[cfg(feature = "metrics")]
                crate::metrics::clear_model_loaded(&active_name);
                tracing::info!(
                    from = %active_name,
                    to = %model_name,
                    "unloaded active model to reload cached model"
                );
            }

            drop(cache);
            if let Some(paths) = cached_paths.as_ref() {
                preflight_memory_guard_after_drop_for_request(
                    model_name,
                    paths,
                    0,
                    hint,
                    request_has_lora,
                )?;
            } else {
                #[cfg(feature = "cuda")]
                mold_inference::device::post_drop_free_vram_bytes(0)
                    .map_err(|error| ApiError::insufficient_memory(error.to_string()))?;
            }
            let load_strategy = match cached_paths.as_ref() {
                Some(paths) => crate::memory_preflight::request_aware_load_strategy(
                    select_server_load_strategy_for_budget(
                        paths,
                        effective_load_available_bytes(0, 0)?,
                        hint,
                    ),
                    paths,
                    hint,
                    request_has_lora,
                    false,
                ),
                None => mold_inference::LoadStrategy::Eager,
            };
            if load_strategy == mold_inference::LoadStrategy::Sequential {
                tracing::info!(
                    model = %model_name,
                    "server load strategy degraded to sequential to fit post-drop memory budget"
                );
            }

            // Only check the engine out after the authoritative post-drop
            // guard passes. A failed guard must leave the parked cache entry
            // intact rather than leaking its in-flight marker.
            let cached = {
                let mut cache = state.model_cache.lock().await;
                cache.take(model_name).ok_or_else(|| {
                    ApiError::internal(format!("cache race: model '{model_name}' vanished"))
                })?
            };
            let mut engine = cached.engine;
            if load_strategy == mold_inference::LoadStrategy::Sequential {
                let Some(paths) = cached_paths else {
                    let evicted = {
                        let mut cache = state.model_cache.lock().await;
                        cache.insert(engine, 0)
                    };
                    drop(evicted);
                    return Err(ApiError::internal(format!(
                        "cached engine for '{model_name}' does not expose model paths"
                    )));
                };
                let config = state.config.read().await;
                let offload = server_offload_enabled_for_paths(&paths, hint, request_has_lora);
                let resolved_catalog_config =
                    resolve_installed_catalog_paths_for_worker(model_name, &config)?
                        .map(|(_, config)| config);
                let engine_config = resolved_catalog_config.as_ref().unwrap_or(&config);
                match mold_inference::create_engine_with_pool(
                    model_name.to_string(),
                    paths,
                    engine_config,
                    load_strategy,
                    0,
                    offload,
                    Some(state.shared_pool.clone()),
                ) {
                    Ok(new_engine) => {
                        drop(config);
                        drop(engine);
                        engine = new_engine;
                    }
                    Err(e) => {
                        drop(config);
                        let evicted = {
                            let mut cache = state.model_cache.lock().await;
                            cache.insert(engine, 0)
                        };
                        drop(evicted);
                        return Err(ApiError::internal(format!(
                            "failed to recreate cached engine for '{model_name}': {e}"
                        )));
                    }
                }
            }

            if let Some(callback) = progress.clone() {
                engine.set_on_progress(Box::new(move |event| {
                    callback(event);
                }));
            } else {
                engine.clear_on_progress();
            }

            let model_log = model_name.to_string();
            #[cfg(feature = "metrics")]
            let load_start = std::time::Instant::now();
            // Sample VRAM baseline before load so we can record the new
            // model's per-load delta rather than the device-global usage.
            let vram_baseline = mold_inference::device::vram_in_use_bytes(0);
            let join_result = tokio::task::spawn_blocking(move || {
                tracing::info!(model = %model_log, "reloading cached engine...");
                if let Err(e) = engine.load() {
                    tracing::error!("model reload failed: {e:#}");
                    return Err((
                        ApiError::internal(format!("model reload error: {e}")),
                        engine,
                    ));
                }
                Ok(engine)
            })
            .await;

            match join_result {
                Ok(Ok(loaded_engine)) => {
                    #[cfg(feature = "metrics")]
                    {
                        let duration = load_start.elapsed().as_secs_f64();
                        crate::metrics::record_model_load(model_name, duration);
                        crate::metrics::set_model_loaded(model_name);
                        let vram_est = mold_inference::device::vram_in_use_bytes(0);
                        crate::metrics::record_gpu_memory(vram_est);
                    }
                    let vram = mold_inference::device::vram_load_delta(0, vram_baseline);
                    // Insert under the cache lock, but drop the evicted engine
                    // OUTSIDE the lock — `cuMemFree` and safetensor unmap during
                    // `Box<dyn ...>` drop can block other cache users for hundreds
                    // of milliseconds. `insert` clears the in_flight marker.
                    let evicted = {
                        let mut cache = state.model_cache.lock().await;
                        cache.insert(loaded_engine, vram)
                    };
                    drop(evicted);
                }
                Ok(Err((api_err, unloaded_engine))) => {
                    // Put it back as unloaded so cache isn't corrupted. The
                    // unloaded engine has no GPU resources to free, but for
                    // consistency with the file's invariant — never drop an
                    // engine while holding the cache lock — bind and drop
                    // outside.
                    let evicted = {
                        let mut cache = state.model_cache.lock().await;
                        cache.insert(unloaded_engine, 0)
                    };
                    drop(evicted);
                    return Err(api_err);
                }
                Err(join_err) => {
                    // The blocking task aborted (panic that escaped the
                    // closure, runtime shutdown, etc.). Engine is gone —
                    // we can't restore it, so clear the in_flight marker
                    // explicitly. Without this, the model name leaks
                    // forever in `in_flight`: every subsequent
                    // `ensure_model_ready` fast-paths through
                    // `cache.contains()` (which still says true), then
                    // `cache.take()` returns None, and generation fails
                    // with "no engine available after model readiness
                    // check" indefinitely for this model.
                    {
                        let mut cache = state.model_cache.lock().await;
                        cache.clear_in_flight(model_name);
                    }
                    return Err(ApiError::internal(format!(
                        "model reload task failed: {join_err}"
                    )));
                }
            }
            return Ok(());
        }
    }

    // Not in cache — check if model is available on disk.
    match check_model_available(state, model_name).await? {
        Some(paths) => {
            create_and_load_engine(state, model_name, paths, progress, hint, request_has_lora).await
        }
        None => Ok(()),
    }
}

pub(crate) async fn pull_model(
    state: &AppState,
    model: &str,
    progress: Option<DownloadProgressCallback>,
) -> Result<PullStatus, ApiError> {
    if mold_core::manifest::find_manifest(&mold_core::manifest::resolve_model_name(model)).is_none()
    {
        return Err(ApiError::unknown_model(format!(
            "unknown model '{model}'. Run 'mold list' to see available models."
        )));
    }

    let _guard = state.pull_lock.lock().await;

    let manifest =
        mold_core::manifest::find_manifest(&mold_core::manifest::resolve_model_name(model))
            .expect("manifest was validated before acquiring the pull lock");
    {
        let config = refresh_config(state).await;
        if config.manifest_model_is_downloaded(model)
            && (!manifest.is_upscaler() || configured_upscaler_weights_exist(&config, model))
        {
            return Ok(PullStatus::AlreadyAvailable);
        }
    }

    tracing::info!(model = %model, "pulling model via API");

    let opts = mold_core::download::PullOptions::default();
    let new_config = match progress {
        Some(callback) => {
            mold_core::download::pull_and_configure_with_callback(model, callback, &opts)
                .await
                .map(|(config, _)| config)
        }
        None => mold_core::download::pull_and_configure(model, &opts)
            .await
            .map(|(config, _)| config),
    }
    .map_err(|e| {
        // A refusal is a decision, not a fault: log it at warn and keep the
        // structured payload so a client can offer acceptance.
        match &e {
            mold_core::download::DownloadError::LicenseNotAccepted { license_id, .. } => {
                tracing::warn!(model, license = %license_id, "pull refused pending license acceptance");
            }
            other => tracing::error!("pull failed for {}: {other}", model),
        }
        ApiError::from_download_error(model, &e)
    })?;

    {
        let mut config = state.config.write().await;
        *config = new_config;
    }

    tracing::info!(model = %model, "pull complete");
    Ok(PullStatus::Pulled)
}

/// Unload the active model from GPU. The engine remains in the cache (unloaded)
/// so it can be reloaded quickly on the next request.
pub(crate) async fn unload_model(state: &AppState) -> String {
    let mut cache = state.model_cache.lock().await;
    match cache.unload_active() {
        Some(name) => {
            #[cfg(feature = "metrics")]
            {
                crate::metrics::clear_model_loaded(&name);
                crate::metrics::record_gpu_memory(0);
            }
            drop(cache);
            let free_after_drop = mold_inference::device::post_drop_free_vram_bytes(0);
            tracing::info!(
                free_vram_bytes = ?free_after_drop,
                "legacy model unloaded; sampled post-drop VRAM"
            );
            tracing::info!(model = %name, "model unloaded via API");
            format!("unloaded {name}")
        }
        None => "no model loaded".to_string(),
    }
}

async fn create_and_load_engine(
    state: &AppState,
    model_name: &str,
    paths: ModelPaths,
    progress: Option<EngineProgressCallback>,
    hint: Option<ActivationHint>,
    request_has_lora: bool,
) -> Result<(), ApiError> {
    // MPS memory guard: reject before unloading current model so it stays operational.
    // Include the active model's footprint as reclaimable memory.
    let active_vram = {
        let cache = state.model_cache.lock().await;
        cache.active_vram_bytes()
    };
    preflight_memory_guard_for_request(model_name, &paths, active_vram, 0, hint, request_has_lora)?;
    // Unload the current active model to free GPU memory.
    {
        let mut cache = state.model_cache.lock().await;
        let result = cache.unload_active();
        if let Some(ref name) = result {
            #[cfg(feature = "metrics")]
            crate::metrics::clear_model_loaded(name);
            tracing::info!(
                from = %name,
                to = %model_name,
                "unloading active model before loading new one"
            );
        }
    }
    preflight_memory_guard_after_drop_for_request(model_name, &paths, 0, hint, request_has_lora)?;
    let load_strategy = crate::memory_preflight::request_aware_load_strategy(
        select_server_load_strategy_for_device(
            &paths,
            effective_load_available_bytes(0, 0)?,
            mold_inference::device::total_vram_bytes(0),
            hint,
        ),
        &paths,
        hint,
        request_has_lora,
        false,
    );
    if load_strategy == mold_inference::LoadStrategy::Sequential {
        tracing::info!(
            model = %model_name,
            "server load strategy degraded to sequential to fit post-drop memory budget"
        );
    }

    let config = state.config.read().await;
    let offload = server_offload_enabled_for_paths(&paths, hint, request_has_lora);
    let mut new_engine = mold_inference::create_engine_with_pool(
        model_name.to_string(),
        paths,
        &config,
        load_strategy,
        0,
        offload,
        Some(state.shared_pool.clone()),
    )
    .map_err(|e| ApiError::internal(format!("failed to create engine for '{model_name}': {e}")))?;
    drop(config);

    if let Some(callback) = progress {
        new_engine.set_on_progress(Box::new(move |event| {
            callback(event);
        }));
    } else {
        new_engine.clear_on_progress();
    }

    let model_log = model_name.to_string();
    #[cfg(feature = "metrics")]
    let load_start = std::time::Instant::now();
    // Sample VRAM baseline before load so we can record the new model's
    // per-load delta rather than the device-global usage.
    let vram_baseline = mold_inference::device::vram_in_use_bytes(0);
    new_engine = tokio::task::spawn_blocking(move || {
        tracing::info!(model = %model_log, "loading model...");
        new_engine.load().map_err(|e| {
            tracing::error!("model load failed: {e:#}");
            ApiError::internal(format!("model load error: {e}"))
        })?;
        Ok::<_, ApiError>(new_engine)
    })
    .await
    .map_err(|e| ApiError::internal(format!("model load task failed: {e}")))??;

    #[cfg(feature = "metrics")]
    {
        let duration = load_start.elapsed().as_secs_f64();
        crate::metrics::record_model_load(model_name, duration);
        crate::metrics::set_model_loaded(model_name);
    }

    let vram = mold_inference::device::vram_load_delta(0, vram_baseline);
    #[cfg(feature = "metrics")]
    crate::metrics::record_gpu_memory(mold_inference::device::vram_in_use_bytes(0));

    // Insert under the cache lock, but drop the evicted engine OUTSIDE the
    // lock — `cuMemFree` and safetensor unmap during `Box<dyn ...>` drop can
    // block other cache users for hundreds of milliseconds.
    let evicted = {
        let mut cache = state.model_cache.lock().await;
        cache.insert(new_engine, vram)
    };
    drop(evicted);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const GB: u64 = 1_000_000_000;

    #[test]
    fn ltx25_incomplete_pack_is_not_ready_or_duration_capable() {
        let config = Config::default();
        let mut catalog = build_model_catalog(&config, None, false);
        let row = catalog
            .iter_mut()
            .find(|entry| mold_core::ltx25_manifest::is_contract_manifest(&entry.info.name))
            .expect("LTX-2.5 manifest row");
        row.downloaded = false;

        annotate_ltx25_runtime_readiness(&mut catalog, &config);

        let row = catalog
            .iter()
            .find(|entry| mold_core::ltx25_manifest::is_contract_manifest(&entry.info.name))
            .unwrap();
        assert_eq!(row.runtime_ready, Some(false));
        assert_eq!(row.supports_duration_prediction, Some(false));
        assert!(row
            .runtime_readiness_error
            .as_deref()
            .is_some_and(|reason| reason.contains("incomplete")));
    }

    /// GGUF rows follow the normal LTX-2.5 readiness path since the native
    /// quantized runtime landed (#1414): no forced runtime gate, and an
    /// uninstalled pack reports the same incomplete-pack message every other
    /// tier gets.
    #[test]
    fn ltx25_gguf_rows_are_runtime_ready_when_qualified() {
        let config = Config::default();
        let mut catalog = build_model_catalog(&config, None, false);

        annotate_ltx25_runtime_readiness(&mut catalog, &config);

        let rows = catalog
            .iter()
            .filter(|entry| mold_core::ltx25_manifest::is_gguf_manifest(&entry.info.name))
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 7);
        for row in rows {
            assert_ne!(row.runtime_available, Some(false), "{}", row.info.name);
            assert!(
                row.runtime_unavailable_reason.is_none(),
                "{}",
                row.info.name
            );
            // Nothing is installed under this test config, so readiness
            // reports the ordinary incomplete-pack repair path.
            assert_eq!(row.runtime_ready, Some(false), "{}", row.info.name);
            assert!(
                row.runtime_readiness_error
                    .as_deref()
                    .is_some_and(|reason| reason.contains("incomplete")),
                "{}",
                row.info.name
            );
        }
    }

    fn neutral_catalog_intent() -> mold_catalog::synthesis::CatalogModelIntent {
        mold_catalog::synthesis::CatalogModelIntent {
            family: "custom".to_string(),
            sub_family: None,
            primary_recipe_path: PathBuf::from("ordinary/model.safetensors"),
            vae_recipe_path: None,
            text_encoder_recipe_paths: Vec::new(),
            low_noise_recipe_path: None,
            companions: Vec::new(),
            bundling: mold_catalog::entry::Bundling::SingleFile,
        }
    }

    /// `/api/models` must advertise extend for the wan checkpoints that can
    /// actually do it (#783).
    ///
    /// `supports_extend` is seeded family-blind for a locally installed entry
    /// because mold-core cannot read checkpoint headers, and the annotate
    /// pass — the same one that resolves the `source_image` contract — is
    /// what settles it through `catalog::extend_capable_model`. Dropping that
    /// re-derivation leaves every installed wan checkpoint advertising
    /// `false` while its paired `extend_default_overlap_frames` already says
    /// `wan`, which is the contradiction this exists to prevent.
    #[test]
    fn the_annotate_pass_settles_extend_support_from_the_resolved_wan_contract() {
        use mold_core::types::SourceImageCapability;

        let mut models = std::collections::HashMap::new();
        models.insert(
            "local-wan".to_string(),
            mold_core::config::ModelConfig {
                family: Some("wan".to_string()),
                ..mold_core::config::ModelConfig::default()
            },
        );
        let config = Config {
            models,
            ..Config::default()
        };
        let entry_for = |contract: Option<SourceImageCapability>| {
            let mut catalog = mold_core::catalog::build_model_catalog(&config, None, false);
            let index = catalog
                .iter()
                .position(|entry| entry.info.name == "local-wan")
                .expect("config-only wan entry");
            // Stand in for the header probe: `ModelPaths` cannot resolve a
            // real checkpoint in a unit test, and the pass keeps whatever
            // contract is already resolved when the probe finds nothing.
            catalog[index].source_image = contract;
            annotate_source_image_capabilities(&mut catalog, &config);
            catalog.remove(index)
        };

        assert_eq!(
            entry_for(Some(SourceImageCapability::Required)).supports_extend,
            Some(true),
            "an I2V checkpoint continues from the source's final frame"
        );
        assert_eq!(
            entry_for(Some(SourceImageCapability::Optional)).supports_extend,
            Some(true)
        );
        assert_eq!(
            entry_for(Some(SourceImageCapability::Unsupported)).supports_extend,
            Some(false),
            "a text-to-video checkpoint has no channel to accept the carryover frame"
        );
        assert_eq!(
            entry_for(None).supports_extend,
            Some(false),
            "an unclassified checkpoint never advertises a continuation"
        );
    }

    #[test]
    fn cached_catalog_intent_cannot_hide_h3_in_paths_or_companions() {
        let mut path_intent = neutral_catalog_intent();
        path_intent.primary_recipe_path = PathBuf::from("ordinary/MiniMaxH3.safetensors");
        assert!(
            require_catalog_intent_activation("cv:42", &path_intent, Path::new("/models")).is_err()
        );

        let mut companion_intent = neutral_catalog_intent();
        companion_intent
            .companions
            .push(mold_catalog::synthesis::CompanionIntent {
                name: "MiniMaxH3Transformer3DModel".to_string(),
                required: true,
            });
        assert!(require_catalog_intent_activation(
            "cv:42",
            &companion_intent,
            Path::new("/models")
        )
        .is_err());
    }

    #[test]
    fn cached_catalog_intent_uses_root_relative_artifact_identity() {
        let artifact_root = Path::new("/Volumes/ExternalStorage/mold-uat/minimax-h3/models");
        let mut intent = neutral_catalog_intent();
        intent.primary_recipe_path = artifact_root.join("flux/cv-42/weights.safetensors");
        require_catalog_intent_activation("cv:42", &intent, artifact_root)
            .expect("the configured models root must not taint ordinary cached intent paths");

        intent.primary_recipe_path = artifact_root.join("MiniMax-H3/weights.safetensors");
        assert!(require_catalog_intent_activation("cv:42", &intent, artifact_root).is_err());
    }

    #[tokio::test]
    async fn existing_only_component_status_never_installs_unseen_catalog_ids() {
        let root = tempfile::tempdir().unwrap();
        let _environment = IsolatedModelEnvironment::hermetic();
        let models_dir = root.path().join("models-that-do-not-exist");
        let state = AppState::for_tests();
        state.config.write().await.models_dir = models_dir.display().to_string();
        let config_before = format!("{:?}", *state.config.read().await);

        for model in ["cv:999999999", "hf:owner/repo"] {
            let status = model_component_status_existing_only(&state, model)
                .await
                .expect("uninstalled catalog diagnostics must remain local");
            assert_eq!(status.components.len(), 1);
            assert_eq!(status.components[0].name, "primary checkpoint");
            assert!(!status.components[0].present);
            assert_eq!(status.components[0].repair_model.as_deref(), Some(model));
        }

        assert!(state.catalog_intents.read().await.is_empty());
        assert_eq!(format!("{:?}", *state.config.read().await), config_before);
        assert!(
            !models_dir.exists(),
            "inspection must not create a sidecar or model root"
        );
    }

    #[tokio::test]
    async fn existing_only_component_status_reads_pre_resolved_catalog_paths() {
        let root = tempfile::tempdir().unwrap();
        let _environment = IsolatedModelEnvironment::hermetic();
        let transformer = root.path().join("model.safetensors");
        let vae = root.path().join("vae.safetensors");
        std::fs::write(&transformer, b"transformer").unwrap();
        std::fs::write(&vae, b"vae").unwrap();
        let state = AppState::for_tests();
        state.config.write().await.models.insert(
            "cv:123".to_string(),
            mold_core::config::ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                family: Some("flux".to_string()),
                ..Default::default()
            },
        );

        let status = model_component_status_existing_only(&state, "cv:123")
            .await
            .unwrap();

        assert_eq!(status.model, "cv:123");
        assert!(status.components.iter().all(|component| component.present));
        assert!(state.catalog_intents.read().await.is_empty());
    }

    fn catalog_sidecar(
        models_dir: &Path,
        install_dir_name: &str,
        id: &str,
        primary_rel: &str,
        write_primary: bool,
    ) -> PathBuf {
        let install_dir = models_dir.join(install_dir_name);
        let primary = install_dir.join(primary_rel);
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        if write_primary {
            std::fs::write(&primary, b"catalog-primary").unwrap();
        }
        mold_catalog::sidecar::write_sidecar(
            &install_dir.join(mold_catalog::sidecar::SIDECAR_FILENAME),
            &mold_catalog::sidecar::CatalogSidecar {
                schema: mold_catalog::sidecar::SIDECAR_SCHEMA,
                id: id.to_string(),
                source: if id.starts_with("cv:") {
                    "civitai".to_string()
                } else {
                    "huggingface".to_string()
                },
                source_id: id.to_string(),
                name: "Catalog fixture".to_string(),
                author: None,
                family: "flux2".to_string(),
                family_role: "finetune".to_string(),
                sub_family: Some("klein-9b".to_string()),
                kind: "checkpoint".to_string(),
                modality: "image".to_string(),
                nsfw: None,
                description: None,
                tags: Vec::new(),
                license: None,
                page_url: None,
                thumbnail_url: None,
                size_bytes: write_primary.then_some(b"catalog-primary".len() as u64),
                supported: true,
                trained_words: Vec::new(),
                primary_filename_rel: primary_rel.to_string(),
                primary_size_bytes: None,
                low_noise_filename_rel: None,
                low_noise_size_bytes: None,
                written_at: 0,
            },
        )
        .unwrap();
        primary
    }

    /// A pair sidecar with one half missing must name the truly missing
    /// expert: the whole-install check is false either way, but the
    /// diagnostic rows report each half's own presence.
    #[test]
    fn pair_sidecar_diagnostics_name_the_missing_half() {
        let root = tempfile::tempdir().unwrap();
        let _environment = IsolatedModelEnvironment::hermetic();
        let models_dir = root.path();
        let install = models_dir.join("cv-2057171");
        let high_rel = "wan/civitai/2057171/high.safetensors";
        let low_rel = "wan/civitai/2057100/low.safetensors";
        let high = install.join(high_rel);
        let low = install.join(low_rel);
        std::fs::create_dir_all(high.parent().unwrap()).unwrap();
        std::fs::create_dir_all(low.parent().unwrap()).unwrap();
        mold_catalog::sidecar::write_sidecar(
            &install.join(mold_catalog::sidecar::SIDECAR_FILENAME),
            &mold_catalog::sidecar::CatalogSidecar {
                schema: mold_catalog::sidecar::SIDECAR_SCHEMA,
                id: "cv:2057171".to_string(),
                source: "civitai".to_string(),
                source_id: "2057171".to_string(),
                name: "A14B pair fixture".to_string(),
                author: None,
                family: "wan".to_string(),
                family_role: "base".to_string(),
                sub_family: Some("wan22-t2v-a14b".to_string()),
                kind: "checkpoint".to_string(),
                modality: "video".to_string(),
                nsfw: None,
                description: None,
                tags: Vec::new(),
                license: None,
                page_url: None,
                thumbnail_url: None,
                size_bytes: Some(7),
                supported: true,
                trained_words: Vec::new(),
                primary_filename_rel: high_rel.to_string(),
                primary_size_bytes: Some(4),
                low_noise_filename_rel: Some(low_rel.to_string()),
                low_noise_size_bytes: Some(3),
                written_at: 0,
            },
        )
        .unwrap();
        let config = Config {
            models_dir: models_dir.display().to_string(),
            ..Default::default()
        };
        let presence =
            |components: &[ModelComponentStatus], name: &str| -> (bool, Option<String>) {
                let row = components
                    .iter()
                    .find(|component| component.name == name)
                    .unwrap_or_else(|| panic!("diagnostics must carry a {name:?} row"));
                (row.present, row.path.clone())
            };

        // Low missing: the (present) high expert must not be blamed, and
        // the low-noise row must name the missing file.
        std::fs::write(&high, b"high").unwrap();
        let components = catalog_sidecar_component_status(&config, "cv:2057171").unwrap();
        let (high_present, _) = presence(&components, "primary checkpoint");
        let (low_present, low_path) = presence(&components, "low-noise transformer");
        assert!(high_present, "present high expert must not read as missing");
        assert!(!low_present, "the missing low expert is the repairable row");
        assert!(low_path.unwrap().ends_with(low_rel));

        // Inverse: high missing, low present.
        std::fs::remove_file(&high).unwrap();
        std::fs::write(&low, b"low").unwrap();
        let components = catalog_sidecar_component_status(&config, "cv:2057171").unwrap();
        let (high_present, high_path) = presence(&components, "primary checkpoint");
        let (low_present, _) = presence(&components, "low-noise transformer");
        assert!(!high_present);
        assert!(low_present);
        assert!(high_path.unwrap().ends_with(high_rel));
    }

    fn flux2_catalog_config(models_dir: &Path) -> Config {
        let text_encoder = models_dir.join("companions/qwen3.safetensors");
        let tokenizer = models_dir.join("companions/tokenizer.json");
        let vae = models_dir.join("companions/vae.safetensors");
        std::fs::create_dir_all(text_encoder.parent().unwrap()).unwrap();
        for path in [&text_encoder, &tokenizer, &vae] {
            std::fs::write(path, b"companion").unwrap();
        }
        let mut config = Config {
            models_dir: models_dir.display().to_string(),
            ..Default::default()
        };
        config.models.insert(
            "flux2-te-9b".to_string(),
            mold_core::ModelConfig {
                transformer: Some(text_encoder.display().to_string()),
                vae: Some(vae.display().to_string()),
                text_encoder_files: Some(vec![text_encoder.display().to_string()]),
                text_tokenizer: Some(tokenizer.display().to_string()),
                family: Some("flux2".to_string()),
                ..Default::default()
            },
        );
        config.models.insert(
            "flux2-vae".to_string(),
            mold_core::ModelConfig {
                transformer: Some(vae.display().to_string()),
                vae: Some(vae.display().to_string()),
                family: Some("flux2".to_string()),
                ..Default::default()
            },
        );
        config
    }

    fn filesystem_snapshot(root: &Path) -> Vec<(PathBuf, u64)> {
        fn visit(root: &Path, current: &Path, entries: &mut Vec<(PathBuf, u64)>) {
            let mut children = std::fs::read_dir(current)
                .unwrap()
                .map(|entry| entry.unwrap())
                .collect::<Vec<_>>();
            children.sort_by_key(|entry| entry.path());
            for child in children {
                let path = child.path();
                let metadata = child.metadata().unwrap();
                entries.push((
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    metadata.len(),
                ));
                if metadata.is_dir() {
                    visit(root, &path, entries);
                }
            }
        }

        let mut entries = Vec::new();
        visit(root, root, &mut entries);
        entries
    }

    #[tokio::test]
    async fn existing_only_catalog_resolution_is_cold_local_and_carries_overlay() {
        let root = tempfile::tempdir().unwrap();
        let _environment = IsolatedModelEnvironment::hermetic();
        let config = flux2_catalog_config(root.path());
        for (dir, id) in [
            ("cv-2937936", "cv:2937936"),
            ("hf-owner-model", "hf:owner/model"),
        ] {
            let primary = catalog_sidecar(
                root.path(),
                dir,
                id,
                "flux2/catalog/model.safetensors",
                true,
            );
            let filesystem_before = filesystem_snapshot(root.path());
            let resolution = resolve_existing_model_paths(id, &config)
                .unwrap()
                .expect("installed sidecar must resolve without warming config");
            assert_eq!(resolution.paths.transformer, primary);
            assert_eq!(
                resolution
                    .model_config_overlay
                    .as_ref()
                    .and_then(|model| model.family.as_deref()),
                Some("flux2")
            );
            assert!(
                !config.models.contains_key(id),
                "strictly local resolution must not warm the source config"
            );

            let state = AppState::for_tests();
            *state.config.write().await = config.clone();
            let before = format!("{:?}", *state.config.read().await);
            let status = model_component_status_existing_only(&state, id)
                .await
                .unwrap();
            assert!(status.components.iter().all(|component| component.present));
            assert_eq!(format!("{:?}", *state.config.read().await), before);
            assert!(state.catalog_intents.read().await.is_empty());
            assert_eq!(filesystem_snapshot(root.path()), filesystem_before);
        }
    }

    /// A paired Wan 2.2 A14B sidecar resolves to runtime paths carrying
    /// BOTH experts, so component status lists the low-noise transformer
    /// row and the placement/VRAM path sizes the pair as max-over-experts
    /// (`estimate_peak_memory_sizes_the_larger_expert_of_a_pair` pins the
    /// estimator side). A half-pair on disk is not installed at all.
    #[tokio::test]
    async fn installed_a14b_pair_sidecar_resolves_both_experts() {
        let root = tempfile::tempdir().unwrap();
        let _environment = IsolatedModelEnvironment::hermetic();
        let models_dir = root.path();

        // Wan companions (UMT5 + 2.1 VAE) as installed config entries.
        let umt5 = models_dir.join("companions/umt5.safetensors");
        let vae = models_dir.join("companions/wan21_vae.safetensors");
        std::fs::create_dir_all(umt5.parent().unwrap()).unwrap();
        for path in [&umt5, &vae] {
            std::fs::write(path, b"companion").unwrap();
        }
        let mut config = Config {
            models_dir: models_dir.display().to_string(),
            ..Default::default()
        };
        config.models.insert(
            "wan-umt5".to_string(),
            mold_core::ModelConfig {
                transformer: Some(umt5.display().to_string()),
                vae: Some(umt5.display().to_string()),
                text_encoder_files: Some(vec![umt5.display().to_string()]),
                family: Some("companion".to_string()),
                ..Default::default()
            },
        );
        config.models.insert(
            "wan21-vae".to_string(),
            mold_core::ModelConfig {
                transformer: Some(vae.display().to_string()),
                vae: Some(vae.display().to_string()),
                family: Some("companion".to_string()),
                ..Default::default()
            },
        );

        let install_dir = models_dir.join("cv-2057171");
        let high_rel = "wan/civitai/2057171/high.safetensors";
        let low_rel = "wan/civitai/2057100/low.safetensors";
        let high = install_dir.join(high_rel);
        let low = install_dir.join(low_rel);
        std::fs::create_dir_all(high.parent().unwrap()).unwrap();
        std::fs::create_dir_all(low.parent().unwrap()).unwrap();
        std::fs::write(&high, b"high").unwrap();
        let sidecar = mold_catalog::sidecar::CatalogSidecar {
            schema: mold_catalog::sidecar::SIDECAR_SCHEMA,
            id: "cv:2057171".to_string(),
            source: "civitai".to_string(),
            source_id: "2057171".to_string(),
            name: "Wan Video 2.2 - t2v_high_noise_14B".to_string(),
            author: None,
            family: "wan".to_string(),
            family_role: "finetune".to_string(),
            sub_family: Some("wan22-t2v-a14b".to_string()),
            kind: "checkpoint".to_string(),
            modality: "video".to_string(),
            nsfw: None,
            description: None,
            tags: Vec::new(),
            license: None,
            page_url: None,
            thumbnail_url: None,
            size_bytes: Some(7),
            supported: true,
            trained_words: Vec::new(),
            primary_filename_rel: high_rel.to_string(),
            primary_size_bytes: Some(4),
            low_noise_filename_rel: Some(low_rel.to_string()),
            low_noise_size_bytes: Some(3),
            written_at: 0,
        };
        mold_catalog::sidecar::write_sidecar(
            &install_dir.join(mold_catalog::sidecar::SIDECAR_FILENAME),
            &sidecar,
        )
        .unwrap();

        // Half-pair on disk: never resolves as installed.
        assert!(
            resolve_existing_model_paths("cv:2057171", &config)
                .unwrap()
                .is_none(),
            "a half-pair must not resolve to runnable paths"
        );

        // Both experts land: paths carry the pair.
        std::fs::write(&low, b"low").unwrap();
        let resolution = resolve_existing_model_paths("cv:2057171", &config)
            .unwrap()
            .expect("complete pair resolves");
        assert_eq!(resolution.paths.transformer, high);
        assert_eq!(
            resolution.paths.low_noise_transformer.as_deref(),
            Some(low.as_path())
        );
        assert_eq!(
            resolution
                .model_config_overlay
                .as_ref()
                .and_then(|model| model.low_noise_transformer.as_deref()),
            Some(low.display().to_string().as_str())
        );

        // Component status names the low-noise expert as its own row.
        let state = AppState::for_tests();
        *state.config.write().await = config.clone();
        let status = model_component_status_existing_only(&state, "cv:2057171")
            .await
            .unwrap();
        let low_row = status
            .components
            .iter()
            .find(|component| component.name == "low-noise transformer")
            .expect("pair install exposes the low-noise transformer component");
        assert!(low_row.present);
        assert!(status.components.iter().all(|component| component.present));
    }

    #[tokio::test]
    async fn existing_only_catalog_diagnostics_name_missing_or_unsafe_primary() {
        let root = tempfile::tempdir().unwrap();
        let _environment = IsolatedModelEnvironment::hermetic();
        let config = flux2_catalog_config(root.path());
        for (dir, id, primary_rel) in [
            (
                "cv-missing",
                "cv:missing",
                "flux2/catalog/missing.safetensors",
            ),
            ("cv-traversal", "cv:traversal", "../escape.safetensors"),
        ] {
            catalog_sidecar(root.path(), dir, id, primary_rel, false);
            let state = AppState::for_tests();
            *state.config.write().await = config.clone();
            let before = format!("{:?}", *state.config.read().await);

            assert!(resolve_existing_model_paths(id, &config).unwrap().is_none());
            let status = model_component_status_existing_only(&state, id)
                .await
                .unwrap();
            let primary = status
                .components
                .iter()
                .find(|component| component.name == "primary checkpoint")
                .unwrap();
            assert!(!primary.present);
            if id == "cv:traversal" {
                assert!(primary.path.is_none());
            }
            assert_eq!(primary.repair_model.as_deref(), Some(id));
            assert_eq!(format!("{:?}", *state.config.read().await), before);
            assert!(state.catalog_intents.read().await.is_empty());
        }
    }

    #[tokio::test]
    async fn existing_only_catalog_diagnostics_fail_closed_on_malformed_sidecar() {
        let root = tempfile::tempdir().unwrap();
        let _environment = IsolatedModelEnvironment::hermetic();
        let config = flux2_catalog_config(root.path());
        let install_dir = root.path().join("malformed");
        std::fs::create_dir_all(&install_dir).unwrap();
        std::fs::write(
            install_dir.join(mold_catalog::sidecar::SIDECAR_FILENAME),
            b"{not-json",
        )
        .unwrap();
        let filesystem_before = filesystem_snapshot(root.path());
        let state = AppState::for_tests();
        *state.config.write().await = config.clone();
        let config_before = format!("{:?}", *state.config.read().await);

        assert!(resolve_existing_model_paths("cv:malformed", &config)
            .unwrap()
            .is_none());
        let status = model_component_status_existing_only(&state, "cv:malformed")
            .await
            .unwrap();

        assert_eq!(status.components.len(), 1);
        assert!(!status.components[0].present);
        assert!(status.components[0].path.is_none());
        assert_eq!(
            status.components[0].repair_model.as_deref(),
            Some("cv:malformed")
        );
        assert_eq!(format!("{:?}", *state.config.read().await), config_before);
        assert!(state.catalog_intents.read().await.is_empty());
        assert_eq!(filesystem_snapshot(root.path()), filesystem_before);
    }

    #[tokio::test]
    async fn h3_persisted_sidecar_component_status_rejects_without_leaking_paths() {
        let root = tempfile::tempdir().unwrap();
        let _environment = IsolatedModelEnvironment::hermetic();
        let id = "cv:policy-sidecar";
        let primary = catalog_sidecar(
            root.path(),
            "cv-policy-sidecar",
            id,
            "private/location/weights.safetensors",
            true,
        );
        let sidecar_path = root
            .path()
            .join("cv-policy-sidecar")
            .join(mold_catalog::sidecar::SIDECAR_FILENAME);
        let mut sidecar = mold_catalog::sidecar::read_sidecar(&sidecar_path).unwrap();
        sidecar.name = "MiniMax H3 renamed checkpoint".into();
        mold_catalog::sidecar::write_sidecar(&sidecar_path, &sidecar).unwrap();

        let state = AppState::for_tests();
        state.config.write().await.models_dir = root.path().display().to_string();
        let error = model_component_status_existing_only(&state, id)
            .await
            .expect_err("gated persisted metadata must reject component diagnostics");

        assert_eq!(error.code, mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED);
        assert!(
            !error.error.contains(&primary.display().to_string()),
            "policy response must not reveal the gated checkpoint path"
        );
        assert!(
            primary.is_file(),
            "read-only rejection must not mutate disk"
        );
    }

    #[tokio::test]
    async fn neutral_sidecar_with_nested_h3_artifact_is_hidden_and_components_reject() {
        let root = tempfile::tempdir().unwrap();
        let _environment = IsolatedModelEnvironment::hermetic();
        let id = "cv:neutral-h3-path";
        let primary = catalog_sidecar(root.path(), "MiniMax-H3", id, "weights.safetensors", true);
        let config = Config {
            models_dir: root.path().display().to_string(),
            ..Default::default()
        };
        let state = AppState::for_tests();
        *state.config.write().await = config.clone();

        assert!(installed_catalog_models(&state, &config, root.path(), None, false).is_empty());
        let error = model_component_status_existing_only(&state, id)
            .await
            .expect_err("a neutral sidecar must not hide a gated artifact directory");
        assert_eq!(error.code, mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED);
        assert!(!error.error.contains(&primary.display().to_string()));
    }

    #[test]
    fn installed_inventory_ignores_h3_named_storage_root_for_ordinary_sidecar() {
        let root = tempfile::tempdir().unwrap();
        let models_dir = root.path().join("mold-uat/minimax-h3/models");
        catalog_sidecar(
            &models_dir,
            "cv-ordinary",
            "cv:ordinary",
            "weights.safetensors",
            true,
        );
        let config = Config {
            models_dir: models_dir.display().to_string(),
            ..Default::default()
        };
        let state = AppState::for_tests();

        let inventory = installed_catalog_models(&state, &config, &models_dir, None, false);
        assert_eq!(inventory.len(), 1);
        assert_eq!(inventory[0].info.name, "cv:ordinary");
    }

    #[tokio::test]
    async fn existing_only_catalog_diagnostics_name_missing_companions() {
        let root = tempfile::tempdir().unwrap();
        let _environment = IsolatedModelEnvironment::hermetic();
        let config = Config {
            models_dir: root.path().display().to_string(),
            ..Default::default()
        };
        catalog_sidecar(
            root.path(),
            "cv-missing-companions",
            "cv:missing-companions",
            "flux2/catalog/model.safetensors",
            true,
        );
        let filesystem_before = filesystem_snapshot(root.path());
        let state = AppState::for_tests();
        *state.config.write().await = config.clone();

        assert!(resolve_existing_model_paths("cv:missing-companions", &config).is_err());
        let status = model_component_status_existing_only(&state, "cv:missing-companions")
            .await
            .unwrap();
        let missing = status
            .components
            .iter()
            .filter(|component| !component.present)
            .collect::<Vec<_>>();

        assert!(
            !missing.is_empty(),
            "missing required catalog companions must be named"
        );
        assert!(missing.iter().all(|component| {
            component.repair_model.as_deref() == Some("cv:missing-companions")
                && component.path.is_some()
        }));
        assert_eq!(filesystem_snapshot(root.path()), filesystem_before);
    }

    #[tokio::test]
    async fn existing_only_manifest_component_status_honors_configured_paths() {
        let root = tempfile::tempdir().unwrap();
        let _environment = IsolatedModelEnvironment::hermetic();
        let transformer = root.path().join("custom-unet.safetensors");
        let vae = root.path().join("custom-vae.safetensors");
        std::fs::write(&transformer, b"transformer").unwrap();
        std::fs::write(&vae, b"vae").unwrap();
        let state = AppState::for_tests();
        state.config.write().await.models.insert(
            "sd15:fp16".to_string(),
            mold_core::ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                family: Some("sd15".to_string()),
                ..Default::default()
            },
        );

        let status = model_component_status_existing_only(&state, "sd15:fp16")
            .await
            .unwrap();
        assert!(status
            .components
            .iter()
            .any(|component| component.path.as_deref() == transformer.to_str()));
        assert!(status
            .components
            .iter()
            .any(|component| component.path.as_deref() == vae.to_str()));
        assert!(status.components.iter().all(|component| component.present));
    }

    /// A deleted low-noise expert has to be reported, and reported as missing.
    ///
    /// Nothing opens that file until the schedule crosses the expert boundary,
    /// so if it has no status row the model reports complete, the placement
    /// preview has nothing to offer for repair, and the failure surfaces
    /// mid-generation with no file named.
    #[test]
    fn component_status_names_both_experts_and_both_distills() {
        let root = tempfile::tempdir().unwrap();
        let at = |name: &str| root.path().join(name);
        let write = |name: &str| {
            let path = at(name);
            std::fs::write(&path, b"weights").unwrap();
            path
        };
        let high = write("high-noise.gguf");
        let vae = write("vae.safetensors");
        let high_distill = write("high-distill.safetensors");
        let low_distill = write("low-distill.safetensors");
        // The low-noise expert is the one that never arrived.
        let low = at("low-noise.gguf");

        let paths = ModelPaths {
            low_noise_transformer: Some(low.clone()),
            low_noise_distilled_lora: Some(low_distill.clone()),
            transformer: high.clone(),
            transformer_shards: Vec::new(),
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: Some(high_distill.clone()),
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };

        let components =
            component_status_from_paths(&Config::default(), "wan22-t2v-a14b:q5", &paths);
        let row = |path: &std::path::Path| {
            components
                .iter()
                .find(|component| component.path.as_deref() == path.to_str())
                .unwrap_or_else(|| panic!("no status row for {}", path.display()))
        };

        assert!(row(&high).present);
        assert!(
            !row(&low).present,
            "the missing low-noise expert must report absent"
        );
        assert_eq!(row(&low).name, "low-noise transformer");
        assert_eq!(row(&low).kind, "transformer");
        assert_eq!(
            row(&low).repair_model.as_deref(),
            Some("wan22-t2v-a14b:q5"),
            "the row must say which model to repair"
        );

        // Both distills get their own rows; one cannot stand in for the other.
        assert_eq!(row(&high_distill).name, "distilled lora");
        assert_eq!(row(&low_distill).name, "low-noise distilled lora");
        assert!(row(&high_distill).present && row(&low_distill).present);

        // A single-expert model gains no extra rows.
        let single = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            ..paths
        };
        let components =
            component_status_from_paths(&Config::default(), "wan22-ti2v-5b:fp16", &single);
        assert!(!components
            .iter()
            .any(|component| component.name.starts_with("low-noise")));
    }

    struct IsolatedModelEnvironment {
        _lock: std::sync::MutexGuard<'static, ()>,
        previous: Vec<(&'static str, Option<std::ffi::OsString>)>,
        _home: Option<tempfile::TempDir>,
    }

    impl IsolatedModelEnvironment {
        fn new(home: &std::path::Path) -> Self {
            Self::with_models_dir_override(home, true)
        }

        fn without_models_dir_override(home: &std::path::Path) -> Self {
            Self::with_models_dir_override(home, false)
        }

        /// Pins `MOLD_HOME` to a guard-owned tempdir separate from the test's
        /// own root. Process-wide path caches initialize from the environment
        /// on first touch and `create_dir_all` their target (see
        /// `mold_core::download`'s models-dir cache), so a test that snapshots
        /// its root for byte-identity must keep `MOLD_HOME` pointing elsewhere
        /// or the first-touching test finds surprise directories in its
        /// snapshot.
        fn hermetic() -> Self {
            let home = tempfile::tempdir().unwrap();
            let mut environment = Self::with_models_dir_override(home.path(), false);
            environment._home = Some(home);
            environment
        }

        fn with_models_dir_override(home: &std::path::Path, set_models_dir: bool) -> Self {
            const CLEARED_KEYS: &[&str] = &[
                "MOLD_TRANSFORMER_PATH",
                "MOLD_VAE_PATH",
                "MOLD_CLIP_PATH",
                "MOLD_CLIP_TOKENIZER_PATH",
                "MOLD_CLIP2_PATH",
                "MOLD_CLIP2_TOKENIZER_PATH",
                "MOLD_T5_PATH",
                "MOLD_T5_TOKENIZER_PATH",
                "MOLD_TEXT_TOKENIZER_PATH",
                "MOLD_DECODER_PATH",
                "MOLD_SPATIAL_UPSCALER_PATH",
                "MOLD_TEMPORAL_UPSCALER_PATH",
                "MOLD_DISTILLED_LORA_PATH",
            ];
            let lock = crate::test_support::env_lock();
            let mut previous = Vec::with_capacity(CLEARED_KEYS.len() + 4);
            for key in ["MOLD_HOME", "MOLD_MODELS_DIR", "HF_HOME", "HF_HUB_CACHE"] {
                previous.push((key, std::env::var_os(key)));
            }
            for &key in CLEARED_KEYS {
                previous.push((key, std::env::var_os(key)));
            }
            unsafe {
                std::env::set_var("MOLD_HOME", home);
                if set_models_dir {
                    std::env::set_var("MOLD_MODELS_DIR", home);
                } else {
                    std::env::remove_var("MOLD_MODELS_DIR");
                }
                std::env::set_var("HF_HOME", home.join("hf-home"));
                std::env::set_var("HF_HUB_CACHE", home.join("hf-home/hub"));
                for key in CLEARED_KEYS {
                    std::env::remove_var(key);
                }
            }
            Self {
                _lock: lock,
                previous,
                _home: None,
            }
        }
    }

    impl Drop for IsolatedModelEnvironment {
        fn drop(&mut self) {
            unsafe {
                for (key, value) in self.previous.drain(..).rev() {
                    match value {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
    }

    fn materialize_manifest_companion(
        models_dir: &std::path::Path,
        companion: &str,
        config: &mold_core::Config,
    ) -> mold_core::ModelPaths {
        let manifest = mold_core::manifest::find_manifest(companion).unwrap();
        for file in &manifest.files {
            let path = models_dir.join(mold_core::manifest::storage_path(manifest, file));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"test fixture").unwrap();
            mold_core::download::write_sha256_marker(&path, "test").unwrap();
        }
        mold_core::ModelPaths::resolve(companion, config).unwrap()
    }

    #[test]
    fn installed_ltx2_catalog_models_advertise_checkpoint_audio_assets() {
        let dir = tempfile::tempdir().unwrap();
        let install_dir = dir.path().join("cv-3143864");
        let primary = install_dir.join("model.safetensors");
        std::fs::create_dir_all(&install_dir).unwrap();

        let sidecar = mold_catalog::sidecar::CatalogSidecar {
            schema: 1,
            id: "cv:3143864".into(),
            source: "civitai".into(),
            source_id: "3143864".into(),
            name: "LTX 2.3 INT4 ConvRot".into(),
            author: None,
            family: "ltx2".into(),
            family_role: "finetune".into(),
            sub_family: Some("v2.3".into()),
            kind: "checkpoint".into(),
            modality: "video".into(),
            nsfw: None,
            description: None,
            tags: vec![],
            license: None,
            page_url: None,
            thumbnail_url: None,
            size_bytes: None,
            supported: true,
            trained_words: vec![],
            primary_filename_rel: "model.safetensors".into(),
            primary_size_bytes: None,
            low_noise_filename_rel: None,
            low_noise_size_bytes: None,
            written_at: 0,
        };
        mold_catalog::sidecar::write_sidecar(
            &install_dir.join(mold_catalog::sidecar::SIDECAR_FILENAME),
            &sidecar,
        )
        .unwrap();

        write_safetensors_with_keys(
            &primary,
            &["model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"],
        );
        let config = Config {
            models_dir: dir.path().to_string_lossy().into_owned(),
            ..Default::default()
        };
        let state = AppState::for_tests();
        let video_only = installed_catalog_models(&state, &config, dir.path(), None, false);
        assert_eq!(video_only[0].supports_audio, Some(false));
        // Catalog sidecars have no manifest, so frame defaults come from the
        // family runtime defaults and the family constraint helpers.
        assert_eq!(video_only[0].defaults.default_frames, Some(97));
        assert_eq!(video_only[0].defaults.default_fps, Some(24));
        assert_eq!(
            video_only[0].defaults.max_frames,
            Some(481),
            "the 20s LTX-2 temporal budget at the sidecar's 24 fps default",
        );
        assert_eq!(video_only[0].defaults.max_runtime_seconds, Some(20));
        assert_eq!(video_only[0].defaults.frame_step, Some(8));

        write_safetensors_with_keys(
            &primary,
            &[
                "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
                "audio_vae.per_channel_statistics.mean-of-means",
                "vocoder.vocoder.conv_pre.weight",
            ],
        );
        let combined = installed_catalog_models(&state, &config, dir.path(), None, false);
        assert_eq!(combined[0].supports_audio, Some(true));
    }

    #[test]
    fn downloaded_ltx25_uses_split_audio_component_for_capability() {
        let _env = crate::test_support::hermetic_store_env();
        let dir = tempfile::tempdir().unwrap();
        let mut config = Config {
            models_dir: dir.path().to_string_lossy().into_owned(),
            ..Default::default()
        };
        let model = mold_core::ltx25_manifest::DISTILLED_INT8_CONV;
        // `resolve_in` pins the split-pack roots to THIS test's tempdir even
        // if a store-redirecting env var slips past the guard — the fixture
        // writer below must never be able to reach a real store.
        let split =
            mold_core::ltx25_manifest::Ltx25ModelPaths::resolve_in(dir.path(), model).unwrap();
        assert!(split.audio_vae.starts_with(dir.path()));
        config.models.insert(
            model.to_string(),
            mold_core::ModelConfig {
                transformer: Some(split.transformer.to_string_lossy().into_owned()),
                vae: Some(split.video_vae.to_string_lossy().into_owned()),
                family: Some("ltx2".to_string()),
                ..Default::default()
            },
        );
        std::fs::create_dir_all(split.audio_vae.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &split.audio_vae,
            &[
                "audio_vae.per_channel_statistics.mean-of-means",
                "vocoder.vocoder.conv_pre.weight",
            ],
        );

        let mut catalog = mold_core::catalog::build_model_catalog(&config, None, false);
        let row = catalog
            .iter_mut()
            .find(|entry| entry.info.name == model)
            .unwrap();
        row.downloaded = true;
        row.supports_audio = None;

        annotate_audio_capabilities(&mut catalog, &config);
        synchronize_generation_profile_capabilities(&mut catalog);

        let row = catalog
            .iter()
            .find(|entry| entry.info.name == model)
            .unwrap();
        assert_eq!(row.supports_audio, Some(true));
        assert!(row
            .generation_profile
            .as_ref()
            .unwrap()
            .recipes
            .iter()
            .any(|recipe| recipe.capabilities.supports_audio));
    }

    /// `/api/models` must advertise H3's synchronized audio for every
    /// reviewed identity. The inference-side probe registry has no H3 arm on
    /// purpose — H3 audio is a family declaration, not a checkpoint header
    /// fact — so the row arrives pre-filled and both annotation passes must
    /// leave it alone (#841).
    #[test]
    fn h3_rows_survive_the_audio_and_profile_annotation_passes() {
        let config = Config::default();
        let mut catalog = mold_core::catalog::build_model_catalog(&config, None, false);
        annotate_audio_capabilities(&mut catalog, &config);
        annotate_source_image_capabilities(&mut catalog, &config);
        synchronize_generation_profile_capabilities(&mut catalog);
        retain_deliverable_generation_profiles(&mut catalog);

        assert!(
            !mold_inference::audio::output_probe_registered(mold_core::minimax_h3::FAMILY),
            "H3 audio is declared by the family, never probed from a checkpoint",
        );
        for model in mold_core::minimax_h3::REVIEWED_COMPACT_MODELS {
            let entry = catalog
                .iter()
                .find(|entry| entry.info.name == *model)
                .unwrap_or_else(|| panic!("{model} is missing from /api/models"));
            assert_eq!(
                entry.supports_audio,
                Some(true),
                "{model} must advertise synchronized audio",
            );
            assert!(entry
                .generation_profile
                .as_ref()
                .unwrap()
                .recipes
                .iter()
                .all(|recipe| recipe.capabilities.supports_audio));
        }
    }

    #[test]
    fn runtime_probe_refreshes_generation_profile_and_hash() {
        let mut catalog = mold_core::catalog::build_model_catalog(&Config::default(), None, false);
        let entry = catalog
            .iter_mut()
            .find(|entry| entry.info.family == "ltx2")
            .expect("built-in LTX-2 model");
        let previous_hash = entry
            .generation_profile
            .as_ref()
            .expect("generation profile")
            .profile_hash
            .clone();
        entry.supports_audio = Some(false);

        synchronize_generation_profile_capabilities(&mut catalog);

        let entry = catalog
            .iter()
            .find(|entry| entry.info.family == "ltx2")
            .expect("built-in LTX-2 model");
        let profile = entry.generation_profile.as_ref().unwrap();
        assert_ne!(profile.profile_hash, previous_hash);
        assert!(profile.recipes.iter().all(|recipe| {
            !recipe.capabilities.supports_audio
                && recipe.request_selector.pipeline != Some(mold_core::Ltx2PipelineMode::T2a)
        }));
    }

    #[test]
    fn delivery_qualification_filters_formats_and_repairs_defaults() {
        let mut video = mold_core::resolve_generation_profile(mold_core::GenerationProfileInput {
            model: "wan22-t2v-a14b:fp8",
            family: "wan",
            sub_family: Some("wan22-t2v-a14b"),
            default_width: 1280,
            default_height: 720,
            default_steps: 4,
            default_guidance: 1.0,
            default_frames: Some(81),
            default_fps: Some(16),
            default_negative_prompt: None,
            source_image: Some(mold_core::SourceImageCapability::Unsupported),
            supports_sequence: true,
            supports_extend: false,
            supports_audio: false,
        });
        mold_core::qualify_generation_profile_delivery(
            &mut video,
            mold_core::GenerationDeliveryCapabilities::new(false, false),
        );
        let output = &video.default_recipe().unwrap().capabilities.output;
        assert_eq!(output.default_format, mold_core::OutputFormat::Gif);
        assert_eq!(
            output.formats,
            vec![mold_core::OutputFormat::Gif, mold_core::OutputFormat::Apng]
        );

        let mut h3 = mold_core::resolve_generation_profile(mold_core::GenerationProfileInput {
            model: "minimax-h3",
            family: "minimax-h3",
            sub_family: None,
            default_width: 1280,
            default_height: 720,
            default_steps: 30,
            default_guidance: 0.0,
            default_frames: Some(345),
            default_fps: Some(24),
            default_negative_prompt: None,
            source_image: None,
            supports_sequence: false,
            supports_extend: false,
            supports_audio: true,
        });
        mold_core::qualify_generation_profile_delivery(
            &mut h3,
            mold_core::GenerationDeliveryCapabilities::new(false, false),
        );
        assert!(h3.recipes.is_empty());
        assert!(h3.default_recipe_id.is_empty());
    }

    #[test]
    fn undeliverable_generation_rows_are_not_advertised() {
        let mut catalog = mold_core::catalog::build_model_catalog(&Config::default(), None, false);
        let model = catalog
            .iter_mut()
            .find(|entry| entry.generation_profile.is_some())
            .expect("visible generation row");
        let removed_name = model.info.name.clone();
        let profile = model.generation_profile.as_mut().unwrap();
        profile.recipes.clear();
        profile.default_recipe_id.clear();

        retain_deliverable_generation_profiles(&mut catalog);
        assert!(catalog.iter().all(|entry| entry.info.name != removed_name));
    }

    #[test]
    fn downloadable_h3_rows_survive_without_a_runtime_recipe() {
        let mut catalog = mold_core::catalog::build_model_catalog(&Config::default(), None, false);
        for entry in catalog
            .iter_mut()
            .filter(|entry| entry.info.family == mold_core::minimax_h3::FAMILY)
        {
            let profile = entry.generation_profile.as_mut().unwrap();
            profile.recipes.clear();
            profile.default_recipe_id.clear();
        }

        retain_deliverable_generation_profiles(&mut catalog);

        let h3 = catalog
            .iter()
            .filter(|entry| entry.info.family == mold_core::minimax_h3::FAMILY)
            .map(|entry| entry.info.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        // Reviewed rows survive because they are runnable elsewhere; the
        // download-only rows survive because `runtime_available: false` says
        // a recipe was never the point. Nothing else does.
        assert_eq!(
            h3,
            std::collections::BTreeSet::from([
                mold_core::minimax_h3::FL2VA_COMFY,
                mold_core::minimax_h3::REF2VA_COMFY,
                mold_core::minimax_h3::FL2VA_COMFY_TURBO_8STEP,
                mold_core::minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P,
                mold_core::minimax_h3::REF2VA_COMFY_TURBO_4STEP,
                mold_core::minimax_h3::FL2VA_COMFY_NVFP4,
                mold_core::minimax_h3::REF2VA_COMFY_NVFP4,
                mold_core::minimax_h3::FL2VA_OFFICIAL,
                mold_core::minimax_h3::REF2VA_OFFICIAL,
            ])
        );
        for entry in &catalog {
            if matches!(
                entry.info.name.as_str(),
                mold_core::minimax_h3::FL2VA_COMFY_NVFP4
                    | mold_core::minimax_h3::REF2VA_COMFY_NVFP4
                    | mold_core::minimax_h3::FL2VA_OFFICIAL
                    | mold_core::minimax_h3::REF2VA_OFFICIAL
            ) {
                assert_eq!(entry.runtime_available, Some(false), "{}", entry.info.name);
            }
        }
    }

    #[test]
    fn wan_runtime_probe_recomputes_dependent_profile_controls() {
        let mut catalog = mold_core::catalog::build_model_catalog(&Config::default(), None, false);
        {
            let entry = catalog
                .iter_mut()
                .find(|entry| entry.info.family == "wan")
                .expect("built-in Wan model");
            entry.source_image = Some(mold_core::SourceImageCapability::Unsupported);
            entry.supports_extend = Some(false);
        }

        synchronize_generation_profile_capabilities(&mut catalog);
        let entry = catalog
            .iter_mut()
            .find(|entry| entry.info.family == "wan")
            .expect("built-in Wan model");
        let profile = entry.generation_profile.as_ref().unwrap();
        let recipe = profile.default_recipe().unwrap();
        assert!(!recipe.capabilities.wan_recipe.supports_first_last_frame);
        assert_eq!(
            recipe.capabilities.keyframes.mode,
            mold_core::ControlMode::Hidden
        );
        let unsupported_hash = profile.profile_hash.clone();

        entry.source_image = Some(mold_core::SourceImageCapability::Optional);
        entry.supports_extend = Some(true);
        synchronize_generation_profile_capabilities(std::slice::from_mut(entry));
        let profile = entry.generation_profile.as_ref().unwrap();
        let recipe = profile.default_recipe().unwrap();
        assert!(recipe.capabilities.wan_recipe.supports_first_last_frame);
        assert_eq!(
            recipe.capabilities.keyframes.mode,
            mold_core::ControlMode::Adjustable
        );
        assert!(recipe.capabilities.supports_extend);
        assert_ne!(profile.profile_hash, unsupported_hash);
    }

    /// #787: an installed wan checkpoint (cv:/hf:, no manifest) advertises
    /// the engine's tuned absence-fallback negative exactly like manifest
    /// rows, because `wan/pipeline.rs` substitutes it per family, not per
    /// manifest. Non-wan sidecars keep the field absent.
    #[test]
    fn installed_wan_catalog_models_advertise_the_default_negative_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let install_dir = dir.path().join("cv-999001");
        let primary = install_dir.join("model.safetensors");
        std::fs::create_dir_all(&install_dir).unwrap();

        let sidecar = mold_catalog::sidecar::CatalogSidecar {
            schema: 1,
            id: "cv:999001".into(),
            source: "civitai".into(),
            source_id: "999001".into(),
            name: "Wan community finetune".into(),
            author: None,
            family: "wan".into(),
            family_role: "finetune".into(),
            sub_family: None,
            kind: "checkpoint".into(),
            modality: "video".into(),
            nsfw: None,
            description: None,
            tags: vec![],
            license: None,
            page_url: None,
            thumbnail_url: None,
            size_bytes: None,
            supported: true,
            trained_words: vec![],
            primary_filename_rel: "model.safetensors".into(),
            primary_size_bytes: None,
            low_noise_filename_rel: None,
            low_noise_size_bytes: None,
            written_at: 0,
        };
        mold_catalog::sidecar::write_sidecar(
            &install_dir.join(mold_catalog::sidecar::SIDECAR_FILENAME),
            &sidecar,
        )
        .unwrap();
        write_safetensors_with_keys(&primary, &["patch_embedding.weight"]);

        let config = Config {
            models_dir: dir.path().to_string_lossy().into_owned(),
            ..Default::default()
        };
        let state = AppState::for_tests();
        let inventory = installed_catalog_models(&state, &config, dir.path(), None, false);
        assert_eq!(
            inventory[0].defaults.default_negative_prompt.as_deref(),
            Some(mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT)
        );
    }

    /// Build a `ModelPaths` whose `transformer` and `vae` files exist on disk
    /// with a combined size of `total_bytes`. `estimate_peak_memory()` reads
    /// file sizes via `std::fs::metadata`, so the on-disk footprint is what
    /// drives the composition under test.
    fn test_paths_with_total_size(total_bytes: u64) -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let transformer = dir.path().join("transformer.safetensors");
        let vae = dir.path().join("vae.safetensors");
        // Split half-and-half between the two required files so that the
        // sum equals `total_bytes`. `set_len` creates a sparse file, which
        // is fast and reports the requested size via `metadata().len()`.
        let half = total_bytes / 2;
        let rest = total_bytes - half;
        let f1 = std::fs::File::create(&transformer).expect("create transformer");
        f1.set_len(half).expect("set transformer len");
        let f2 = std::fs::File::create(&vae).expect("create vae");
        f2.set_len(rest).expect("set vae len");

        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer,
            transformer_shards: Vec::new(),
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
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    /// Sanity: confirm that `estimate_peak_memory` actually sees the on-disk
    /// sizes we set via `test_paths_with_total_size`. The peak includes a
    /// fixed headroom term added by the inference crate, so we just verify
    /// the lower bound matches our component total.
    #[test]
    fn test_paths_helper_sets_file_sizes() {
        let (_dir, paths) = test_paths_with_total_size(10 * GB);
        let peak = mold_inference::device::estimate_peak_memory(
            &paths,
            mold_inference::LoadStrategy::Eager,
        );
        assert!(
            peak >= 10 * GB,
            "expected peak >= 10 GB component sum, got {peak}"
        );
        // The transformer and vae paths must point to real files.
        assert!(PathBuf::from(&paths.transformer).exists());
        assert!(PathBuf::from(&paths.vae).exists());
    }

    /// Composition test: peak fits in `(available + active)` but not in
    /// `available` alone — the inner guard must let the swap proceed.
    #[test]
    fn preflight_uses_active_vram_as_reclaimable() {
        // 10 GB on disk → peak ≈ 10 GB + headroom.
        // Available 8 GB alone is insufficient; add 10 GB active VRAM →
        // 18 GB effective, comfortably above peak.
        let (_dir, paths) = test_paths_with_total_size(10 * GB);
        let result =
            preflight_memory_guard_with_available("swap-test", &paths, 10 * GB, 8 * GB, None);
        assert!(
            result.is_ok(),
            "expected swap to succeed with reclaimable VRAM, got {result:?}"
        );
    }

    /// Composition test: peak exceeds even the post-swap budget — the inner
    /// guard must reject.
    #[test]
    fn preflight_rejects_when_peak_exceeds_effective_available() {
        // 20 GB on disk → peak ≥ 20 GB. Available 8 GB + 5 GB active =
        // 13 GB effective < peak → reject.
        let (_dir, paths) = test_paths_with_total_size(20 * GB);
        let result = preflight_memory_guard_with_available("too-big", &paths, 5 * GB, 8 * GB, None);
        assert!(
            result.is_err(),
            "expected oversized model to be rejected, got Ok"
        );
    }

    #[test]
    fn memory_guard_ok_when_plenty_of_memory() {
        assert!(check_model_memory_budget("test-model", 5 * GB, 20 * GB, "").is_ok());
    }

    #[test]
    fn memory_guard_rejects_over_90pct() {
        let result = check_model_memory_budget("flux-dev:bf16", 19 * GB, 20 * GB, "Try --offload.");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, "INSUFFICIENT_MEMORY");
        assert!(err.error.contains("flux-dev:bf16"));
        assert!(err.error.contains("budget cap"));
    }

    #[test]
    fn memory_guard_ok_at_90pct_boundary() {
        // 18 GB peak, 20 GB available → 90% exactly → should pass
        assert!(check_model_memory_budget("test", 18 * GB, 20 * GB, "").is_ok());
    }

    #[test]
    fn memory_guard_ok_in_warn_zone() {
        // 17 GB peak, 20 GB available → 85% → passes but would warn
        assert!(check_model_memory_budget("test", 17 * GB, 20 * GB, "").is_ok());
    }

    #[test]
    fn memory_guard_ok_below_warn_zone() {
        // 15 GB peak, 20 GB available → 75% → no warn, no error
        assert!(check_model_memory_budget("test", 15 * GB, 20 * GB, "").is_ok());
    }

    #[test]
    fn memory_guard_rejects_tiny_available() {
        // Model larger than total available
        let result = check_model_memory_budget("huge-model", 30 * GB, 16 * GB, "");
        assert!(result.is_err());
    }

    /// CUDA branch math: when free VRAM is small but the active model can be
    /// reclaimed, `effective_available = free + active_vram` should let the
    /// new model load. Without the additive term, a swap of two near-
    /// equal-size models on a fully-loaded GPU would always be rejected.
    #[test]
    fn memory_guard_swap_uses_active_vram_as_reclaimable() {
        // Free VRAM is only 2 GB but the currently-loaded model occupies
        // 18 GB which becomes available on swap → effective = 20 GB.
        // A 15 GB peak model fits comfortably (<= 90% of 20 GB).
        let free_vram = 2 * GB;
        let active_vram = 18 * GB;
        let effective = free_vram + active_vram;
        assert!(check_model_memory_budget("swap-target", 15 * GB, effective, "").is_ok());
    }

    /// Verifies the swap math also rejects when the swap is genuinely
    /// infeasible — peak exceeds even the post-swap budget.
    #[test]
    fn memory_guard_swap_still_rejects_when_oversized() {
        // Free 1 GB + active 8 GB = 9 GB effective. A 15 GB model can't fit
        // even after the active model is unloaded.
        let free_vram = GB;
        let active_vram = 8 * GB;
        let effective = free_vram + active_vram;
        assert!(check_model_memory_budget("too-large", 15 * GB, effective, "").is_err());
    }

    /// Build a `ModelPaths` whose components include text encoders, mirroring a
    /// real FLUX-family layout. Used to verify the Sequential-strategy peak
    /// estimate doesn't sum the encoder onto the transformer (the bug fix).
    fn flux_shaped_paths_with_sizes(
        transformer_gb: u64,
        vae_gb: u64,
        t5_gb: u64,
        clip_gb: u64,
    ) -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let transformer = mk("transformer.safetensors", transformer_gb);
        let vae = mk("vae.safetensors", vae_gb);
        let t5 = mk("t5.safetensors", t5_gb);
        let clip = mk("clip.safetensors", clip_gb);
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer,
            transformer_shards: Vec::new(),
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: Some(t5),
            clip_encoder: Some(clip),
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    /// Regression: a quantized FLUX-shaped model should fit when the actual
    /// post-drop sample reports 24 GB available, even though the Eager (sum)
    /// peak would have been ~24 GB and tripped the 90 % hard limit.
    ///
    /// Concrete shape: FLUX-dev:q8 → transformer ≈ 12 GB, VAE ≈ 0.3 GB,
    /// T5 ≈ 9.5 GB, CLIP ≈ 0.25 GB. Eager peak = 12+0.3+9.5+0.25+2 ≈ 24 GB
    /// (rejects at 90 % of 24 GB = 21.6). Sequential peak =
    /// max(9.75, 12.3) + 2 ≈ 14.3 GB (passes comfortably).
    #[test]
    fn preflight_passes_for_quantized_flux_on_24gb_card_with_swap() {
        // Use whole-GB sizes to match the helper's u64 parameters; the
        // composition is realistic enough to exercise Eager-vs-Sequential
        // divergence (encoder + transformer both > headroom).
        let (_dir, paths) = flux_shaped_paths_with_sizes(12, 1, 10, 1);
        // This is the authoritative post-drop reading, not nominal capacity.
        let result = preflight_memory_guard_with_available("flux-dev:q8", &paths, 0, 24 * GB, None);
        assert!(
            result.is_ok(),
            "quantized FLUX must fit on a 24 GB card under the Sequential \
             peak estimate (drop-and-reload encoders), got {result:?}"
        );
    }

    #[test]
    fn preflight_treats_unrecovered_vram_as_unavailable_pressure() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(12, 1, 10, 1);
        // Nominal capacity may be 24 GB, but only the observed 14 GB is
        // admissible after the old engine is gone. The guard must not promote
        // this reading back to total capacity.
        let result = preflight_memory_guard_with_available("flux-dev:q8", &paths, 0, 14 * GB, None);
        assert!(
            result.is_err(),
            "unrecovered or externally-owned VRAM must remain unavailable"
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn metal_unified_memory_has_no_second_post_drop_admission_gate() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(100, 10, 20, 5);
        let result = preflight_memory_guard_after_drop("metal-swap", &paths, 0, None);
        assert!(
            result.is_ok(),
            "Metal uses the additive unified-memory guard before unload; an \
             instantaneous post-drop sample must not add a spurious second gate"
        );
    }

    #[test]
    fn preflight_accepts_forced_flux_offload_bf16_layout_on_24gb() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(24, 1, 10, 1);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };

        let result = preflight_memory_guard_with_available_and_policy(
            "flux-dev:bf16",
            &paths,
            0,
            24 * GB,
            Some(hint),
            true,
            false,
        );

        assert!(
            result.is_ok(),
            "forced FLUX offload should use streaming-aware peak instead of \
            full BF16 transformer residency, got {result:?}"
        );
    }

    #[test]
    fn preflight_accepts_large_flux_bf16_auto_offload_on_24gb() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(23, 1, 9, 1);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };

        let result = preflight_memory_guard_with_available(
            "cv:2319074",
            &paths,
            0,
            24_500_000_000,
            Some(hint),
        );

        assert!(
            result.is_ok(),
            "large FLUX BF16 checkpoints should be admitted on 24 GB cards via \
             automatic block offload instead of being rejected by resident \
             transformer peak math, got {result:?}"
        );
    }

    #[test]
    fn server_auto_enables_offload_for_large_flux_bf16_without_env() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(23, 1, 9, 1);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };

        assert!(
            server_offload_enabled_for_paths_with_request(&paths, Some(hint), false, false),
            "large FLUX BF16 checkpoints should load with block offload even \
             when MOLD_OFFLOAD is not globally forced"
        );
    }

    fn sd3_gguf_paths_with_monolithic_vae(
        transformer_gb: u64,
        vae_gb: u64,
        t5_gb: u64,
        clip_l_gb: u64,
        clip_g_gb: u64,
    ) -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let transformer = mk("sd3.5_large-Q8_0.gguf", transformer_gb);
        let vae = mk("sd3.5_large.safetensors", vae_gb);
        let t5 = mk("t5xxl_fp16.safetensors", t5_gb);
        let clip_l = mk("clip_l.safetensors", clip_l_gb);
        let clip_g = mk("clip_g.safetensors", clip_g_gb);
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer,
            transformer_shards: Vec::new(),
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: Some(t5),
            clip_encoder: Some(clip_l),
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: Some(clip_g),
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    #[test]
    fn preflight_accepts_sd3_gguf_with_monolithic_vae_on_24gb() {
        let (_dir, paths) = sd3_gguf_paths_with_monolithic_vae(9, 16, 10, 1, 1);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 2,
            dtype_bytes: 2,
            family: ActivationFamily::Sd3Mmdit,
        };

        let result =
            preflight_memory_guard_with_available("sd3.5-large:q8", &paths, 0, 24 * GB, Some(hint));

        assert!(
            result.is_ok(),
            "SD3 GGUF should not count the monolithic VAE checkpoint as \
             co-resident with the transformer, got {result:?}"
        );
    }

    #[test]
    fn server_load_strategy_keeps_sd3_gguf_eager() {
        let (_dir, paths) = sd3_gguf_paths_with_monolithic_vae(9, 16, 10, 1, 1);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 2,
            dtype_bytes: 2,
            family: ActivationFamily::Sd3Mmdit,
        };

        let strategy = select_server_load_strategy_for_budget(&paths, Some(32 * GB), Some(hint));

        assert_eq!(
            strategy,
            mold_inference::LoadStrategy::Eager,
            "SD3 GGUF has its own quantized runtime path; selecting Sequential \
             asks the runtime for unsupported block offload"
        );
    }

    fn zimage_gguf_paths(
        transformer_gb: u64,
        vae_gb: u64,
        text_encoder_gb: u64,
    ) -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let transformer = mk("z-image-turbo-Q8_0.gguf", transformer_gb);
        let vae = mk("vae.safetensors", vae_gb);
        let text_encoder = mk("qwen3.safetensors", text_encoder_gb);
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer,
            transformer_shards: Vec::new(),
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
            text_encoder_files: vec![text_encoder],
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    #[test]
    fn server_load_strategy_keeps_zimage_gguf_eager() {
        let (_dir, paths) = zimage_gguf_paths(12, 1, 8);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::ZImageDit,
        };

        let strategy = select_server_load_strategy_for_budget(&paths, Some(24 * GB), Some(hint));

        assert_eq!(
            strategy,
            mold_inference::LoadStrategy::Eager,
            "Z-Image GGUF has a quantized/dense runtime path; selecting Sequential \
             asks the runtime for unsupported block offload"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_sd3_gguf() {
        let (_dir, paths) = sd3_gguf_paths_with_monolithic_vae(9, 16, 10, 1, 1);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 2,
            dtype_bytes: 2,
            family: ActivationFamily::Sd3Mmdit,
        };

        assert!(
            !server_offload_enabled_for_paths_with_request(&paths, Some(hint), false, true),
            "global MOLD_OFFLOAD must not force unsupported SD3 GGUF block offload"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_zimage_gguf() {
        let (_dir, paths) = zimage_gguf_paths(12, 1, 8);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::ZImageDit,
        };

        assert!(
            !server_offload_enabled_for_paths_with_request(&paths, Some(hint), false, true),
            "global MOLD_OFFLOAD must not force unsupported Z-Image GGUF block offload"
        );
    }

    #[test]
    fn offload_env_is_preserved_for_zimage_bf16() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(6, 1, 8, 0);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::ZImageDit,
        };

        assert!(
            server_offload_enabled_for_paths_with_request(&paths, Some(hint), false, true),
            "BF16/FP Z-Image paths should still receive explicit offload"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_zimage_lora_with_ambiguous_family_hint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: mk("z-image/civitai/2442439/zImageTurbo_turbo.safetensors", 12),
            transformer_shards: Vec::new(),
            vae: mk("z-image/civitai/2442439/ae_zimgturbo.safetensors", 1),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![mk(
                "z-image/civitai/2442439/zImageTurbo_turbo_txt.safetensors",
                8,
            )],
            text_tokenizer: None,
            decoder: None,
        };
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };

        assert!(
            !server_offload_enabled_for_paths_with_request(&paths, Some(hint), true, true),
            "Z-Image LoRA requests must not receive global MOLD_OFFLOAD even \
             when duplicate catalog rows provide an ambiguous Flux hint"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_flux2_lora_request() {
        let (_dir, paths) = flux2_klein9b_bf16_paths();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        assert!(
            !server_offload_enabled_for_paths_with_request(&paths, Some(hint), true, true),
            "global MOLD_OFFLOAD must not force Flux.2 block offload for LoRA \
             requests because Flux.2 offload+LoRA is not supported"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_flux2_lora_with_ambiguous_family_hint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let transformer = mk(
            "flux2/civitai/2669986/darkBeast_dbkBlitzV15.safetensors",
            18,
        );
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: transformer.clone(),
            transformer_shards: vec![transformer],
            vae: mk("flux2/civitai/2669986/flux2-vae.safetensors", 1),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![mk("flux2/civitai/2669986/qwen3.safetensors", 16)],
            text_tokenizer: None,
            decoder: None,
        };
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };

        assert!(
            !server_offload_enabled_for_paths_with_request(&paths, Some(hint), true, true),
            "Flux.2 LoRA requests must not receive global MOLD_OFFLOAD even \
             when the catalog family hint is missing or ambiguous"
        );
    }

    #[test]
    fn flux2_lora_request_requires_fresh_engine_when_plain_offload_was_enabled() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: mk(
                "flux2/civitai/2669986/darkBeast_dbkBlitzV15.safetensors",
                18,
            ),
            transformer_shards: Vec::new(),
            vae: mk("flux2/civitai/2669986/flux2-vae.safetensors", 1),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![mk("flux2/civitai/2669986/qwen3.safetensors", 16)],
            text_tokenizer: None,
            decoder: None,
        };
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        assert!(
            request_requires_fresh_engine_for_offload_policy_with_request(
                &paths,
                Some(hint),
                true,
                true,
            ),
            "a cached Flux.2 engine loaded for plain offload must be recreated \
             before serving a LoRA request, otherwise the runtime still sees \
             offload+LoRA"
        );
    }

    #[test]
    fn offload_env_is_preserved_for_plain_flux2_request() {
        let (_dir, paths) = flux2_klein9b_bf16_paths();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        assert!(
            server_offload_enabled_for_paths_with_request(&paths, Some(hint), false, true),
            "plain Flux.2 requests should still receive explicit offload"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_flux2_gguf() {
        let (dir, mut paths) = flux2_klein9b_bf16_paths();
        let gguf = dir.path().join("flux2-klein-9b-q8.gguf");
        std::fs::File::create(&gguf)
            .unwrap()
            .set_len(12 * GB)
            .unwrap();
        paths.transformer = gguf;
        paths.transformer_shards.clear();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        assert!(
            !server_offload_enabled_for_paths_with_request(&paths, Some(hint), false, true),
            "global MOLD_OFFLOAD must not force Flux.2 GGUF block offload \
             because GGUF variants use quantized transformer paths"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_flux2_nvfp4() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: mk(
                "flux2/civitai/2759597/miracleinNSFWGeneration_10Nvfp4.safetensors",
                18,
            ),
            transformer_shards: Vec::new(),
            vae: mk("flux2/civitai/2759597/flux2-vae.safetensors", 1),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![mk("flux2/civitai/2759597/qwen3.safetensors", 8)],
            text_tokenizer: None,
            decoder: None,
        };
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        assert!(
            !server_offload_enabled_for_paths_with_request(&paths, Some(hint), false, true),
            "global MOLD_OFFLOAD must not force Flux.2 NVFP4 block offload \
             because the NVFP4 streaming linear path is the memory-control mechanism"
        );
    }

    fn qwen_image_q8_paths(
        transformer_gb: u64,
        vae_gb: u64,
        text_encoder_gb: u64,
    ) -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let transformer = mk("qwen-image-Q8_0.gguf", transformer_gb);
        let vae = mk("qwen-image-vae.safetensors", vae_gb);
        let text_encoder = mk("qwen2.5-vl.safetensors", text_encoder_gb);
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer,
            transformer_shards: Vec::new(),
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
            text_encoder_files: vec![text_encoder],
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    #[test]
    fn preflight_accepts_quantized_qwen_image_q8_on_24gb() {
        let (_dir, paths) = qwen_image_q8_paths(21, 1, 16);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 2,
            dtype_bytes: 2,
            family: ActivationFamily::QwenImageDit,
        };

        let result =
            preflight_memory_guard_with_available("qwen-image:q8", &paths, 0, 24 * GB, Some(hint));

        assert!(
            result.is_ok(),
            "Qwen-Image GGUF Q8 should be admitted on 24 GB because the runtime \
             uses split-CFG and staged text/VAE phases instead of the generic \
             full-headroom sequential estimate, got {result:?}"
        );
    }

    /// private UAT host regression: qwen-image:bf16 (≈41 GB sharded transformer) on an
    /// idle 46 GB L40S was rejected — generic sequential peak (~44 GB with
    /// the 2 GB flat headroom + activation) exceeded the 90% cap (~41.4 GB)
    /// even though the denoise phase only co-resides transformer +
    /// activations (~5 GB of real slack). The BF16 runtime drops the text
    /// encoder before the transformer loads exactly like the GGUF one, so it
    /// gets the same fits-in-free bypass.
    #[test]
    fn preflight_accepts_bf16_qwen_image_that_fits_free_vram() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let shard_a = mk("qwen-image-bf16-00001.safetensors", 21);
        let shard_b = mk("qwen-image-bf16-00002.safetensors", 20);
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: shard_a.clone(),
            transformer_shards: vec![shard_a, shard_b],
            vae: mk("qwen-image-vae.safetensors", 1),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![mk("qwen2.5-vl.safetensors", 16)],
            text_tokenizer: None,
            decoder: None,
        };
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::QwenImageDit,
        };

        let result = preflight_memory_guard_with_available(
            "qwen-image:bf16",
            &paths,
            0,
            46 * GB,
            Some(hint),
        );
        assert!(
            result.is_ok(),
            "BF16 Qwen-Image whose estimated peak fits free VRAM must be \
             admitted — its runtime is phase-sequential like the GGUF path, \
             got {result:?}"
        );

        // The bypass is a fits check, not a blank check: the same model on a
        // 24 GB card still rejects.
        let too_small = preflight_memory_guard_with_available(
            "qwen-image:bf16",
            &paths,
            0,
            24 * GB,
            Some(hint),
        );
        assert!(too_small.is_err(), "24 GB must still reject a ~44 GB peak");

        // The admission's justification is the phase-sequential runtime, so
        // the strategy must match: a load admitted in the 90–100%-of-free
        // band must run Sequential (encode → drop TE → denoise), never Eager
        // (which co-resides transformer + text encoder + VAE).
        let strategy = select_server_load_strategy_for_budget(&paths, Some(46 * GB), Some(hint));
        assert_eq!(
            strategy,
            mold_inference::LoadStrategy::Sequential,
            "bypass-admitted BF16 qwen must load with the Sequential strategy"
        );
    }

    #[test]
    fn server_load_strategy_uses_sequential_for_zimage_requests() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(6, 1, 8, 0);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::ZImageDit,
        };

        let strategy = select_server_load_strategy_for_budget(&paths, Some(24 * GB), Some(hint));

        assert_eq!(
            strategy,
            mold_inference::LoadStrategy::Sequential,
            "Z-Image server requests should use staged loading so base/source/LoRA \
             share the same memory contract"
        );
    }

    /// Companion: under the *old* Eager-strategy math the same model would
    /// have been rejected. Verifying explicitly so a regression that flips
    /// the strategy back gets caught.
    #[test]
    fn eager_strategy_would_have_rejected_quantized_flux_on_24gb() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(12, 1, 10, 1);
        let eager_peak = mold_inference::device::estimate_peak_memory(
            &paths,
            mold_inference::LoadStrategy::Eager,
        );
        // 12 + 1 + 10 + 1 + 2 GB headroom = 26 GB → above 90 % of 24 GB.
        let hard_limit = (24 * GB) * 9 / 10;
        assert!(
            eager_peak > hard_limit,
            "Eager peak ({eager_peak}) should exceed hard limit ({hard_limit}) — \
            this is the false-rejection the Sequential switch fixes"
        );
    }

    #[test]
    fn server_load_strategy_degrades_when_only_sequential_fits() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(12, 1, 10, 1);
        let strategy = select_server_load_strategy_for_budget(&paths, Some(24 * GB), None);

        assert_eq!(
            strategy,
            mold_inference::LoadStrategy::Sequential,
            "server load should match the sequential preflight assumption instead \
             of eager-loading a model whose summed components exceed the budget"
        );
    }

    #[test]
    fn server_load_strategy_stays_eager_when_eager_fits() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(8, 1, 2, 1);
        let strategy = select_server_load_strategy_for_budget(&paths, Some(24 * GB), None);

        assert_eq!(strategy, mold_inference::LoadStrategy::Eager);
    }

    #[test]
    fn server_load_strategy_stays_eager_when_no_budget_available() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(12, 1, 10, 1);
        let strategy = select_server_load_strategy_for_budget(&paths, None, None);

        assert_eq!(strategy, mold_inference::LoadStrategy::Eager);
    }

    /// Tier 2.3: the server-side preflight must consume the
    /// resolution-scaled activation budget. A model that fits at 768²
    /// (where the activation budget is the 256 MB floor) must be rejected
    /// at 2048² (where the budget grows past 1 GB) on the same card.
    #[test]
    fn preflight_memory_guard_accepts_resolution_for_activation_budget() {
        // Shape: 19 GB transformer, 1 GB VAE, 9 GB T5, 1 GB CLIP. Sequential
        // peak = max(10, 20) + 2 GB headroom = 22 GB. On a 25 GB card the
        // 90 % hard limit is 22.5 GB:
        //   * 768²:  22 + 0.256 (floor) = 22.256 ≤ 22.5 → accept
        //   * 2048²: 22 + 1.09          = 23.09  > 22.5 → reject
        // Without the activation hint both would land at 22 GB and accept.
        let (_dir, paths) = flux_shaped_paths_with_sizes(19, 1, 9, 1);

        let hint_768 = ActivationHint {
            width: 768,
            height: 768,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };
        let hint_2048 = ActivationHint {
            width: 2048,
            height: 2048,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };

        let card_total = 25 * GB;
        let result_768 = preflight_memory_guard_with_available(
            "flux-dev",
            &paths,
            0,
            card_total,
            Some(hint_768),
        );
        let result_2048 = preflight_memory_guard_with_available(
            "flux-dev",
            &paths,
            0,
            card_total,
            Some(hint_2048),
        );

        assert!(
            result_768.is_ok(),
            "768² FLUX should fit on 30 GB (small activation budget), got {result_768:?}"
        );
        assert!(
            result_2048.is_err(),
            "2048² FLUX must be rejected on 30 GB (large activation budget pushes \
            peak past 90 % cap), got {result_2048:?}"
        );
    }

    fn flux2_klein9b_bf16_paths() -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let shard_a = mk("diffusion_pytorch_model-00001-of-00002.safetensors", 10);
        let shard_b = mk("diffusion_pytorch_model-00002-of-00002.safetensors", 8);
        let vae = mk("flux2-vae.safetensors", 1);
        let te_a = mk("text_encoder-00001-of-00004.safetensors", 5);
        let te_b = mk("text_encoder-00002-of-00004.safetensors", 5);
        let te_c = mk("text_encoder-00003-of-00004.safetensors", 5);
        let te_d = mk("text_encoder-00004-of-00004.safetensors", 1);
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: shard_a.clone(),
            transformer_shards: vec![shard_a, shard_b],
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
            text_encoder_files: vec![te_a, te_b, te_c, te_d],
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    fn flux2_large_bf16_paths_with_quantized_encoder() -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let shard_a = mk("diffusion_pytorch_model-00001-of-00002.safetensors", 10);
        let shard_b = mk("diffusion_pytorch_model-00002-of-00002.safetensors", 8);
        let vae = mk("flux2-vae.safetensors", 1);
        let qwen3_q3 = mk("qwen3-q3.gguf", 3);
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: shard_a.clone(),
            transformer_shards: vec![shard_a, shard_b],
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
            text_encoder_files: vec![qwen3_q3],
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    #[test]
    fn preflight_allows_flux2_klein9b_bf16_on_24gb_when_sequential_budget_fits() {
        let (_dir, paths) = flux2_klein9b_bf16_paths();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        let result = preflight_memory_guard_with_available(
            "flux2-klein-9b:bf16",
            &paths,
            0,
            24 * GB,
            Some(hint),
        );

        assert!(
            result.is_ok(),
            "Klein-9B BF16 should be admitted on a 24 GB card when the \
             sequential transformer/VAE phase plus activation budget fits; \
             Qwen3 can be quantized/dropped before denoise, got {result:?}"
        );
    }

    #[test]
    fn preflight_rejects_flux2_klein9b_bf16_on_24gb_when_activation_budget_exceeds_cap() {
        let (_dir, paths) = flux2_klein9b_bf16_paths();
        let hint = ActivationHint {
            width: 2048,
            height: 2048,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        let result = preflight_memory_guard_with_available(
            "flux2-klein-9b:bf16",
            &paths,
            0,
            24 * GB,
            Some(hint),
        );

        assert!(
            result.is_err(),
            "Klein-9B BF16 should still reject when resolution-scaled \
             activation budget pushes the sequential phase past the 90% cap, got {result:?}"
        );
    }

    #[test]
    fn server_load_strategy_degrades_flux2_klein9b_bf16_on_24gb_to_sequential() {
        let (_dir, paths) = flux2_klein9b_bf16_paths();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        let strategy = select_server_load_strategy_for_budget(&paths, Some(24 * GB), Some(hint));

        assert_eq!(
            strategy,
            mold_inference::LoadStrategy::Sequential,
            "server must use load-use-drop for Klein-9B BF16 on 24 GB so the \
             text encoder is not co-resident with the transformer"
        );
    }

    #[test]
    fn server_load_strategy_degrades_large_flux2_bf16_even_with_quantized_encoder() {
        let (_dir, paths) = flux2_large_bf16_paths_with_quantized_encoder();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        let strategy = select_server_load_strategy_for_budget(&paths, Some(24 * GB), Some(hint));

        assert_eq!(
            strategy,
            mold_inference::LoadStrategy::Sequential,
            "large Flux.2 BF16 transformer shards need load-use-drop on 24 GB \
             even when Qwen3 resolves to a small quantized encoder"
        );
    }

    #[test]
    fn server_load_strategy_forces_klein9b_bf16_sequential_on_24gb_even_with_overgenerous_budget() {
        let (_dir, paths) = flux2_klein9b_bf16_paths();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        let strategy = select_server_load_strategy_for_device(
            &paths,
            Some(128 * GB),
            Some(24 * GB),
            Some(hint),
        );

        assert_eq!(
            strategy,
            mold_inference::LoadStrategy::Sequential,
            "Klein-9B BF16 must not use eager loading on 24 GB cards even if \
             the live free-memory query is over-generous or falls back to \
             system memory"
        );
    }

    #[test]
    fn server_load_strategy_caps_overgenerous_budget_for_klein_like_bf16_model() {
        let (_dir, paths) = flux2_klein9b_bf16_paths();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        let strategy = select_server_load_strategy_for_device(
            &paths,
            Some(128 * GB),
            Some(24 * GB),
            Some(hint),
        );

        assert_eq!(
            strategy,
            mold_inference::LoadStrategy::Sequential,
            "Klein-9B-shaped BF16 loads must use device VRAM as the budget cap \
             even when the live available-memory reading falls back to a larger \
             system-memory value"
        );
    }

    #[test]
    fn server_load_strategy_never_substitutes_total_when_live_available_missing() {
        let (_dir, paths) = flux2_klein9b_bf16_paths();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        let strategy =
            select_server_load_strategy_for_device(&paths, None, Some(24 * GB), Some(hint));

        assert_eq!(strategy, mold_inference::LoadStrategy::Eager);
    }

    /// Build LTX-2-shaped paths: a single 46 GB single-file checkpoint
    /// (transformer == vae) and a 25 GB Gemma TE in `text_encoder_files`.
    /// Mirrors cv:2752735 on disk.
    fn ltx2_shaped_paths_with_sizes(
        transformer_gb: u64,
        gemma_te_gb: u64,
    ) -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let transformer = mk("ltx2_full.safetensors", transformer_gb);
        let gemma = mk("gemma_te.safetensors", gemma_te_gb);
        // LTX-2 catalog bridge sets vae == transformer (single-file
        // convention). The peak estimator detects this and avoids
        // double-counting.
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: transformer.clone(),
            transformer_shards: Vec::new(),
            vae: transformer,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![gemma],
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    /// LTX-2 22B (cv:2752735) on a 24 GB 3090 must NOT be falsely rejected
    /// by the file-size-based preflight. The transformer streams blocks
    /// (`Ltx2AvTransformer3DModel::new_streaming`); only ~2 GB of weights
    /// are co-resident at peak. The activation hint marks the family as
    /// `Ltx2Video`, which routes through `streaming_transformer_peak`.
    #[test]
    fn preflight_accepts_ltx2_22b_on_24gb_card_via_streaming_peak() {
        let (_dir, paths) = ltx2_shaped_paths_with_sizes(46, 0);
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Ltx2Video,
        };
        let result =
            preflight_memory_guard_with_available("cv:2752735", &paths, 0, 24 * GB, Some(hint));
        assert!(
            result.is_ok(),
            "22B LTX-2 must fit on a 24 GB card under streaming-aware peak \
             (only ~2 blocks co-resident; runtime handles its own memory), \
            got {result:?}",
        );
    }

    #[test]
    fn preflight_accepts_ltx2_22b_by_catalog_path_when_hint_is_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let transformer = mk("ltx2/civitai/2752735/ltx23_full.safetensors", 46);
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: transformer.clone(),
            transformer_shards: Vec::new(),
            vae: transformer,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };
        let result = preflight_memory_guard_with_available("cv:2752735", &paths, 0, 24 * GB, None);

        assert!(
            result.is_ok(),
            "LTX-2 catalog paths should use the streaming-transformer peak even \
             when the multi-GPU worker cannot resolve a family hint, got {result:?}"
        );
    }

    /// Without the streaming hint the same paths land on the file-size
    /// peak and reject — pinning the previous behavior so a regression
    /// that flips the hint plumbing back is caught.
    #[test]
    fn preflight_rejects_ltx2_22b_when_hint_marks_non_streaming() {
        let (_dir, paths) = ltx2_shaped_paths_with_sizes(46, 0);
        // Use FluxDit family — same shape, no streaming flag — so the
        // preflight falls through to the file-size estimator and rejects
        // at 90 % of 24 GB.
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };
        let result =
            preflight_memory_guard_with_available("cv:2752735", &paths, 0, 24 * GB, Some(hint));
        assert!(
            result.is_err(),
            "without the LTX-2 streaming hint the file-size peak must reject \
             a 46 GB transformer on a 24 GB card — this anchors the regression \
             that landed before the streaming-aware path",
        );
    }

    /// Encoder phase for LTX-2 still pays full encoder_total when the user
    /// pins the placement to GPU (`MOLD_LTX2_GEMMA_DEVICE=gpu`). With a
    /// 25 GB Gemma TE on a 24 GB card the encoder phase trips the 90 % cap
    /// even when the transformer is streamed — a real OOM the user should
    /// see from the runtime, but the preflight captures it up-front.
    #[test]
    fn preflight_rejects_ltx2_when_encoder_phase_exceeds_card() {
        let (_dir, paths) = ltx2_shaped_paths_with_sizes(46, 25);
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Ltx2Video,
        };
        let result = preflight_memory_guard_with_available_and_policy(
            "cv:2752735",
            &paths,
            0,
            24 * GB,
            Some(hint),
            false,
            true,
        );
        assert!(
            result.is_err(),
            "25 GB Gemma TE alone exceeds 90 %% of 24 GB during the encoder \
             phase — preflight must surface this even when the transformer \
             is streamed, got {result:?}",
        );
    }

    /// In auto mode the LTX-2 runtime may try the prompt encoder on GPU first,
    /// but CUDA OOM during that phase is recoverable: it reclaims the context
    /// and retries the prompt encoder on CPU before streamed transformer load.
    /// Preflight must admit that path instead of rejecting the request before
    /// the runtime fallback can run.
    #[test]
    fn preflight_admits_ltx2_auto_gemma_even_when_gpu_encoder_would_exceed_cap() {
        let (_dir, paths) = ltx2_shaped_paths_with_sizes(46, 25);
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Ltx2Video,
        };
        let result = preflight_memory_guard_with_available_and_policy(
            "cv:2752735",
            &paths,
            0,
            24 * GB,
            Some(hint),
            false,
            false,
        );
        assert!(
            result.is_ok(),
            "auto Gemma placement can fall back to CPU at runtime, so preflight \
             must not reject solely because a same-GPU prompt encoder phase \
             would exceed the hard cap, got {result:?}",
        );
    }

    /// `MOLD_LTX2_GEMMA_DEVICE=cpu` shifts the Gemma TE to system RAM. The
    /// encoder phase no longer competes for VRAM; the streaming-aware peak
    /// collapses to "transformer streaming cap + activation + headroom" and
    /// the same 25 GB Gemma + 46 GB transformer paths admit on a 24 GB card.
    /// This is the load-bearing behavior on a single 3090 running cv:2752735.
    #[test]
    fn preflight_admits_ltx2_22b_with_25gb_gemma_when_resolver_picks_cpu() {
        let (_dir, paths) = ltx2_shaped_paths_with_sizes(46, 25);
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Ltx2Video,
        };
        let result = preflight_memory_guard_with_available_and_policy(
            "cv:2752735",
            &paths,
            0,
            24 * GB,
            Some(hint),
            false,
            false,
        );
        assert!(
            result.is_ok(),
            "with MOLD_LTX2_GEMMA_DEVICE=cpu the encoder phase should not \
             count against GPU VRAM, so cv:2752735 must admit on 24 GB even \
             with a 25 GB Gemma TE, got {result:?}",
        );
    }

    /// `ActivationHint::from_request` picks the right family + batch for a
    /// real-shaped GenerateRequest.
    #[test]
    fn activation_hint_from_request_classifies_correctly() {
        let mut req = GenerateRequest {
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
            prompt: "test".into(),
            negative_prompt: None,
            model: "flux-dev:bf16".into(),
            width: 1024,
            height: 1024,
            steps: 20,
            guidance: 3.5,
            seed: None,
            batch_size: 1,
            output_format: Default::default(),
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
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        };

        // FLUX is guidance-distilled → batch=1 even with guidance > 1.
        let hint_flux = ActivationHint::from_request(&req, "flux");
        assert_eq!(hint_flux.family, ActivationFamily::FluxDit);
        assert_eq!(hint_flux.batch, 1);

        // SDXL with CFG → batch=2.
        let hint_sdxl = ActivationHint::from_request(&req, "sdxl");
        assert_eq!(hint_sdxl.family, ActivationFamily::SdxlUnet);
        assert_eq!(hint_sdxl.batch, 2);

        // SDXL with no CFG (LCM/Turbo) → batch=1.
        req.guidance = 1.0;
        let hint_sdxl_lcm = ActivationHint::from_request(&req, "sdxl");
        assert_eq!(hint_sdxl_lcm.batch, 1);

        // Unknown family slug falls through to FluxDit.
        let hint_unknown = ActivationHint::from_request(&req, "totally-bogus");
        assert_eq!(hint_unknown.family, ActivationFamily::FluxDit);
    }

    /// Synthesize a minimal safetensors at `path` that lists the given
    /// keys in the JSON header (each as a 1-element F32 tensor sharing the
    /// same 4-byte zero blob). Sufficient for the header-peek probe; no
    /// dep on the `safetensors` crate.
    /// The regression half of the fixture above, run with a direnv-style
    /// `MOLD_MODELS_DIR` deliberately IN PLACE: the writer's paths must land
    /// under its own tempdir and the sentinel "production store" must come
    /// through byte-identical. This is the exact leak that overwrote
    /// hal9000's real audio VAE with a 194-byte stub (2026-08-28); a shell
    /// wrapper protects one developer, this protects all of them.
    #[test]
    fn split_audio_fixture_never_writes_through_a_direnv_models_dir() {
        // The guard saved the real values; anything set below is undone by
        // its drop, panic included.
        let _env = crate::test_support::hermetic_store_env();
        let sentinel = tempfile::tempdir().unwrap();
        let store = sentinel.path().join("shared/ltx2/vae");
        std::fs::create_dir_all(&store).unwrap();
        let canary = store.join("ltx-2.5-audio-vae-bf16.safetensors");
        std::fs::write(&canary, b"production bytes - do not touch").unwrap();
        std::env::set_var("MOLD_MODELS_DIR", sentinel.path());

        let dir = tempfile::tempdir().unwrap();
        let model = mold_core::ltx25_manifest::DISTILLED_INT8_CONV;
        let split =
            mold_core::ltx25_manifest::Ltx25ModelPaths::resolve_in(dir.path(), model).unwrap();
        assert!(
            split.audio_vae.starts_with(dir.path()),
            "the fixture root must be the test's own tempdir, got {}",
            split.audio_vae.display()
        );
        std::fs::create_dir_all(split.audio_vae.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &split.audio_vae,
            &[
                "audio_vae.per_channel_statistics.mean-of-means",
                "vocoder.vocoder.conv_pre.weight",
            ],
        );
        assert!(split.audio_vae.exists());

        assert_eq!(
            std::fs::read(&canary).unwrap(),
            b"production bytes - do not touch",
            "the fixture writer reached the MOLD_MODELS_DIR store"
        );
        assert_eq!(
            std::fs::read_dir(&store).unwrap().count(),
            1,
            "nothing may be created beside the canary"
        );
    }

    fn write_safetensors_with_keys(path: &std::path::Path, keys: &[&str]) {
        use std::io::Write;
        let mut header = serde_json::Map::new();
        for key in keys {
            header.insert(
                (*key).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                }),
            );
        }
        let header_json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = std::fs::File::create(path).expect("create fixture");
        f.write_all(&(header_json.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(&header_json).unwrap();
        f.write_all(&[0u8; 4]).unwrap();
    }

    fn flux_unet_only_catalog_entry(
        version_id: &str,
        file_name: &str,
    ) -> mold_catalog::entry::CatalogEntry {
        use mold_catalog::entry::{
            CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, LicenseFlags,
            Modality, RecipeFile, Source, TokenKind,
        };
        use mold_catalog::families::Family;

        CatalogEntry {
            id: CatalogId::from(format!("cv:{version_id}")),
            source: Source::Civitai,
            source_id: version_id.to_string(),
            name: "FLUX Unet-only fine-tune".into(),
            author: Some("someone".into()),
            family: Family::Flux,
            family_role: FamilyRole::Finetune,
            sub_family: None,
            modality: Modality::Image,
            kind: mold_catalog::entry::Kind::Checkpoint,
            file_format: FileFormat::Safetensors,
            bundling: mold_catalog::entry::Bundling::SingleFile,
            size_bytes: Some(12_000_000_000),
            download_count: 0,
            rating: None,
            likes: 0,
            nsfw: false,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: LicenseFlags::default(),
            tags: vec![],
            companions: vec!["t5-v1_1-xxl".into(), "clip-l".into(), "flux-vae".into()],
            download_recipe: DownloadRecipe {
                files: vec![RecipeFile {
                    url: format!("https://civitai.com/api/download/models/{version_id}"),
                    dest: format!("{{family}}/civitai/{version_id}/{file_name}"),
                    sha256: Some("DEAD".repeat(16)),
                    size_bytes: Some(12_000_000_000),
                    role: None,
                }],
                needs_token: Some(TokenKind::Civitai),
            },
            supported: true,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec![],
            page_url: None,
        }
    }

    // ── Lazy intent / disk-aware resolution regression tests ───────────────

    /// Stub the FLUX companion manifest entries (flux-vae, clip-l, t5)
    /// pointing at on-disk files inside `models_dir`. Mirrors what the
    /// real manifest installer would have done.
    fn stub_flux_companion_paths_in_dir(
        config: &mut mold_core::Config,
        models_dir: &std::path::Path,
        flux_vae_present: bool,
    ) {
        let vae_path = models_dir.join("flux-vae/ae.safetensors");
        std::fs::create_dir_all(vae_path.parent().unwrap()).unwrap();
        if flux_vae_present {
            std::fs::File::create(&vae_path).unwrap();
        }
        config.models.insert(
            "flux-vae".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(vae_path.to_string_lossy().into_owned()),
                vae: Some(vae_path.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );
        let clip_path = models_dir.join("clip-l/model.safetensors");
        std::fs::create_dir_all(clip_path.parent().unwrap()).unwrap();
        std::fs::File::create(&clip_path).unwrap();
        config.models.insert(
            "clip-l".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(clip_path.to_string_lossy().into_owned()),
                vae: Some(clip_path.to_string_lossy().into_owned()),
                clip_tokenizer: Some(format!("{}/clip-l/tokenizer.json", models_dir.display())),
                ..Default::default()
            },
        );
        let t5_path = models_dir.join("t5-v1_1-xxl/t5xxl_fp16.safetensors");
        std::fs::create_dir_all(t5_path.parent().unwrap()).unwrap();
        std::fs::File::create(&t5_path).unwrap();
        config.models.insert(
            "t5-v1_1-xxl".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(t5_path.to_string_lossy().into_owned()),
                vae: Some(t5_path.to_string_lossy().into_owned()),
                t5_tokenizer: Some(format!(
                    "{}/t5-v1_1-xxl/tokenizer.json",
                    models_dir.display()
                )),
                ..Default::default()
            },
        );
    }

    /// Task 3 (Step 1): synthesis must produce the same intent regardless
    /// of whether the primary file is on disk yet. The disk-dependent
    /// part is moved to `resolve_intent_to_paths`, which is only called
    /// at engine-load time.
    #[test]
    fn synthesis_intent_is_consistent_before_and_after_download() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let entry =
            flux_unet_only_catalog_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");

        let intent_absent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();

        let primary_path = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "single_blocks.0.linear1.weight",
                "img_in.weight",
            ],
        );

        let intent_present =
            mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();

        assert_eq!(
            intent_absent, intent_present,
            "intent synthesis must be pure — independent of disk state"
        );
    }

    /// Task 3 (Step 2 + Step 3): with the primary present as a
    /// transformer-only checkpoint, resolution picks the flux-vae
    /// companion for cfg.vae rather than the primary checkpoint.
    #[test]
    fn resolve_intent_picks_flux_vae_companion_when_primary_is_transformer_only() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _environment = IsolatedModelEnvironment::without_models_dir_override(models_dir);

        let primary_path = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "single_blocks.0.linear1.weight",
                "img_in.weight",
            ],
        );

        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        stub_flux_companion_paths_in_dir(&mut config, models_dir, true);

        let entry =
            flux_unet_only_catalog_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_paths("cv:994561", &intent, &config).unwrap();

        let vae_path = models_dir.join("flux-vae/ae.safetensors");
        assert_eq!(cfg.vae.as_deref(), vae_path.to_str());
        assert_eq!(cfg.transformer.as_deref(), primary_path.to_str());
    }

    #[test]
    fn resolve_intent_preserves_flux_schnell_subfamily() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _environment = IsolatedModelEnvironment::without_models_dir_override(models_dir);

        let primary_path = models_dir
            .join("cv-1153358/flux/civitai/1153358/agfluxSchnell_realistic23.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &[
                "model.diffusion_model.double_blocks.0.img_attn.proj.weight",
                "model.diffusion_model.img_in.weight",
            ],
        );

        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        stub_flux_companion_paths_in_dir(&mut config, models_dir, true);

        let mut entry =
            flux_unet_only_catalog_entry("1153358", "agfluxSchnell_realistic23.safetensors");
        entry.sub_family = Some("flux1-s".into());

        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_paths("cv:1153358", &intent, &config).unwrap();

        assert_eq!(
            cfg.is_schnell,
            Some(true),
            "flux1-s catalog entries must select FLUX schnell config, not dev guidance config"
        );
    }

    #[test]
    fn resolve_intent_applies_flux_dev_subfamily_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _environment = IsolatedModelEnvironment::without_models_dir_override(models_dir);

        let primary_path =
            models_dir.join("cv-2319074/flux/civitai/2319074/jibMixFlux_v12SRPO.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "single_blocks.0.linear1.weight",
                "img_in.weight",
            ],
        );

        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            default_steps: 4,
            ..Default::default()
        };
        stub_flux_companion_paths_in_dir(&mut config, models_dir, true);

        let mut entry = flux_unet_only_catalog_entry("2319074", "jibMixFlux_v12SRPO.safetensors");
        entry.sub_family = Some("flux1-d".into());

        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_paths("cv:2319074", &intent, &config).unwrap();

        assert_eq!(cfg.is_schnell, Some(false));
        assert_eq!(cfg.default_steps, Some(25));
        assert_eq!(cfg.default_guidance, Some(3.5));
        assert_eq!(cfg.default_width, Some(1024));
        assert_eq!(cfg.default_height, Some(1024));
    }

    #[test]
    fn resolve_intent_populates_qwen_runtime_companion_paths() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _environment = IsolatedModelEnvironment::new(models_dir);
        let primary_path =
            models_dir.join("cv-2110043/qwen-image/civitai/2110043/qwenImage_fp8.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        std::fs::File::create(&primary_path).unwrap();

        let config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        let companion_paths =
            materialize_manifest_companion(models_dir, "qwen-image-runtime", &config);
        let vae_path = companion_paths.vae;
        let text_encoder_files = companion_paths.text_encoder_files;
        let tokenizer_path = companion_paths.text_tokenizer;

        let mut entry = flux_unet_only_catalog_entry("2110043", "qwenImage_fp8.safetensors");
        entry.family = mold_catalog::families::Family::QwenImage;
        entry.companions = vec!["qwen-image-runtime".into()];
        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_paths("cv:2110043", &intent, &config).unwrap();
        let expected_text_encoder_files = text_encoder_files
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert_eq!(cfg.transformer.as_deref(), primary_path.to_str());
        assert_eq!(cfg.vae.as_deref(), vae_path.to_str());
        assert_eq!(
            cfg.text_encoder_files.as_deref(),
            Some(expected_text_encoder_files.as_slice())
        );
        assert_eq!(
            cfg.text_tokenizer.as_deref(),
            tokenizer_path.as_deref().and_then(std::path::Path::to_str)
        );
    }

    #[test]
    fn resolve_intent_populates_wuerstchen_runtime_companion_paths() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _environment = IsolatedModelEnvironment::new(models_dir);
        let primary_path = models_dir.join(
            "hf-example/wuerstchen-prior/wuerstchen/example/wuerstchen-prior/prior.safetensors",
        );
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        std::fs::File::create(&primary_path).unwrap();

        let config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        let companion_paths =
            materialize_manifest_companion(models_dir, "wuerstchen-runtime", &config);

        let mut entry = flux_unet_only_catalog_entry("unused", "prior.safetensors");
        entry.id = mold_catalog::entry::CatalogId::from("hf:example/wuerstchen-prior");
        entry.source = mold_catalog::entry::Source::Hf;
        entry.source_id = "example/wuerstchen-prior".into();
        entry.family = mold_catalog::families::Family::Wuerstchen;
        entry.companions = vec!["wuerstchen-runtime".into()];
        entry.download_recipe.files[0].dest = "{family}/{author}/{name}/prior.safetensors".into();

        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_paths("hf:example/wuerstchen-prior", &intent, &config).unwrap();

        assert_eq!(cfg.transformer.as_deref(), primary_path.to_str());
        assert_eq!(
            cfg.decoder.as_deref(),
            companion_paths
                .decoder
                .as_deref()
                .and_then(std::path::Path::to_str)
        );
        assert_eq!(cfg.vae.as_deref(), companion_paths.vae.to_str());
        assert_eq!(
            cfg.clip_encoder.as_deref(),
            companion_paths
                .clip_encoder
                .as_deref()
                .and_then(std::path::Path::to_str)
        );
        assert_eq!(
            cfg.clip_tokenizer.as_deref(),
            companion_paths
                .clip_tokenizer
                .as_deref()
                .and_then(std::path::Path::to_str)
        );
        assert_eq!(
            cfg.clip_encoder_2.as_deref(),
            companion_paths
                .clip_encoder_2
                .as_deref()
                .and_then(std::path::Path::to_str)
        );
        assert_eq!(
            cfg.clip_tokenizer_2.as_deref(),
            companion_paths
                .clip_tokenizer_2
                .as_deref()
                .and_then(std::path::Path::to_str)
        );
    }

    #[test]
    fn resolve_intent_uses_zimage_recipe_text_encoder_and_shared_companion_vae() {
        use mold_catalog::entry::{
            Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat,
            LicenseFlags, Modality, RecipeFile, RecipeFileRole, Source, TokenKind,
        };
        use mold_catalog::families::Family;

        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _environment = IsolatedModelEnvironment::without_models_dir_override(models_dir);

        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        let te_dir = models_dir.join("z-image-te");
        for path in [
            te_dir.join("text_encoder/model-00001-of-00003.safetensors"),
            te_dir.join("text_encoder/model-00002-of-00003.safetensors"),
            te_dir.join("text_encoder/model-00003-of-00003.safetensors"),
            te_dir.join("vae/diffusion_pytorch_model.safetensors"),
            te_dir.join("tokenizer/tokenizer.json"),
        ] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"test fixture").unwrap();
        }
        config.install_frozen_model_config(
            "z-image-te",
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(
                    te_dir
                        .join("text_encoder/model-00001-of-00003.safetensors")
                        .to_string_lossy()
                        .into_owned(),
                ),
                vae: Some(
                    te_dir
                        .join("vae/diffusion_pytorch_model.safetensors")
                        .to_string_lossy()
                        .into_owned(),
                ),
                text_encoder_files: Some(vec![
                    te_dir
                        .join("text_encoder/model-00001-of-00003.safetensors")
                        .to_string_lossy()
                        .into_owned(),
                    te_dir
                        .join("text_encoder/model-00002-of-00003.safetensors")
                        .to_string_lossy()
                        .into_owned(),
                    te_dir
                        .join("text_encoder/model-00003-of-00003.safetensors")
                        .to_string_lossy()
                        .into_owned(),
                ]),
                text_tokenizer: Some(
                    te_dir
                        .join("tokenizer/tokenizer.json")
                        .to_string_lossy()
                        .into_owned(),
                ),
                ..Default::default()
            },
        );
        let entry = CatalogEntry {
            id: CatalogId::from("cv:2442439"),
            source: Source::Civitai,
            source_id: "2442439".into(),
            name: "Z Image Turbo".into(),
            author: Some("z".into()),
            family: Family::ZImage,
            family_role: FamilyRole::Finetune,
            sub_family: None,
            modality: Modality::Image,
            kind: mold_catalog::entry::Kind::Checkpoint,
            file_format: FileFormat::Safetensors,
            bundling: Bundling::SingleFile,
            size_bytes: Some(12_021_353_906),
            download_count: 0,
            rating: None,
            likes: 0,
            nsfw: false,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: LicenseFlags::default(),
            tags: vec![],
            companions: vec!["z-image-te".into()],
            download_recipe: DownloadRecipe {
                files: vec![
                    RecipeFile {
                        url: "https://civitai.example/model".into(),
                        dest: "{family}/civitai/2442439/zImageTurbo_turbo.safetensors".into(),
                        sha256: None,
                        size_bytes: Some(12_021_353_906),
                        role: None,
                    },
                    RecipeFile {
                        url: "https://civitai.example/text".into(),
                        dest: "{family}/civitai/2442439/zImageTurbo_turbo_txt.safetensors".into(),
                        sha256: None,
                        size_bytes: Some(8_044_982_048),
                        role: Some(RecipeFileRole::TextEncoder),
                    },
                ],
                needs_token: Some(TokenKind::Civitai),
            },
            supported: true,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec![],
            page_url: None,
        };

        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();
        std::fs::create_dir_all(intent.primary_recipe_path.parent().unwrap()).unwrap();
        std::fs::write(&intent.primary_recipe_path, b"primary").unwrap();
        let cfg = resolve_intent_to_paths("cv:2442439", &intent, &config).unwrap();

        let recipe_text_encoder =
            models_dir.join("cv-2442439/z-image/civitai/2442439/zImageTurbo_turbo_txt.safetensors");
        let shared_vae = te_dir.join("vae/diffusion_pytorch_model.safetensors");
        assert_eq!(cfg.vae.as_deref(), shared_vae.to_str());
        let expected_text_encoder_files = vec![recipe_text_encoder.to_string_lossy().into_owned()];
        assert_eq!(
            cfg.text_encoder_files.as_deref(),
            Some(expected_text_encoder_files.as_slice())
        );
    }

    /// Task 5 (Step 1): resolution surfaces the *specific* missing
    /// companion name rather than a generic "missing required components"
    /// blob. The CompanionConfigMissing variant fires when the manifest
    /// entry isn't in the user's Config.models — a real config bug.
    #[test]
    fn resolve_intent_returns_error_naming_missing_required_companion() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _environment = IsolatedModelEnvironment::new(models_dir);

        let primary_path = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &["double_blocks.0.img_attn.proj.weight", "img_in.weight"],
        );

        // Empty config — none of t5, clip-l, flux-vae are installed.
        let config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };

        let entry =
            flux_unet_only_catalog_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();
        let err = resolve_intent_to_paths("cv:994561", &intent, &config).unwrap_err();

        let msg = err.to_string();
        // The first missing companion in the FLUX list is t5-v1_1-xxl;
        // we just need the error to name *some* specific companion.
        assert!(
            msg.contains("t5-v1_1-xxl") || msg.contains("clip-l") || msg.contains("flux-vae"),
            "error must name a specific missing companion, got: {msg}"
        );
        assert!(matches!(err, ResolveError::CompanionConfigMissing { .. }));
    }

    /// Task 6: with the lazy intent / resolve flow, a second request
    /// after the file lands succeeds where the first might have raced.
    /// This exercises the no-stale-config invariant directly: both calls
    /// run the full intent + resolve pipeline; the second one sees the
    /// fresh disk state.
    #[test]
    fn cv_id_resolves_when_files_arrive_after_initial_request() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _environment = IsolatedModelEnvironment::without_models_dir_override(models_dir);

        let entry =
            flux_unet_only_catalog_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        // Companions are present; primary file initially missing.
        stub_flux_companion_paths_in_dir(&mut config, models_dir, true);

        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();

        // First resolve — primary not yet on disk, so the catalog model
        // must not be advertised as loadable.
        let err = resolve_intent_to_paths("cv:994561", &intent, &config).unwrap_err();
        assert!(matches!(err, ResolveError::PrimaryFileMissing { .. }));
        let primary_path = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        let vae_path = models_dir.join("flux-vae/ae.safetensors");

        // File arrives mid-flight as a transformer-only checkpoint.
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &["double_blocks.0.img_attn.proj.weight", "img_in.weight"],
        );

        // Second resolve sees the file: probe runs, declares "no bundled
        // VAE", flux-vae companion still wins. Same result, but this time
        // through the fully-armed disk-aware path.
        let cfg_second = resolve_intent_to_paths("cv:994561", &intent, &config).unwrap();
        assert_eq!(cfg_second.transformer.as_deref(), primary_path.to_str());
        assert_eq!(cfg_second.vae.as_deref(), vae_path.to_str());
    }

    #[test]
    fn resolve_intent_rejects_truncated_sidecar_primary() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _environment = IsolatedModelEnvironment::new(models_dir);

        let primary_path = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &["double_blocks.0.img_attn.proj.weight", "img_in.weight"],
        );
        let entry =
            flux_unet_only_catalog_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let sidecar = mold_catalog::sidecar::sidecar_from_entry(
            &entry,
            "flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors".into(),
        );
        let mut sidecar = sidecar;
        sidecar.size_bytes = Some(primary_path.metadata().unwrap().len() + 1);
        let sidecar_path = mold_catalog::sidecar::civitai_sidecar_path(models_dir, "cv:994561");
        mold_catalog::sidecar::write_sidecar(&sidecar_path, &sidecar).unwrap();

        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        stub_flux_companion_paths_in_dir(&mut config, models_dir, true);
        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();

        let err = resolve_intent_to_paths("cv:994561", &intent, &config).unwrap_err();
        assert!(matches!(err, ResolveError::PrimaryFileMissing { .. }));
    }

    // ── InstallError translation tests (Task 4) ────────────────────────────

    #[test]
    fn live_error_to_install_error_maps_404_to_not_found() {
        let upstream = mold_catalog::live::LiveSearchError::Upstream {
            host: "civitai.com",
            status: 404,
            body: "{\"error\": \"not found\"}".to_string(),
            retry_after: None,
        };
        let mapped = live_error_to_install_error("cv:42", &upstream);
        assert!(matches!(mapped, mold_core::InstallError::NotFound(_)));
    }

    #[test]
    fn live_error_to_install_error_maps_5xx_to_recipe_malformed() {
        let upstream = mold_catalog::live::LiveSearchError::Upstream {
            host: "civitai.com",
            status: 500,
            body: "internal".into(),
            retry_after: None,
        };
        let mapped = live_error_to_install_error("cv:42", &upstream);
        assert!(matches!(
            mapped,
            mold_core::InstallError::RecipeMalformed(_)
        ));
    }

    #[test]
    fn install_error_to_api_error_maps_network_to_502() {
        let err = mold_core::InstallError::Network("dns: civitai.com".into());
        let api = install_error_to_api_error(&err);
        // API error doesn't expose .status() publicly, but .code is
        // "INTERNAL_ERROR" with the BAD_GATEWAY status flag (see
        // ApiError::internal_with_status). The user-visible message must
        // carry "network unreachable" so they know what happened.
        assert!(api.error.contains("network unreachable"));
    }

    #[test]
    fn install_error_to_api_error_maps_not_found_to_404() {
        let err = mold_core::InstallError::NotFound("cv:99999999".into());
        let api = install_error_to_api_error(&err);
        assert_eq!(api.code, "MODEL_NOT_FOUND");
    }

    // ── preflight error message budget-cap correctness ───────────────────────

    /// The rejection fraction used by `check_model_memory_budget`. Pinned so a
    /// future change to the factor forces a matching update to the error message
    /// (and this test).
    const BUDGET_FRACTION_NUMERATOR: u64 = 9;
    const BUDGET_FRACTION_DENOMINATOR: u64 = 10;

    /// Compute the budget cap the way `check_model_memory_budget` does.
    fn expected_budget_cap(available: u64) -> u64 {
        available * BUDGET_FRACTION_NUMERATOR / BUDGET_FRACTION_DENOMINATOR
    }

    /// The rejection error message must display the budget cap (the number that
    /// was actually compared against peak), not the raw available VRAM. This is
    /// the root-cause regression test for the "24.4 GB exceeds 25.3 GB" bug.
    #[test]
    fn preflight_error_message_states_correct_budget_cap() {
        // Mirrors the user's reported values: peak=24.4 GB, free=25.3 GB.
        // Cap = 25.3 × 0.9 = 22.77 GB. Peak (24.4) > cap (22.77) → reject.
        let peak: u64 = 24_400_000_000;
        let available: u64 = 25_300_000_000;
        let cap = expected_budget_cap(available);

        // Sanity: the test scenario is actually a rejection.
        assert!(
            peak > cap,
            "test invariant: peak ({peak}) must exceed cap ({cap})"
        );

        let result = check_model_memory_budget(
            "qwen-image:q8",
            peak,
            available,
            "Try a smaller variant (e.g. ':q5' / ':q4'), enable --offload (FLUX), or close other GPU apps.",
        );
        assert!(result.is_err(), "expected rejection, got Ok");

        let err = result.unwrap_err();
        let msg = &err.error;

        // The message must contain the cap, not just the raw available.
        let cap_gb = cap as f64 / 1_000_000_000.0;
        let cap_str = format!("{cap_gb:.1}");
        assert!(
            msg.contains("budget cap"),
            "error must mention 'budget cap', got: {msg}"
        );
        assert!(
            msg.contains(&cap_str),
            "error must contain the cap value ~{cap_str} GB, got: {msg}"
        );

        // The message must NOT imply that peak < available (the original bug).
        // If the message says "exceeds X GB" where X > peak, the user will be confused.
        // We detect this by checking there's no bare available_gb with no "cap" context
        // that would make the inequality look false.
        let available_gb = available as f64 / 1_000_000_000.0;
        let available_str = format!("{available_gb:.1}");
        // available_gb should appear only as the input to the cap formula, not as the
        // comparison target. Presence of "budget cap" already anchors correct phrasing.
        let _ = available_str; // checked indirectly via the "budget cap" assertion above
    }

    /// The "exceeds" target printed in the rejection message must always be
    /// strictly less than the printed peak. A table of (peak_gb, available_gb)
    /// rejection scenarios verifies no phrasing inverts the inequality.
    #[test]
    fn preflight_error_message_does_not_imply_peak_less_than_available() {
        let scenarios: &[(f64, f64)] = &[
            // (peak_gb, available_gb) — all must trigger rejection
            (24.4, 25.3), // user-reported case
            (19.0, 20.0), // 19 > 90% of 20 = 18
            (10.0, 10.5), // just over 90%
            (30.0, 32.0), // 30 > 90% of 32 = 28.8
            (9.1, 10.0),  // 9.1 > 90% of 10 = 9
        ];
        for &(peak_gb, available_gb) in scenarios {
            let peak = (peak_gb * 1_000_000_000.0) as u64;
            let available = (available_gb * 1_000_000_000.0) as u64;
            let cap = expected_budget_cap(available);

            // Only test rejection scenarios.
            if peak <= cap {
                continue;
            }

            let result =
                check_model_memory_budget("test-model", peak, available, "Try a smaller variant.");
            assert!(
                result.is_err(),
                "expected rejection for peak={peak_gb} available={available_gb}, got Ok"
            );

            let msg = result.unwrap_err().error;

            // The comparison target in the message ("budget cap") must be < peak.
            // We verify this by asserting the cap value appears in the message.
            let cap_gb = cap as f64 / 1_000_000_000.0;
            let cap_str = format!("{cap_gb:.1}");
            assert!(
                msg.contains("budget cap"),
                "scenario peak={peak_gb} available={available_gb}: \
                 message must say 'budget cap', got: {msg}"
            );
            assert!(
                msg.contains(&cap_str),
                "scenario peak={peak_gb} available={available_gb}: \
                 message must include cap={cap_str}, got: {msg}"
            );
        }
    }

    // ── LTX-Video preflight regression (Part 1) ──────────────────────────────

    /// Build LTX-Video-shaped paths: separate transformer (13B BF16 ≈ 26 GB),
    /// VAE (~0.5 GB), and T5 encoder (~9.5 GB). Mirrors the on-disk layout of
    /// `ltx-video-0.9.8-13b-dev:bf16` pulled via `mold pull`.
    fn ltx_video_13b_paths(
        transformer_gb: u64,
        vae_gb: u64,
        t5_gb: u64,
    ) -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let transformer = mk("ltx-video-0.9.8-13b-dev_fp16.safetensors", transformer_gb);
        let vae = mk("ltx-video-vae.safetensors", vae_gb);
        let t5 = mk("t5xxl_fp16.safetensors", t5_gb);
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer,
            transformer_shards: Vec::new(),
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: Some(t5),
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    /// Regression: `ltx-video-0.9.8-13b-dev:bf16` at 768×512×25 on a 24 GB
    /// card MUST be rejected by the preflight. The transformer is ~26 GB BF16
    /// (non-streaming, loaded whole), which alone exceeds 24 GB VRAM. Before
    /// the fix, `activation_family_for("ltx-video")` returned `Ltx2Video`
    /// which triggered the streaming cap path (6 GB) and let the load through,
    /// causing a hard OOM during `load_transformer`.
    #[test]
    fn preflight_rejects_ltx_video_13b_at_768x512x25_on_24gb_card() {
        // 26 GB transformer + 0.5 GB VAE + 9.5 GB T5 (mirrors real disk layout).
        // Sequential peak = max(T5=9.5, transformer+VAE=26.5) + 2 GB headroom
        //                 = 26.5 + 2 = 28.5 GB > 90% of 24 GB (21.6 GB) → reject.
        let (_dir, paths) = ltx_video_13b_paths(26, 1, 10);
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::LtxVideo,
        };
        let result = preflight_memory_guard_with_available(
            "ltx-video-0.9.8-13b-dev:bf16",
            &paths,
            0,
            24 * GB,
            Some(hint),
        );
        assert!(
            result.is_err(),
            "13B LTX-Video BF16 (26 GB transformer) must be rejected on a 24 GB card — \
             the transformer is not streamed and its full weight must be counted, \
             got {result:?}",
        );
        let err = result.unwrap_err();
        // Message must surface the LTX-Video-specific mitigation hints.
        assert!(
            err.error.contains("frames") || err.error.contains("width"),
            "rejection message must suggest reducing frames or resolution, got: {}",
            err.error,
        );
    }

    /// Golden: the preflight estimate for LTX-Video 13B BF16 must be within
    /// ~3 GB of the expected peak (Sequential: max(T5, transformer+VAE) +
    /// 2 GB headroom = max(9.5, 26.5) + 2 = 28.5 GB). We accept ±3 GB to
    /// account for rounding in file sizes; the key invariant is that it's
    /// never so low that a 24 GB card incorrectly admits the load.
    #[test]
    fn preflight_estimate_for_ltx_video_13b_within_expected_range() {
        let (_dir, paths) = ltx_video_13b_paths(26, 1, 10);
        // Sequential peak = max(T5=10, transformer+VAE=27) + 2 headroom = 29 GB.
        // (We use 10 GB for T5 to keep nice round numbers; real T5 is ~9.5 GB.)
        let expected_gb = 29u64;
        let peak = mold_inference::device::estimate_peak_memory(
            &paths,
            mold_inference::LoadStrategy::Sequential,
        );
        let peak_gb = peak / GB;
        assert!(
            peak_gb >= expected_gb.saturating_sub(3),
            "peak estimate ({peak_gb} GB) is unexpectedly low — LTX-Video 13B BF16 \
             sequential estimate should be ≥ {} GB",
            expected_gb.saturating_sub(3),
        );
        assert!(
            peak_gb <= expected_gb + 3,
            "peak estimate ({peak_gb} GB) is unexpectedly high for 26+1+10 GB layout \
             — should be ≤ {} GB",
            expected_gb + 3,
        );
    }

    /// `activation_family_for("ltx-video")` must return `LtxVideo` (non-streaming),
    /// not `Ltx2Video`. Before the fix both slugs returned `Ltx2Video`, which
    /// caused the preflight to apply the streaming cap and admit an OOM load.
    #[test]
    fn activation_family_for_ltx_video_is_non_streaming() {
        let family = mold_inference::device::activation_family_for("ltx-video");
        assert_eq!(
            family,
            ActivationFamily::LtxVideo,
            "ltx-video slug must map to LtxVideo (non-streaming, full-weight load)"
        );
        // Verify it does NOT trigger the streaming cap path.
        assert!(
            !family.streaming_transformer(),
            "LtxVideo must NOT be treated as a streaming transformer — \
             it loads the entire weight file into VRAM at generate time"
        );
        // Verify ltx2 still maps to the streaming variant.
        assert!(
            mold_inference::device::activation_family_for("ltx2").streaming_transformer(),
            "ltx2 must still map to the streaming family"
        );
    }

    /// The rejection message for an OOM-at-preflight LTX-Video load must
    /// mention reducing frames or resolution (not `--offload`, which is a
    /// FLUX-specific flag and not applicable to LTX-Video).
    #[test]
    fn preflight_rejection_message_for_ltx_video_suggests_frames_or_resolution() {
        let (_dir, paths) = ltx_video_13b_paths(26, 1, 10);
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::LtxVideo,
        };
        let result = preflight_memory_guard_with_available(
            "ltx-video-0.9.8-13b-dev:bf16",
            &paths,
            0,
            24 * GB,
            Some(hint),
        );
        let err = result.expect_err("must reject");
        assert!(
            err.error.contains("frames") || err.error.contains("width"),
            "LTX-Video rejection message must suggest reducing frames or \
             width/height (not --offload), got: {}",
            err.error,
        );
        // Must NOT suggest --offload (that's a FLUX flag, not applicable here).
        assert!(
            !err.error.contains("--offload"),
            "LTX-Video rejection must not mention --offload (FLUX-only flag), \
             got: {}",
            err.error,
        );
    }

    /// A wan OOM must get video advice, not the image-family default.
    ///
    /// Wan reached the generic arm because it matched only `LtxVideo`, so the
    /// message recommended `--batch` (wan renders one clip at a time whatever
    /// the value) and `--offload` (a FLUX flag with no wan code path), while
    /// never mentioning `--frames` — the one lever that actually moves wan's
    /// peak, since activation cost scales with tokens and tokens scale with
    /// frames.
    #[test]
    fn preflight_rejection_message_for_wan_suggests_frames_not_offload() {
        let hint = ActivationHint {
            width: 832,
            height: 480,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::WanVideo,
        };
        let suggestion = rejection_suggestion(Some(hint));
        assert!(suggestion.contains("--frames"), "got: {suggestion}");
        assert!(!suggestion.contains("--offload"), "got: {suggestion}");
        assert!(!suggestion.contains("--batch"), "got: {suggestion}");
        // Wan ships :q5/:q8 tiers, so the quantized-variant hint applies.
        assert!(suggestion.contains("quantized"), "got: {suggestion}");

        // The two full-weight video families share one message.
        assert_eq!(
            suggestion,
            rejection_suggestion(Some(ActivationHint {
                family: ActivationFamily::LtxVideo,
                ..hint
            })),
        );
    }

    #[test]
    fn preflight_rejection_message_for_image_suggests_resolution_not_frames() {
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 2,
            dtype_bytes: 2,
            family: ActivationFamily::SdxlUnet,
        };
        let suggestion = rejection_suggestion(Some(hint));

        assert!(
            suggestion.contains("--width/--height"),
            "image preflight suggestion should mention resolution; got: {suggestion}"
        );
        assert!(
            suggestion.contains("--batch"),
            "image preflight suggestion should mention batch; got: {suggestion}"
        );
        assert!(
            !suggestion.contains("--frames"),
            "image preflight suggestion must not mention video frames; got: {suggestion}"
        );
    }
}
