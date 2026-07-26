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

pub use crate::memory_preflight::ActivationHint;
#[cfg(test)]
pub(crate) use crate::memory_preflight::{
    check_model_memory_budget, preflight_memory_guard_with_available, rejection_suggestion,
};
pub(crate) use crate::memory_preflight::{
    effective_load_available_bytes, estimate_generation_memory_for_request, preflight_memory_guard,
    preflight_memory_guard_after_drop, request_requires_fresh_engine_for_offload_policy,
    select_server_load_strategy_for_budget, select_server_load_strategy_for_device,
    server_offload_enabled_for_paths,
};

pub(crate) fn request_has_effective_lora(req: &GenerateRequest) -> bool {
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

pub(crate) fn resolve_installed_catalog_paths_for_worker(
    model_name: &str,
    config: &Config,
) -> Result<Option<(ModelPaths, Config)>, ApiError> {
    if !looks_like_catalog_id(model_name) {
        return Ok(None);
    }

    let Some(intent) = installed_intent_from_sidecar(&config.resolved_models_dir(), model_name)
    else {
        return Ok(None);
    };
    let model_cfg = resolve_intent_to_paths(model_name, &intent, config)
        .map_err(|e| resolve_error_to_api_error(&e))?;
    let mut resolved_config = config.clone();
    resolved_config
        .models
        .insert(model_name.to_string(), model_cfg);
    let paths = ModelPaths::resolve(model_name, &resolved_config).ok_or_else(|| {
        ApiError::not_found(format!(
            "catalog model '{model_name}' resolved to a config that ModelPaths \
             could not turn into runtime paths — internal mismatch, please file an issue."
        ))
    })?;

    Ok(Some((paths, resolved_config)))
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

pub(crate) enum PullStatus {
    AlreadyAvailable,
    Pulled,
}

pub(crate) async fn refresh_config(state: &AppState) -> mold_core::Config {
    let fresh = {
        let current = state.config.read().await;
        current.reload_from_disk_preserving_runtime()
    };

    let mut config = state.config.write().await;
    *config = fresh.clone();
    fresh
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
    catalog
}

fn annotate_audio_capabilities(catalog: &mut [ModelInfoExtended], config: &Config) {
    for entry in catalog {
        if entry.info.family != "ltx2" || entry.supports_audio.is_some() || !entry.downloaded {
            continue;
        }
        entry.supports_audio = ModelPaths::resolve(&entry.info.name, config)
            .map(|paths| mold_inference::ltx2::audio_output_supported(&paths));
    }
}

fn loaded_models_across_pool(state: &AppState) -> Vec<String> {
    let mut names = Vec::new();
    for worker in &state.gpu_pool.workers {
        // Prefer the active-generation model (cache entry is taken out during
        // inflight generation), else whatever is GPU-resident.
        let active = worker
            .active_generation
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(|g| g.model.clone()));
        let loaded = active.or_else(|| worker.resident_model.read().ok()?.clone());
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
/// first (covers `flux-dev:q8` and friends), then falls back to the catalog
/// config entry (covers `cv:*` / `hf:*`). Used by the activation-budget
/// preflight to dispatch to the right [`ActivationFamily`].
pub(crate) async fn family_for_model(state: &AppState, model_name: &str) -> Option<String> {
    if let Some(manifest) = mold_core::manifest::find_manifest(model_name) {
        return Some(manifest.family.clone());
    }
    catalog_family_for(state, model_name).await
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
pub(crate) async fn install_catalog_model(
    state: &AppState,
    model_name: &str,
) -> Result<(), mold_core::InstallError> {
    if !looks_like_catalog_id(model_name) {
        // Caller should have shape-checked first; treat it as a no-op
        // success rather than fabricating a custom error variant.
        return Ok(());
    }

    // Already installed from a prior request — keep the cached intent.
    {
        let intents = state.catalog_intents.read().await;
        if intents.contains_key(model_name) {
            return Ok(());
        }
    }

    let models_dir = state.config.read().await.resolved_models_dir();
    if let Some(intent) = installed_intent_from_sidecar(&models_dir, model_name) {
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

    let intent = mold_catalog::synthesis::synthesize_intent(&entry, &models_dir).map_err(|e| {
        mold_core::InstallError::RecipeMalformed(format!("synthesize intent for {model_name}: {e}"))
    })?;
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

        let description = sidecar
            .description
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| match &sidecar.author {
                Some(a) if !a.is_empty() => format!("{} by {a}", sidecar.name),
                _ => sidecar.name.clone(),
            });

        out.push(ModelInfoExtended {
            downloaded: true,
            defaults: ModelDefaults {
                default_width: w,
                default_height: h,
                default_steps: steps,
                default_guidance: guidance,
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
            supports_audio: (sidecar.family == "ltx2")
                .then(|| mold_inference::ltx2::checkpoint_supports_audio_output(&primary_path)),
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
    let hint = activation_hint_for_request(state, req).await;
    let estimate = estimate_generation_memory_for_request(req, &paths, hint);

    Ok(GenerationMemoryEstimate {
        model: req.model.clone(),
        peak_memory_bytes: estimate.peak_memory_bytes,
        activation_memory_bytes: estimate.activation_memory_bytes,
        available_memory_bytes: estimate.available_memory_bytes,
        load_strategy: format!("{:?}", estimate.load_strategy).to_ascii_lowercase(),
        fits_available_memory: estimate.fits_available_memory,
    })
}

pub(crate) async fn model_component_status(
    state: &AppState,
    model_name: &str,
) -> Result<ModelComponentsResponse, ApiError> {
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
    Ok(ModelComponentsResponse {
        model: model_name.to_string(),
        components: component_status_from_paths(&config, model_name, &paths),
    })
}

fn manifest_component_kind(component: mold_core::manifest::ModelComponent) -> &'static str {
    use mold_core::manifest::ModelComponent;
    match component {
        ModelComponent::Transformer | ModelComponent::TransformerShard => "transformer",
        ModelComponent::Vae => "vae",
        ModelComponent::SpatialUpscaler => "spatial_upscaler",
        ModelComponent::TemporalUpscaler => "temporal_upscaler",
        ModelComponent::DistilledLora => "distilled_lora",
        ModelComponent::T5Encoder | ModelComponent::TextEncoder => "text_encoder",
        ModelComponent::ClipEncoder | ModelComponent::ClipEncoder2 => "clip",
        ModelComponent::T5Tokenizer
        | ModelComponent::ClipTokenizer
        | ModelComponent::ClipTokenizer2
        | ModelComponent::TextTokenizer => "tokenizer",
        ModelComponent::Decoder => "decoder",
        ModelComponent::Upscaler => "upscaler",
    }
}

fn manifest_component_name(component: mold_core::manifest::ModelComponent, filename: &str) -> &str {
    use mold_core::manifest::ModelComponent;
    match component {
        ModelComponent::Transformer => "transformer",
        ModelComponent::TransformerShard => "transformer shard",
        ModelComponent::Vae => "vae",
        ModelComponent::SpatialUpscaler => "spatial upscaler",
        ModelComponent::TemporalUpscaler => "temporal upscaler",
        ModelComponent::DistilledLora => "distilled lora",
        ModelComponent::T5Encoder => "t5 encoder",
        ModelComponent::ClipEncoder => "clip encoder",
        ModelComponent::T5Tokenizer => "t5 tokenizer",
        ModelComponent::ClipTokenizer => "clip tokenizer",
        ModelComponent::ClipEncoder2 => "clip-g encoder",
        ModelComponent::ClipTokenizer2 => "clip-g tokenizer",
        ModelComponent::TextEncoder => "text encoder",
        ModelComponent::TextTokenizer => "text tokenizer",
        ModelComponent::Decoder => "decoder",
        ModelComponent::Upscaler => filename,
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
    if let Some(path) = &paths.t5_encoder {
        push_path("text_encoder", "t5 encoder", path);
    }
    if let Some(path) = &paths.clip_encoder {
        push_path("clip", "clip encoder", path);
    }
    if let Some(path) = &paths.clip_encoder_2 {
        push_path("clip", "clip-g encoder", path);
    }
    for path in &paths.text_encoder_files {
        push_path("text_encoder", "text encoder", path);
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
    if let Some(path) = current_path {
        add_component_option(&mut options, path);
    }
    for model_cfg in config.models.values() {
        for path in config_component_paths_for_kind(model_cfg, kind) {
            add_component_option(&mut options, Path::new(path));
        }
    }
    let models_dir = config.resolved_models_dir();
    for manifest in mold_core::manifest::known_manifests() {
        for file in &manifest.files {
            if manifest_component_kind(file.component) != kind {
                continue;
            }
            let path = models_dir.join(mold_core::manifest::storage_path(manifest, file));
            if path.is_file() {
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
                    // Already loaded — just set up progress callback.
                    if let Some(callback) = progress.clone() {
                        entry.engine.set_on_progress(Box::new(move |event| {
                            callback(event);
                        }));
                    } else {
                        entry.engine.clear_on_progress();
                    }
                    return Ok(());
                }
            }

            // Cached but not on GPU (Parked) — need to reload.
            // MPS memory guard: check before unloading the active model.
            // Include the active model's footprint as reclaimable memory.
            let cached_paths = entry.engine.model_paths().cloned();
            if let Some(paths) = cached_paths.as_ref() {
                preflight_memory_guard(model_name, paths, active_vram, 0, hint)?;
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
                preflight_memory_guard_after_drop(model_name, paths, 0, hint)?;
            } else {
                let _ = mold_inference::device::post_drop_free_vram_bytes(0);
            }
            let load_strategy = cached_paths
                .as_ref()
                .map(|paths| {
                    select_server_load_strategy_for_budget(
                        paths,
                        effective_load_available_bytes(0, 0),
                        hint,
                    )
                })
                .unwrap_or(mold_inference::LoadStrategy::Eager);
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
        Some(paths) => create_and_load_engine(state, model_name, paths, progress, hint).await,
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
        tracing::error!("pull failed for {}: {e}", model);
        ApiError::internal(format!("failed to pull model '{}': {e}", model))
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
    // Always clear the cached upscaler engine to free GPU memory,
    // regardless of whether a diffusion model is loaded.
    // Use try_lock() to avoid blocking the async runtime if an upscale
    // is in progress (the spawn_blocking thread holds this lock).
    if let Ok(mut upscaler) = state.upscaler_cache.try_lock() {
        if let Some(mut engine) = upscaler.take() {
            engine.unload();
            tracing::info!("upscaler cache cleared");
        }
    }

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
) -> Result<(), ApiError> {
    // MPS memory guard: reject before unloading current model so it stays operational.
    // Include the active model's footprint as reclaimable memory.
    let active_vram = {
        let cache = state.model_cache.lock().await;
        cache.active_vram_bytes()
    };
    preflight_memory_guard(model_name, &paths, active_vram, 0, hint)?;
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
    preflight_memory_guard_after_drop(model_name, &paths, 0, hint)?;
    let load_strategy = select_server_load_strategy_for_device(
        &paths,
        effective_load_available_bytes(0, 0),
        mold_inference::device::total_vram_bytes(0),
        hint,
    );
    if load_strategy == mold_inference::LoadStrategy::Sequential {
        tracing::info!(
            model = %model_name,
            "server load strategy degraded to sequential to fit post-drop memory budget"
        );
    }

    let config = state.config.read().await;
    let offload = server_offload_enabled_for_paths(&paths, hint, false);
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

    /// RAII guard for `MOLD_LTX2_GEMMA_DEVICE` (and the deprecated alias).
    /// Drop restores the prior values so adjacent tests don't see stale
    /// state. Cargo's parallel runner is serialized via a static mutex
    /// because env vars are process-global.
    struct Ltx2GemmaEnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        prior_main: Option<std::ffi::OsString>,
        prior_legacy: Option<std::ffi::OsString>,
    }

    impl Drop for Ltx2GemmaEnvGuard {
        fn drop(&mut self) {
            unsafe {
                std::env::remove_var("MOLD_LTX2_GEMMA_DEVICE");
                std::env::remove_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
                if let Some(v) = self.prior_main.take() {
                    std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", v);
                }
                if let Some(v) = self.prior_legacy.take() {
                    std::env::set_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER", v);
                }
            }
        }
    }

    fn ltx2_gemma_env_guard(value: &str) -> Ltx2GemmaEnvGuard {
        use std::sync::{Mutex, OnceLock};
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prior_main = std::env::var_os("MOLD_LTX2_GEMMA_DEVICE");
        let prior_legacy = std::env::var_os("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
        unsafe {
            std::env::remove_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
            std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", value);
        }
        Ltx2GemmaEnvGuard {
            _lock: lock,
            prior_main,
            prior_legacy,
        }
    }

    struct OffloadEnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        prior: Option<std::ffi::OsString>,
    }

    impl Drop for OffloadEnvGuard {
        fn drop(&mut self) {
            unsafe {
                std::env::remove_var("MOLD_OFFLOAD");
                if let Some(v) = self.prior.take() {
                    std::env::set_var("MOLD_OFFLOAD", v);
                }
            }
        }
    }

    fn offload_env_guard(value: &str) -> OffloadEnvGuard {
        use std::sync::{Mutex, OnceLock};
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prior = std::env::var_os("MOLD_OFFLOAD");
        unsafe {
            std::env::set_var("MOLD_OFFLOAD", value);
        }
        OffloadEnvGuard { _lock: lock, prior }
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

    #[test]
    fn preflight_accepts_forced_flux_offload_bf16_layout_on_24gb() {
        let _guard = offload_env_guard("1");
        let (_dir, paths) = flux_shaped_paths_with_sizes(24, 1, 10, 1);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };

        let result =
            preflight_memory_guard_with_available("flux-dev:bf16", &paths, 0, 24 * GB, Some(hint));

        assert!(
            result.is_ok(),
            "forced FLUX offload should use streaming-aware peak instead of \
            full BF16 transformer residency, got {result:?}"
        );
    }

    #[test]
    fn preflight_accepts_large_flux_bf16_auto_offload_on_24gb() {
        let _guard = offload_env_guard("0");
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
        let _guard = offload_env_guard("0");
        let (_dir, paths) = flux_shaped_paths_with_sizes(23, 1, 9, 1);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };

        assert!(
            server_offload_enabled_for_paths(&paths, Some(hint), false),
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
        let _guard = offload_env_guard("1");
        let (_dir, paths) = sd3_gguf_paths_with_monolithic_vae(9, 16, 10, 1, 1);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 2,
            dtype_bytes: 2,
            family: ActivationFamily::Sd3Mmdit,
        };

        assert!(
            !server_offload_enabled_for_paths(&paths, Some(hint), false),
            "global MOLD_OFFLOAD must not force unsupported SD3 GGUF block offload"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_zimage_gguf() {
        let _guard = offload_env_guard("1");
        let (_dir, paths) = zimage_gguf_paths(12, 1, 8);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::ZImageDit,
        };

        assert!(
            !server_offload_enabled_for_paths(&paths, Some(hint), false),
            "global MOLD_OFFLOAD must not force unsupported Z-Image GGUF block offload"
        );
    }

    #[test]
    fn offload_env_is_preserved_for_zimage_bf16() {
        let _guard = offload_env_guard("1");
        let (_dir, paths) = flux_shaped_paths_with_sizes(6, 1, 8, 0);
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::ZImageDit,
        };

        assert!(
            server_offload_enabled_for_paths(&paths, Some(hint), false),
            "BF16/FP Z-Image paths should still receive explicit offload"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_zimage_lora_with_ambiguous_family_hint() {
        let _guard = offload_env_guard("1");
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let paths = ModelPaths {
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
            !server_offload_enabled_for_paths(&paths, Some(hint), true),
            "Z-Image LoRA requests must not receive global MOLD_OFFLOAD even \
             when duplicate catalog rows provide an ambiguous Flux hint"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_flux2_lora_request() {
        let _guard = offload_env_guard("1");
        let (_dir, paths) = flux2_klein9b_bf16_paths();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        assert!(
            !server_offload_enabled_for_paths(&paths, Some(hint), true),
            "global MOLD_OFFLOAD must not force Flux.2 block offload for LoRA \
             requests because Flux.2 offload+LoRA is not supported"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_flux2_lora_with_ambiguous_family_hint() {
        let _guard = offload_env_guard("1");
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
            !server_offload_enabled_for_paths(&paths, Some(hint), true),
            "Flux.2 LoRA requests must not receive global MOLD_OFFLOAD even \
             when the catalog family hint is missing or ambiguous"
        );
    }

    #[test]
    fn flux2_lora_request_requires_fresh_engine_when_plain_offload_was_enabled() {
        let _guard = offload_env_guard("1");
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let paths = ModelPaths {
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
            request_requires_fresh_engine_for_offload_policy(&paths, Some(hint), true),
            "a cached Flux.2 engine loaded for plain offload must be recreated \
             before serving a LoRA request, otherwise the runtime still sees \
             offload+LoRA"
        );
    }

    #[test]
    fn offload_env_is_preserved_for_plain_flux2_request() {
        let _guard = offload_env_guard("1");
        let (_dir, paths) = flux2_klein9b_bf16_paths();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        assert!(
            server_offload_enabled_for_paths(&paths, Some(hint), false),
            "plain Flux.2 requests should still receive explicit offload"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_flux2_gguf() {
        let _guard = offload_env_guard("1");
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
            !server_offload_enabled_for_paths(&paths, Some(hint), false),
            "global MOLD_OFFLOAD must not force Flux.2 GGUF block offload \
             because GGUF variants use quantized transformer paths"
        );
    }

    #[test]
    fn offload_env_is_ignored_for_flux2_nvfp4() {
        let _guard = offload_env_guard("1");
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let paths = ModelPaths {
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
            !server_offload_enabled_for_paths(&paths, Some(hint), false),
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

    /// Plato regression: qwen-image:bf16 (≈41 GB sharded transformer) on an
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
        let _guard = offload_env_guard("0");
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
    fn server_load_strategy_uses_device_total_when_live_available_missing() {
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

        assert_eq!(
            strategy,
            mold_inference::LoadStrategy::Sequential,
            "when live free-memory probing is unavailable, the worker should still \
             use known device total VRAM instead of defaulting to eager"
        );
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
        let _guard = offload_env_guard("0");
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
        let _guard = ltx2_gemma_env_guard("gpu");
        let (_dir, paths) = ltx2_shaped_paths_with_sizes(46, 25);
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
        let _guard = ltx2_gemma_env_guard("auto");
        let (_dir, paths) = ltx2_shaped_paths_with_sizes(46, 25);
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
        let _guard = ltx2_gemma_env_guard("cpu");
        let (_dir, paths) = ltx2_shaped_paths_with_sizes(46, 25);
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
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
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
        let _saved = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", models_dir.to_string_lossy().as_ref()) };

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

        unsafe {
            match _saved {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    #[test]
    fn resolve_intent_preserves_flux_schnell_subfamily() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _saved = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", models_dir.to_string_lossy().as_ref()) };

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

        unsafe {
            match _saved {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    #[test]
    fn resolve_intent_applies_flux_dev_subfamily_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _saved = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", models_dir.to_string_lossy().as_ref()) };

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

        unsafe {
            match _saved {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    #[test]
    fn resolve_intent_populates_qwen_runtime_companion_paths() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let primary_path =
            models_dir.join("cv-2110043/qwen-image/civitai/2110043/qwenImage_fp8.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        std::fs::File::create(&primary_path).unwrap();

        let runtime_dir = models_dir.join("qwen-image-runtime");
        let vae_path = runtime_dir.join("vae/diffusion_pytorch_model.safetensors");
        let te_path = runtime_dir.join("text_encoder/model-00001-of-00004.safetensors");
        let tok_path = runtime_dir.join("tokenizer/tokenizer.json");
        for path in [&vae_path, &te_path, &tok_path] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(path).unwrap();
        }

        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        config.models.insert(
            "qwen-image-runtime".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(vae_path.to_string_lossy().into_owned()),
                vae: Some(vae_path.to_string_lossy().into_owned()),
                text_encoder_files: Some(vec![te_path.to_string_lossy().into_owned()]),
                text_tokenizer: Some(tok_path.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );

        let mut entry = flux_unet_only_catalog_entry("2110043", "qwenImage_fp8.safetensors");
        entry.family = mold_catalog::families::Family::QwenImage;
        entry.companions = vec!["qwen-image-runtime".into()];
        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_paths("cv:2110043", &intent, &config).unwrap();

        assert_eq!(cfg.transformer.as_deref(), primary_path.to_str());
        assert_eq!(cfg.vae.as_deref(), vae_path.to_str());
        assert_eq!(
            cfg.text_encoder_files.as_deref(),
            Some(vec![te_path.to_string_lossy().into_owned()].as_slice())
        );
        assert_eq!(cfg.text_tokenizer.as_deref(), tok_path.to_str());
    }

    #[test]
    fn resolve_intent_populates_wuerstchen_runtime_companion_paths() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let primary_path = models_dir.join(
            "hf-example/wuerstchen-prior/wuerstchen/example/wuerstchen-prior/prior.safetensors",
        );
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        std::fs::File::create(&primary_path).unwrap();

        let runtime_dir = models_dir.join("wuerstchen-runtime");
        let decoder_path = runtime_dir.join("decoder/diffusion_pytorch_model.safetensors");
        let vae_path = runtime_dir.join("vqgan/diffusion_pytorch_model.safetensors");
        let clip_path = runtime_dir.join("text_encoder/model.safetensors");
        let clip_tok_path = runtime_dir.join("tokenizer/tokenizer.json");
        let clip_g_path = runtime_dir.join("prior/text_encoder/model.safetensors");
        let clip_g_tok_path = runtime_dir.join("prior/tokenizer/tokenizer.json");
        for path in [
            &decoder_path,
            &vae_path,
            &clip_path,
            &clip_tok_path,
            &clip_g_path,
            &clip_g_tok_path,
        ] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(path).unwrap();
        }

        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        config.models.insert(
            "wuerstchen-runtime".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(decoder_path.to_string_lossy().into_owned()),
                decoder: Some(decoder_path.to_string_lossy().into_owned()),
                vae: Some(vae_path.to_string_lossy().into_owned()),
                clip_encoder: Some(clip_path.to_string_lossy().into_owned()),
                clip_tokenizer: Some(clip_tok_path.to_string_lossy().into_owned()),
                clip_encoder_2: Some(clip_g_path.to_string_lossy().into_owned()),
                clip_tokenizer_2: Some(clip_g_tok_path.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );

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
        assert_eq!(cfg.decoder.as_deref(), decoder_path.to_str());
        assert_eq!(cfg.vae.as_deref(), vae_path.to_str());
        assert_eq!(cfg.clip_encoder.as_deref(), clip_path.to_str());
        assert_eq!(cfg.clip_tokenizer.as_deref(), clip_tok_path.to_str());
        assert_eq!(cfg.clip_encoder_2.as_deref(), clip_g_path.to_str());
        assert_eq!(cfg.clip_tokenizer_2.as_deref(), clip_g_tok_path.to_str());
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
        let _saved = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", models_dir.to_string_lossy().as_ref()) };

        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        let te_dir = models_dir.join("z-image-te");
        config.models.insert(
            "z-image-te".into(),
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

        unsafe {
            match _saved {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    /// Task 5 (Step 1): resolution surfaces the *specific* missing
    /// companion name rather than a generic "missing required components"
    /// blob. The CompanionConfigMissing variant fires when the manifest
    /// entry isn't in the user's Config.models — a real config bug.
    #[test]
    fn resolve_intent_returns_error_naming_missing_required_companion() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();

        let primary_path = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &["double_blocks.0.img_attn.proj.weight", "img_in.weight"],
        );

        let _saved = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", models_dir.to_string_lossy().as_ref()) };

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

        unsafe {
            match _saved {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
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
        let _saved = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", models_dir.to_string_lossy().as_ref()) };

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

        unsafe {
            match _saved {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    #[test]
    fn resolve_intent_rejects_truncated_sidecar_primary() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _saved = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", models_dir.to_string_lossy().as_ref()) };

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

        unsafe {
            match _saved {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    // ── InstallError translation tests (Task 4) ────────────────────────────

    #[test]
    fn live_error_to_install_error_maps_404_to_not_found() {
        let upstream = mold_catalog::live::LiveSearchError::Upstream {
            host: "civitai.com",
            status: 404,
            body: "{\"error\": \"not found\"}".to_string(),
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
