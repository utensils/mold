use std::sync::Arc;

use mold_core::{build_model_catalog, ModelDefaults, ModelInfo, ModelInfoExtended, ModelPaths};

use crate::model_cache::ModelResidency;
use crate::{routes::ApiError, state::AppState};

pub(crate) type EngineProgressCallback = Arc<dyn Fn(mold_inference::ProgressEvent) + Send + Sync>;

// ── MPS memory guard ────────────────────────────────────────────────────────

/// Pure logic for the server memory guard, factored out for testing.
///
/// Hard-fails if peak > 90% of available (model won't fit even with page reclamation).
/// Warns if peak > 80% of available (tight but feasible).
fn check_model_memory_budget(
    model_name: &str,
    peak_bytes: u64,
    available_bytes: u64,
) -> Result<(), ApiError> {
    let hard_limit = available_bytes * 9 / 10; // 90%
    if peak_bytes > hard_limit {
        return Err(ApiError::insufficient_memory(format!(
            "model '{}' needs ~{:.1} GB but only ~{:.1} GB available. \
             Close other applications, unload the current model, or use a smaller variant.",
            model_name,
            peak_bytes as f64 / 1_000_000_000.0,
            available_bytes as f64 / 1_000_000_000.0,
        )));
    }

    let warn_limit = available_bytes * 8 / 10; // 80%
    if peak_bytes > warn_limit {
        tracing::warn!(
            model = %model_name,
            peak_gb = format_args!("{:.1}", peak_bytes as f64 / 1_000_000_000.0),
            available_gb = format_args!("{:.1}", available_bytes as f64 / 1_000_000_000.0),
            "model is close to memory limit — may trigger page reclamation"
        );
    }

    Ok(())
}

/// Pure inner: given an `available_bytes` budget and the active model's
/// reclaimable VRAM, decide whether the new model fits. Adding
/// `active_vram_bytes` to `available_bytes` accounts for the currently-loaded
/// model that will be unloaded before the new one loads — without this, a
/// swap of two near-equal-size models would be falsely rejected even though
/// the swap is feasible.
///
/// Peak is estimated under `LoadStrategy::Sequential` because every diffusion
/// family in this repo (FLUX, SD3, Z-Image, Flux.2, Qwen-Image, LTX) drops
/// text encoders from GPU after encoding before the transformer denoises.
/// The Eager sum (`transformer + vae + all_encoders`) overcounts by the
/// encoder weight on every load — enough to false-reject a quantized FLUX on
/// a 24 GB card even when the swap would actually fit.
pub(crate) fn preflight_memory_guard_with_available(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    available_bytes: u64,
) -> Result<(), ApiError> {
    let peak = mold_inference::device::estimate_peak_memory(
        paths,
        mold_inference::LoadStrategy::Sequential,
    );
    let effective_available = available_bytes.saturating_add(active_vram_bytes);
    check_model_memory_budget(model_name, peak, effective_available)
}

/// Check whether estimated peak memory fits before committing to a model load.
///
/// Budgeting strategy on CUDA:
/// - **No active model on this GPU** — the new load lands in whatever is
///   currently free, so use `free_vram_bytes(gpu_ordinal)`.
/// - **Active model present** — the call site unloads it and runs
///   `cuDevicePrimaryCtxReset_v2`, which releases *every* allocation on the
///   device (transformer, leftover activation buffers, fragmentation in the
///   caching pool). The realistic post-reclaim budget is total VRAM, not
///   `free + recorded active_vram`. Using the latter under-counts whatever
///   the cache forgot to track (notably the encoder churn during the
///   previous generation) and produces false rejections.
///
/// On macOS (unified memory) we keep the additive `available + active_vram`
/// budget because Metal has no equivalent device-wide context reset; tensors
/// freed during `unload()` simply return to the system page cache.
/// On other platforms with no memory query available, the guard is a no-op.
pub(crate) fn preflight_memory_guard(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] gpu_ordinal: usize,
) -> Result<(), ApiError> {
    // CUDA branch: when an active model will be reclaimed via primary-context
    // reset, the post-reclaim budget is the device total, not free+active.
    #[cfg(feature = "cuda")]
    {
        if active_vram_bytes > 0 {
            if let Some(total) = mold_inference::device::total_vram_bytes(gpu_ordinal) {
                return preflight_memory_guard_with_available(model_name, paths, 0, total);
            }
        }
        if let Some(free) = mold_inference::device::free_vram_bytes(gpu_ordinal) {
            return preflight_memory_guard_with_available(
                model_name,
                paths,
                active_vram_bytes,
                free,
            );
        }
    }

    // macOS unified memory: query system memory and add reclaimable footprint.
    if let Some(available) = mold_inference::device::available_system_memory_bytes() {
        if available > 0 {
            return preflight_memory_guard_with_available(
                model_name,
                paths,
                active_vram_bytes,
                available,
            );
        }
    }

    // No memory info available on this platform — skip the guard.
    Ok(())
}
pub(crate) type DownloadProgressCallback =
    Arc<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>;

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
    catalog
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
        let loaded = active.or_else(|| {
            let cache = worker.model_cache.lock().ok()?;
            cache.active_model().map(|s| s.to_string())
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
// Mirrors the logic in `mold-cli/src/catalog_bridge.rs` so the server can
// resolve `cv:*` model IDs (Civitai single-file checkpoints downloaded via
// the catalog web UI) without the binary crate as an intermediary.

fn looks_like_catalog_id(id: &str) -> bool {
    id.starts_with("cv:") || id.starts_with("hf:")
}

/// Best-effort family lookup for a model name, used to feed
/// `validate_generate_request_with_family` so catalog (`cv:*` / `hf:*`) IDs
/// get the same family-gated feature checks as manifest-resident models.
///
/// Returns the catalog DB's family string (`"ltx2"`, `"sdxl"`, …) for an
/// installed catalog entry, or `None` for manifest-resident or unknown
/// model names — in which case validation falls back to its own
/// manifest-aware lookup. Errors are swallowed and returned as `None` so a
/// flaky DB read can't block a generate request that doesn't actually need
/// the family hint (e.g. plain image generation against a manifest model).
pub(crate) fn catalog_family_for(state: &AppState, model_name: &str) -> Option<String> {
    if !looks_like_catalog_id(model_name) {
        return None;
    }
    match state.catalog_db.catalog_get(model_name) {
        Ok(Some(row)) => Some(row.family),
        Ok(None) => None,
        Err(e) => {
            tracing::debug!(model = model_name, error = %e, "catalog_family_for: DB lookup failed");
            None
        }
    }
}

fn copy_catalog_companion(cfg: &mut mold_core::ModelConfig, companion: &str, paths: &ModelPaths) {
    let to_str = |p: &std::path::PathBuf| p.to_str().map(str::to_owned);
    match companion {
        "clip-l" => {
            cfg.clip_encoder = to_str(&paths.transformer);
            cfg.clip_tokenizer = paths.clip_tokenizer.as_ref().and_then(to_str);
        }
        "clip-g" => {
            cfg.clip_encoder_2 = to_str(&paths.transformer);
            cfg.clip_tokenizer_2 = paths
                .clip_tokenizer
                .as_ref()
                .and_then(to_str)
                .or_else(|| cfg.clip_tokenizer.clone());
        }
        "sdxl-vae" | "sd-vae-ft-mse" | "flux-vae" => {}
        "ltx-video-vae" | "flux2-vae" => {
            cfg.vae = to_str(&paths.transformer);
        }
        "t5-v1_1-xxl" => {
            cfg.t5_encoder = to_str(&paths.transformer);
            cfg.t5_tokenizer = paths.t5_tokenizer.as_ref().and_then(to_str);
        }
        "z-image-te" | "flux2-te" | "flux2-te-9b" => {
            cfg.text_encoder_files = paths
                .text_encoder_files
                .iter()
                .filter_map(to_str)
                .collect::<Vec<_>>()
                .into();
            cfg.text_tokenizer = paths.text_tokenizer.as_ref().and_then(to_str);
        }
        "ltx2-te" => {
            // Gemma 3 12B for LTX-2. The runtime calls `gemma_root` which
            // takes the parent directory of `text_encoder_files[0]`, so
            // populating that vec is sufficient — we don't need a separate
            // `text_tokenizer` field (the manifest tags every Gemma file
            // including `tokenizer.json` / `tokenizer.model` as
            // ModelComponent::TextEncoder, so the tokenizer rides along).
            cfg.text_encoder_files = paths
                .text_encoder_files
                .iter()
                .filter_map(to_str)
                .collect::<Vec<_>>()
                .into();
        }
        _ => {}
    }
}

fn synthesize_catalog_config(
    row: &mold_db::catalog::CatalogRow,
    models_dir: &std::path::Path,
    config: &mold_core::Config,
) -> anyhow::Result<mold_core::ModelConfig> {
    let recipe = serde_json::from_str::<mold_catalog::entry::DownloadRecipe>(&row.download_recipe)
        .map_err(|e| anyhow::anyhow!("malformed download_recipe for {}: {e}", row.id))?;

    let primary = recipe
        .files
        .first()
        .ok_or_else(|| anyhow::anyhow!("empty recipe for {}", row.id))?;

    let sanitized = mold_core::download::sanitize_recipe_id(&row.id);
    let (author, name) = match row.source_id.split_once('/') {
        Some((a, n)) => (a, n),
        None => ("", row.source_id.as_str()),
    };
    let rendered_dest =
        mold_catalog::entry::render_recipe_dest(&primary.dest, &row.family, author, name);
    let primary_path = models_dir.join(&sanitized).join(&rendered_dest);
    let primary_str = primary_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("path not UTF-8: {primary_path:?}"))?
        .to_string();

    let mut cfg = mold_core::ModelConfig {
        family: Some(row.family.clone()),
        ..Default::default()
    };

    if row.bundling == "single-file" {
        cfg.transformer = Some(primary_str.clone());
        cfg.vae = Some(primary_str);
    } else {
        anyhow::bail!(
            "bundling={:?} not supported (single-file only)",
            row.bundling
        );
    }

    // Populate companion paths from already-resolved manifest entries.
    use mold_catalog::companions::companions_for;
    use mold_catalog::entry::{Bundling, Kind};
    use mold_catalog::families::Family;
    let fam = match row.family.as_str() {
        "sd15" => Some(Family::Sd15),
        "sdxl" => Some(Family::Sdxl),
        "flux" => Some(Family::Flux),
        "flux2" => Some(Family::Flux2),
        "z-image" => Some(Family::ZImage),
        "ltx-video" => Some(Family::LtxVideo),
        "ltx2" => Some(Family::Ltx2),
        "qwen-image" => Some(Family::QwenImage),
        "wuerstchen" => Some(Family::Wuerstchen),
        _ => None,
    };
    // The row stores kind as a kebab-case string ("checkpoint", "lora", ...);
    // parse it back so `companions_for` can short-circuit for LoRAs and other
    // adapter-shaped entries that don't pull T5/CLIP/VAE companions.
    let kind: Kind = serde_json::from_value(serde_json::Value::String(row.kind.clone()))
        .unwrap_or(Kind::Checkpoint);
    if let Some(fam) = fam {
        for companion in companions_for(fam, row.sub_family.as_deref(), Bundling::SingleFile, kind)
        {
            match ModelPaths::resolve(&companion, config) {
                Some(paths) => copy_catalog_companion(&mut cfg, &companion, &paths),
                None => tracing::warn!(
                    catalog_id = %row.id,
                    companion = %companion,
                    "companion did not resolve from manifest paths — engine load may fail"
                ),
            }
        }
    }

    Ok(cfg)
}

/// If `model_name` is a catalog ID and not yet in the config, synthesize its
/// `ModelConfig` from the catalog DB and insert it. Returns `true` when a
/// config entry was added (or already existed).
async fn install_catalog_model(state: &AppState, model_name: &str) -> bool {
    if !looks_like_catalog_id(model_name) {
        return false;
    }

    // Already synthesized from a prior request?
    {
        let config = state.config.read().await;
        if config.models.contains_key(model_name) {
            return true;
        }
    }

    let row = match state.catalog_db.catalog_get(model_name) {
        Ok(Some(r)) => r,
        Ok(None) => {
            tracing::debug!(model = model_name, "not found in catalog DB");
            return false;
        }
        Err(e) => {
            tracing::warn!(model = model_name, error = %e, "catalog DB error");
            return false;
        }
    };

    let synth = {
        let config = state.config.read().await;
        let models_dir = config.resolved_models_dir();
        match synthesize_catalog_config(&row, &models_dir, &config) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(model = model_name, error = %e, "synthesize failed");
                return false;
            }
        }
    };

    let mut config = state.config.write().await;
    config.models.insert(model_name.to_string(), synth);
    true
}

/// Return `ModelInfoExtended` entries for every installed Civitai single-file
/// checkpoint in the catalog DB that isn't already covered by a manifest.
fn installed_catalog_models(
    state: &AppState,
    config: &mold_core::Config,
    models_dir: &std::path::Path,
    loaded_model: Option<&str>,
    engine_is_loaded: bool,
) -> Vec<ModelInfoExtended> {
    let params = mold_db::catalog::ListParams {
        kind: Some("checkpoint".to_string()),
        source: Some("civitai".to_string()),
        include_nsfw: true,
        limit: 1000,
        ..Default::default()
    };
    let rows = match state.catalog_db.catalog_list(&params) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "failed to query catalog for installed models");
            return vec![];
        }
    };

    let mut out = Vec::new();
    for row in rows {
        if row.bundling != "single-file" {
            continue;
        }
        // Skip entries already in the manifest (they're covered by build_model_catalog).
        if mold_core::manifest::find_manifest_by_hf_repo(&row.source_id).is_some() {
            continue;
        }
        let recipe =
            match serde_json::from_str::<mold_catalog::entry::DownloadRecipe>(&row.download_recipe)
            {
                Ok(r) => r,
                Err(_) => continue,
            };
        let (author, name) = match row.source_id.split_once('/') {
            Some((a, n)) => (a, n),
            None => ("", row.source_id.as_str()),
        };
        let rendered_dests: Vec<String> = recipe
            .files
            .iter()
            .map(|f| mold_catalog::entry::render_recipe_dest(&f.dest, &row.family, author, name))
            .collect();
        let files: Vec<mold_core::download::RecipeFetchFile<'_>> = recipe
            .files
            .iter()
            .zip(rendered_dests.iter())
            .map(|(f, dest)| mold_core::download::RecipeFetchFile {
                url: f.url.as_str(),
                dest: dest.as_str(),
                sha256: f.sha256.as_deref(),
                size_bytes: f.size_bytes,
            })
            .collect();
        if !mold_core::download::catalog_entry_installed(models_dir, &row.id, &files) {
            continue;
        }

        let size_gb = row
            .size_bytes
            .map(|b| b as f32 / 1_000_000_000.0)
            .unwrap_or(0.0);

        // Use defaults from a visible manifest in the same family, fall back to
        // family-specific constants.
        let (w, h, steps, guidance) = mold_core::manifest::visible_manifests()
            .find(|m| m.family == row.family)
            .map(|m| {
                let cfg = config.resolved_model_config(&m.name);
                (
                    cfg.effective_width(config),
                    cfg.effective_height(config),
                    cfg.effective_steps(config),
                    cfg.effective_guidance(),
                )
            })
            .unwrap_or_else(|| match row.family.as_str() {
                "ltx-video" | "ltx2" => (768, 512, 25, 3.5),
                "sdxl" => (1024, 1024, 20, 7.5),
                "sd15" => (512, 512, 20, 7.5),
                "flux" => (1024, 1024, 20, 3.5),
                "flux2" => (512, 512, 4, 0.0),
                _ => (1024, 1024, 20, 3.5),
            });

        let description = match &row.author {
            Some(a) if !a.is_empty() => format!("{} by {a}", row.name),
            _ => row.name.clone(),
        };

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
                name: row.id.clone(),
                family: row.family.clone(),
                size_gb,
                is_loaded: loaded_model.is_some_and(|n| engine_is_loaded && n == row.id),
                last_used: None,
                hf_repo: String::new(),
            },
            disk_usage_bytes: row.size_bytes.map(|b| b as u64),
            remaining_download_bytes: Some(0),
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

    // Catalog bridge: synthesize config for installed cv:* / hf:* entries so
    // the web UI can generate with models downloaded from the catalog.
    if looks_like_catalog_id(model_name) {
        if install_catalog_model(state, model_name).await {
            let config = state.config.read().await;
            if let Some(paths) = ModelPaths::resolve(model_name, &config) {
                return Ok(Some(paths));
            }
        }
        return Err(ApiError::not_found(format!(
            "catalog model '{model_name}' is not installed. Download it from the catalog first."
        )));
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

/// Ensure the requested model is loaded on GPU and ready for inference.
///
/// Checks the model cache: if already loaded, just touches the LRU order.
/// If cached but unloaded, reloads it. If not in cache, creates a new engine.
pub(crate) async fn ensure_model_ready(
    state: &AppState,
    model_name: &str,
    progress: Option<EngineProgressCallback>,
) -> Result<(), ApiError> {
    let _guard = state.model_load_lock.lock().await;

    // Fast path: model is in cache and loaded.
    {
        let mut cache = state.model_cache.lock().await;
        // Grab active model's VRAM before mutable borrow via get_mut.
        let active_vram = cache.active_vram_bytes();
        if let Some(entry) = cache.get_mut(model_name) {
            if entry.residency == ModelResidency::Gpu {
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

            // Cached but not on GPU (Parked) — need to reload.
            // MPS memory guard: check before unloading the active model.
            // Include the active model's footprint as reclaimable memory.
            if let Some(paths) = entry.engine.model_paths() {
                preflight_memory_guard(model_name, paths, active_vram, 0)?;
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
                // Legacy no-worker path only: hardcoded ordinal 0 is safe here
                // because `state.model_load_lock` (taken above) is the only
                // lock protecting GPU 0's primary context on this path — the
                // GpuPool path uses `worker.model_load_lock` and
                // `reclaim_gpu_memory(worker.gpu.ordinal)` via `gpu_worker`.
                mold_inference::reclaim_gpu_memory(0);
            }

            // Take the engine out of cache to load in spawn_blocking. Using
            // `take()` (not `remove()`) keeps the model name in the cache's
            // `in_flight` set so concurrent `check_model_available` calls
            // still see it as logically cached during the load window.
            let cached = cache.take(model_name).ok_or_else(|| {
                ApiError::internal(format!("cache race: model '{model_name}' vanished"))
            })?;
            drop(cache);

            let mut engine = cached.engine;

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
        Some(paths) => create_and_load_engine(state, model_name, paths, progress).await,
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

    {
        let config = refresh_config(state).await;
        if config.manifest_model_is_downloaded(model) {
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
        if upscaler.is_some() {
            *upscaler = None;
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
            // Legacy no-worker path only: hardcoded ordinal 0 is safe here
            // because `state.model_load_lock` (taken above) is the only
            // lock protecting GPU 0's primary context on this path — the
            // GpuPool path uses `worker.model_load_lock` and
            // `reclaim_gpu_memory(worker.gpu.ordinal)` via `gpu_worker`.
            mold_inference::reclaim_gpu_memory(0);
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
) -> Result<(), ApiError> {
    // MPS memory guard: reject before unloading current model so it stays operational.
    // Include the active model's footprint as reclaimable memory.
    let active_vram = {
        let cache = state.model_cache.lock().await;
        cache.active_vram_bytes()
    };
    preflight_memory_guard(model_name, &paths, active_vram, 0)?;

    // Unload the current active model to free GPU memory.
    // Only reclaim GPU memory if there was an active model — calling
    // reclaim_gpu_memory() (CUDA primary context reset) when nothing was
    // loaded is unnecessary and may misbehave on some driver versions.
    let had_active = {
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
        result.is_some()
    };
    if had_active {
        // Legacy no-worker path only: hardcoded ordinal 0 is safe here
        // because `state.model_load_lock` (taken above) is the only
        // lock protecting GPU 0's primary context on this path — the
        // GpuPool path uses `worker.model_load_lock` and
        // `reclaim_gpu_memory(worker.gpu.ordinal)` via `gpu_worker`.
        mold_inference::reclaim_gpu_memory(0);
    }

    let config = state.config.read().await;
    let offload = std::env::var("MOLD_OFFLOAD").is_ok_and(|v| v == "1");
    let mut new_engine = mold_inference::create_engine_with_pool(
        model_name.to_string(),
        paths,
        &config,
        mold_inference::LoadStrategy::Eager,
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
        let result = preflight_memory_guard_with_available("swap-test", &paths, 10 * GB, 8 * GB);
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
        let result = preflight_memory_guard_with_available("too-big", &paths, 5 * GB, 8 * GB);
        assert!(
            result.is_err(),
            "expected oversized model to be rejected, got Ok"
        );
    }

    #[test]
    fn memory_guard_ok_when_plenty_of_memory() {
        assert!(check_model_memory_budget("test-model", 5 * GB, 20 * GB).is_ok());
    }

    #[test]
    fn memory_guard_rejects_over_90pct() {
        let result = check_model_memory_budget("flux-dev:bf16", 19 * GB, 20 * GB);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, "INSUFFICIENT_MEMORY");
        assert!(err.error.contains("flux-dev:bf16"));
        assert!(err.error.contains("available"));
    }

    #[test]
    fn memory_guard_ok_at_90pct_boundary() {
        // 18 GB peak, 20 GB available → 90% exactly → should pass
        assert!(check_model_memory_budget("test", 18 * GB, 20 * GB).is_ok());
    }

    #[test]
    fn memory_guard_ok_in_warn_zone() {
        // 17 GB peak, 20 GB available → 85% → passes but would warn
        assert!(check_model_memory_budget("test", 17 * GB, 20 * GB).is_ok());
    }

    #[test]
    fn memory_guard_ok_below_warn_zone() {
        // 15 GB peak, 20 GB available → 75% → no warn, no error
        assert!(check_model_memory_budget("test", 15 * GB, 20 * GB).is_ok());
    }

    #[test]
    fn memory_guard_rejects_tiny_available() {
        // Model larger than total available
        let result = check_model_memory_budget("huge-model", 30 * GB, 16 * GB);
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
        assert!(check_model_memory_budget("swap-target", 15 * GB, effective).is_ok());
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
        assert!(check_model_memory_budget("too-large", 15 * GB, effective).is_err());
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

    /// Regression: a quantized FLUX-shaped model should fit on a 24 GB card
    /// when the sibling model is unloaded and the context reset, even though
    /// the Eager (sum) peak would have been ~24 GB and tripped the 90 %
    /// hard limit.
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
        // Free 4 GB on a 24 GB card with an 18 GB sibling about to be
        // reclaimed → effective_available passed in by the outer guard
        // is total_vram = 24 GB on CUDA, but we test the inner directly with
        // the Sequential strategy in mind.
        let result = preflight_memory_guard_with_available("flux-dev:q8", &paths, 0, 24 * GB);
        assert!(
            result.is_ok(),
            "quantized FLUX must fit on a 24 GB card under the Sequential \
             peak estimate (drop-and-reload encoders), got {result:?}"
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
}
