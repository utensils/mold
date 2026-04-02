use std::sync::Arc;

use mold_core::{build_model_catalog, ModelInfoExtended, ModelPaths};

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
    let hard_limit = available_bytes / 10 * 9; // 90%
    if peak_bytes > hard_limit {
        return Err(ApiError::insufficient_memory(format!(
            "model '{}' needs ~{:.1} GB but only ~{:.1} GB available. \
             Close other applications, unload the current model, or use a smaller variant.",
            model_name,
            peak_bytes as f64 / 1_000_000_000.0,
            available_bytes as f64 / 1_000_000_000.0,
        )));
    }

    let warn_limit = available_bytes / 10 * 8; // 80%
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

/// On macOS (MPS/unified memory), check whether estimated peak memory fits
/// before committing to a model load. No-op on CUDA or non-macOS.
///
/// `active_vram_bytes` is the footprint of the currently GPU-resident model
/// that will be unloaded before loading the new one. This memory will become
/// available, so we add it to the budget to avoid false rejections during
/// model swaps.
fn preflight_memory_guard(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
) -> Result<(), ApiError> {
    let available = match mold_inference::device::available_system_memory_bytes() {
        Some(a) if a > 0 => a,
        _ => return Ok(()), // Non-macOS or can't query — skip
    };

    let peak =
        mold_inference::device::estimate_peak_memory(paths, mold_inference::LoadStrategy::Eager);

    // The active model will be unloaded before loading the new one,
    // so its footprint becomes available memory.
    let effective_available = available.saturating_add(active_vram_bytes);

    check_model_memory_budget(model_name, peak, effective_available)
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
    let snapshot = state.engine_snapshot.read().await.clone();

    let config = refresh_config(state).await;
    build_model_catalog(&config, snapshot.model_name.as_deref(), snapshot.is_loaded)
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

    // Check the snapshot as a fallback — it retains the model name even
    // while the engine is temporarily taken out during loading.
    {
        let snapshot = state.engine_snapshot.read().await;
        if snapshot.model_name.as_deref() == Some(model_name) {
            return Ok(None);
        }
    }

    let paths = {
        let config = state.config.read().await;
        ModelPaths::resolve(model_name, &config)
    };
    if let Some(paths) = paths {
        return Ok(Some(paths));
    }

    {
        let current = state.config.read().await.clone();
        let fresh_config = current.reload_from_disk_preserving_runtime();
        if let Some(paths) = ModelPaths::resolve(model_name, &fresh_config) {
            let mut config = state.config.write().await;
            *config = fresh_config;
            return Ok(Some(paths));
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

            // Cached but not on GPU (Unloaded or Parked) — need to reload.
            // MPS memory guard: check before unloading the active model.
            // Include the active model's footprint as reclaimable memory.
            if let Some(paths) = entry.engine.model_paths() {
                preflight_memory_guard(model_name, paths, active_vram)?;
            }

            // Parked engines retain tokenizers/caches for faster reload.
            // First unload the currently active model (if any) to free VRAM.
            if let Some(active_name) = cache.unload_active() {
                tracing::info!(
                    from = %active_name,
                    to = %model_name,
                    "unloaded active model to reload cached model"
                );
                mold_inference::reclaim_gpu_memory();
            }

            // Take the engine out of cache to load in spawn_blocking.
            let mut engine = cache.remove(model_name).unwrap();
            drop(cache);

            if let Some(callback) = progress.clone() {
                engine.set_on_progress(Box::new(move |event| {
                    callback(event);
                }));
            } else {
                engine.clear_on_progress();
            }

            let model_log = model_name.to_string();
            let result = tokio::task::spawn_blocking(move || {
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
            .await
            .map_err(|e| ApiError::internal(format!("model reload task failed: {e}")))?;

            match result {
                Ok(loaded_engine) => {
                    let vram = mold_inference::device::vram_used_estimate();
                    let mut cache = state.model_cache.lock().await;
                    cache.insert(loaded_engine, vram);
                    update_snapshot(state, &cache).await;
                }
                Err((api_err, unloaded_engine)) => {
                    // Put it back as unloaded so cache isn't corrupted.
                    let mut cache = state.model_cache.lock().await;
                    cache.insert(unloaded_engine, 0);
                    return Err(api_err);
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
        if ModelPaths::resolve(model, &config).is_some() {
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
    let mut cache = state.model_cache.lock().await;
    match cache.unload_active() {
        Some(name) => {
            update_snapshot(state, &cache).await;
            drop(cache);
            mold_inference::reclaim_gpu_memory();
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
    preflight_memory_guard(model_name, &paths, active_vram)?;

    // Unload the current active model to free GPU memory.
    // Only reclaim GPU memory if there was an active model — calling
    // reclaim_gpu_memory() (CUDA primary context reset) when nothing was
    // loaded is unnecessary and may misbehave on some driver versions.
    let had_active = {
        let mut cache = state.model_cache.lock().await;
        let result = cache.unload_active();
        if let Some(ref name) = result {
            tracing::info!(
                from = %name,
                to = %model_name,
                "unloading active model before loading new one"
            );
        }
        update_snapshot(state, &cache).await;
        result.is_some()
    };
    if had_active {
        mold_inference::reclaim_gpu_memory();
    }

    let config = state.config.read().await;
    let offload = std::env::var("MOLD_OFFLOAD").is_ok_and(|v| v == "1");
    let mut new_engine = mold_inference::create_engine_with_pool(
        model_name.to_string(),
        paths,
        &config,
        mold_inference::LoadStrategy::Eager,
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

    let vram = mold_inference::device::vram_used_estimate();
    let mut cache = state.model_cache.lock().await;
    // Evicted engine (if any) is dropped here, freeing its resources.
    let _evicted = cache.insert(new_engine, vram);
    update_snapshot(state, &cache).await;
    drop(cache);

    Ok(())
}

/// Synchronize the engine snapshot with the current cache state.
async fn update_snapshot(state: &AppState, cache: &crate::model_cache::ModelCache) {
    let mut snapshot = state.engine_snapshot.write().await;
    snapshot.model_name = cache.active_model().map(|s| s.to_string());
    snapshot.is_loaded = cache.active_model().is_some();
    snapshot.cached_models = cache.cached_model_names();
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

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
}
