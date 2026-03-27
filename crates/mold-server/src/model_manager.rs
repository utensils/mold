use std::sync::Arc;

use mold_core::{build_model_catalog, ModelInfoExtended, ModelPaths};

use crate::{routes::ApiError, state::AppState};

pub(crate) type EngineProgressCallback = Arc<dyn Fn(mold_inference::ProgressEvent) + Send + Sync>;
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

pub(crate) async fn check_model_available(
    state: &AppState,
    model_name: &str,
) -> Result<Option<ModelPaths>, ApiError> {
    {
        let engine = state.engine.lock().await;
        if let Some(ref e) = *engine {
            if e.model_name() == model_name {
                // Engine exists for this model (loaded or unloaded) —
                // ensure_model_ready will handle loading if needed.
                return Ok(None);
            }
        }
    }

    // The engine may be temporarily taken out of the slot during loading
    // (ensure_model_ready uses .take() to avoid holding the mutex across
    // spawn_blocking). Check the snapshot as a fallback — it retains the
    // model name even while the engine is being loaded.
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

pub(crate) async fn ensure_model_ready(
    state: &AppState,
    model_name: &str,
    progress: Option<EngineProgressCallback>,
) -> Result<(), ApiError> {
    let _guard = state.model_load_lock.lock().await;
    {
        let mut guard = state.engine.lock().await;
        if let Some(engine) = guard.as_mut() {
            if engine.model_name() == model_name {
                if let Some(callback) = progress.clone() {
                    engine.set_on_progress(Box::new(move |event| {
                        callback(event);
                    }));
                } else {
                    engine.clear_on_progress();
                }
                if !engine.is_loaded() {
                    // Take the engine out so we can load it in spawn_blocking
                    // without holding the tokio mutex across a blocking call.
                    // This keeps the runtime responsive for SSE event delivery
                    // during long operations like FP8→Q8 conversion.
                    let mut taken = guard.take().unwrap();
                    drop(guard);

                    let model_log = model_name.to_string();
                    let result = tokio::task::spawn_blocking(move || {
                        tracing::info!(model = %model_log, "loading existing engine...");
                        if let Err(e) = taken.load() {
                            tracing::error!("model load failed: {e:#}");
                            return Err((
                                ApiError::internal(format!("model load error: {e}")),
                                taken,
                            ));
                        }
                        Ok(taken)
                    })
                    .await
                    .map_err(|e| ApiError::internal(format!("model load task failed: {e}")))?;

                    match result {
                        Ok(loaded) => {
                            let mut guard = state.engine.lock().await;
                            *guard = Some(loaded);
                            let mut snapshot = state.engine_snapshot.write().await;
                            snapshot.model_name = Some(model_name.to_string());
                            snapshot.is_loaded = true;
                        }
                        Err((api_err, unloaded)) => {
                            // Restore the unloaded engine so the next request
                            // can retry without recreating from scratch.
                            let mut guard = state.engine.lock().await;
                            *guard = Some(unloaded);
                            return Err(api_err);
                        }
                    }
                }
                return Ok(());
            }
        }
    }

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

pub(crate) async fn unload_model(state: &AppState) -> String {
    let mut engine = state.engine.lock().await;
    match engine.as_ref() {
        Some(e) if e.is_loaded() => {
            let name = e.model_name().to_string();
            // Drop the entire engine (not just its loaded state) so all GPU
            // resources — tensors, CUDA device handles, cuDNN/cuBLAS workspace
            // buffers — are fully released back to the system.
            *engine = None;
            // Update snapshot while still holding the engine lock to avoid
            // a window where engine is unloaded but snapshot still says loaded.
            let mut snapshot = state.engine_snapshot.write().await;
            snapshot.model_name = None;
            snapshot.is_loaded = false;
            drop(snapshot);

            // Reset the CUDA primary context to reclaim all GPU memory —
            // cuBLAS workspace caches, compiled kernel modules, memory pools.
            // Safe because the engine (and all its Device references) was
            // already dropped above, and we still hold the engine mutex so
            // no concurrent load can begin creating new CUDA objects.
            mold_inference::reclaim_gpu_memory();

            drop(engine);

            tracing::info!(model = %name, "model unloaded via API");
            format!("unloaded {name}")
        }
        _ => "no model loaded".to_string(),
    }
}

async fn create_and_load_engine(
    state: &AppState,
    model_name: &str,
    paths: ModelPaths,
    progress: Option<EngineProgressCallback>,
) -> Result<(), ApiError> {
    let config = state.config.read().await;
    let offload = std::env::var("MOLD_OFFLOAD").is_ok_and(|v| v == "1");
    let mut new_engine = mold_inference::create_engine(
        model_name.to_string(),
        paths,
        &config,
        mold_inference::LoadStrategy::Eager,
        offload,
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

    // Load model weights in a blocking thread to avoid starving the tokio
    // runtime. This is critical for long-running operations like FP8→Q8
    // GGUF conversion (can take minutes) — without spawn_blocking, the
    // blocked worker thread prevents SSE progress events from reaching
    // the client, causing the CLI to show a spinner with no status.
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

    // Place the loaded engine into shared state and update the snapshot
    // while holding the engine lock to prevent unload_model from racing.
    let mut engine = state.engine.lock().await;

    if let Some(ref old) = *engine {
        if old.model_name() != model_name {
            tracing::info!(
                from = %old.model_name(),
                to = %model_name,
                "hot-swapping model"
            );
        }
    }

    *engine = Some(new_engine);

    {
        let mut snapshot = state.engine_snapshot.write().await;
        snapshot.model_name = Some(model_name.to_string());
        snapshot.is_loaded = true;
    }
    drop(engine);

    Ok(())
}
