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
            if e.model_name() == model_name && e.is_loaded() {
                return Ok(None);
            }
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
                    tracing::info!(model = %model_name, "loading existing engine...");
                    engine.load().map_err(|e| {
                        tracing::error!("model load failed: {e:#}");
                        ApiError::internal(format!("model load error: {e}"))
                    })?;
                    // Update snapshot while holding engine lock to prevent
                    // unload_model from racing and leaving stale state.
                    let mut snapshot = state.engine_snapshot.write().await;
                    snapshot.model_name = Some(model_name.to_string());
                    snapshot.is_loaded = true;
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
    match engine.as_mut() {
        Some(e) if e.is_loaded() => {
            let name = e.model_name().to_string();
            e.unload();
            // Update snapshot while still holding the engine lock to avoid
            // a window where engine is unloaded but snapshot still says loaded.
            let mut snapshot = state.engine_snapshot.write().await;
            snapshot.model_name = Some(name.clone());
            snapshot.is_loaded = false;
            drop(snapshot);
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
    let mut new_engine = mold_inference::create_engine(
        model_name.to_string(),
        paths,
        &config,
        mold_inference::LoadStrategy::Eager,
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

    // Update snapshot only after the engine swap — not before the lock is acquired,
    // otherwise /api/status reports "not loaded" while the old model is still serving.
    if let Some(ref mut e) = *engine {
        if !e.is_loaded() {
            tracing::info!(model = %model_name, "loading model...");
            e.load().map_err(|e| {
                tracing::error!("model load failed: {e:#}");
                ApiError::internal(format!("model load error: {e}"))
            })?;
        }
    }

    // Update snapshot while holding engine lock to prevent unload_model
    // from racing and leaving snapshot in a stale is_loaded=true state.
    {
        let mut snapshot = state.engine_snapshot.write().await;
        snapshot.model_name = Some(model_name.to_string());
        snapshot.is_loaded = true;
    }
    drop(engine);

    Ok(())
}
