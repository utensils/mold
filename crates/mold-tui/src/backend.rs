use std::sync::Arc;

use mold_core::{
    classify_generate_error, download::DownloadProgressEvent, ChainRequest, GenerateRequest,
    GenerateResponse, GenerateServerAction, LoraWeight, MoldClient, SseProgressEvent,
};
use tokio::sync::mpsc;

use crate::app::{BackgroundEvent, GenerateParams, InferenceMode};

/// Run a generation request — tries remote first, falls back to local on connection error.
/// When batch > 1, loops client-side with `batch_size=1` per iteration (matching CLI behavior).
pub async fn run_generation(
    server_url: Option<String>,
    params: GenerateParams,
    prompt: String,
    negative_prompt: Option<String>,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    let batch = params.batch;
    let base_seed = params.seed;

    for i in 0..batch {
        let mut iter_params = params.clone();
        iter_params.batch = 1;

        // Increment seed for each batch iteration (first uses original seed)
        if i > 0 {
            iter_params.seed = base_seed.map(|s| s.wrapping_add(i as u64));
        }

        if batch > 1 {
            let _ = tx.send(BackgroundEvent::Progress(SseProgressEvent::Info {
                message: format!("Generating image {}/{batch}...", i + 1),
            }));
        }

        if iter_params.inference_mode == InferenceMode::Local {
            run_local_generation_single(iter_params, prompt.clone(), negative_prompt.clone(), &tx)
                .await;
        } else {
            let effective_url = iter_params.host.clone().or_else(|| server_url.clone());

            let mut fell_through = false;
            if let Some(url) = effective_url {
                let client = MoldClient::new(&url);
                let req = build_request(&iter_params, &prompt, &negative_prompt);

                match try_server_generate(&client, &req, &tx).await {
                    ServerResult::Done => {}
                    ServerResult::FallbackLocal => {
                        fell_through = true;
                    }
                    ServerResult::Error(e) => {
                        let _ = tx.send(BackgroundEvent::Error(e));
                        return;
                    }
                }
            } else {
                fell_through = true;
            }

            if fell_through {
                if iter_params.inference_mode == InferenceMode::Remote {
                    let _ = tx.send(BackgroundEvent::Error(
                        "Server unreachable and mode is set to 'remote'. Switch to 'auto' or 'local'."
                            .to_string(),
                    ));
                    return;
                }
                run_local_generation_single(
                    iter_params,
                    prompt.clone(),
                    negative_prompt.clone(),
                    &tx,
                )
                .await;
            }
        }
    }
}

/// Run a chain generation request via the server's `/api/generate/chain/stream`
/// endpoint. Chain generation is server-only — there is no local fallback.
pub async fn run_chain_generation(
    server_url: Option<String>,
    req: ChainRequest,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    let Some(url) = server_url else {
        let _ = tx.send(BackgroundEvent::ChainError(
            "chain generation requires a running mold server".into(),
        ));
        return;
    };

    let client = MoldClient::new(&url);
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel();

    let bg_tx = tx.clone();
    let forward_handle = tokio::spawn(async move {
        while let Some(event) = progress_rx.recv().await {
            let _ = bg_tx.send(BackgroundEvent::ChainProgress(event));
        }
    });

    match client.generate_chain_stream(&req, progress_tx).await {
        Ok(Some(response)) => {
            let _ = tx.send(BackgroundEvent::ChainComplete {
                response: Box::new(response),
            });
        }
        Ok(None) => {
            let _ = tx.send(BackgroundEvent::ChainError(
                "server does not support chain generation (404)".into(),
            ));
        }
        Err(e) => {
            let _ = tx.send(BackgroundEvent::ChainError(format!("{e}")));
        }
    }

    forward_handle.abort();
}

/// Run a single local generation and send the result via `tx`.
async fn run_local_generation_single(
    params: GenerateParams,
    prompt: String,
    negative_prompt: Option<String>,
    tx: &mpsc::UnboundedSender<BackgroundEvent>,
) {
    run_local_generation(params, prompt, negative_prompt, tx.clone()).await;
}

/// Pull a model with progress reporting to the TUI.
pub async fn auto_pull_model(
    model_name: &str,
    tx: &mpsc::UnboundedSender<BackgroundEvent>,
) -> Result<mold_core::Config, String> {
    use mold_core::download::{self, PullOptions};
    use mold_core::manifest::{compute_download_size, find_manifest};

    let manifest = match find_manifest(model_name) {
        Some(m) => m,
        None => {
            return Err(format!(
                "Unknown model '{}'. Run 'mold list' to see available models.",
                model_name
            ));
        }
    };

    let (_total_bytes, remaining_bytes) = compute_download_size(manifest);
    let remaining_gb = remaining_bytes as f64 / 1_073_741_824.0;

    let _ = tx.send(BackgroundEvent::Progress(SseProgressEvent::Info {
        message: format!(
            "Model '{}' not found locally, pulling ({:.1}GB)...",
            model_name, remaining_gb
        ),
    }));

    // Create a callback that converts download events to TUI progress events
    let tx_dl = tx.clone();
    let callback: mold_core::download::DownloadProgressCallback =
        Arc::new(move |event: DownloadProgressEvent| {
            let sse = match event {
                DownloadProgressEvent::FileStart {
                    filename,
                    file_index,
                    total_files,
                    size_bytes,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms,
                } => SseProgressEvent::DownloadProgress {
                    filename,
                    file_index,
                    total_files,
                    bytes_downloaded: 0,
                    bytes_total: size_bytes,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms,
                },
                DownloadProgressEvent::FileProgress {
                    filename,
                    file_index,
                    bytes_downloaded,
                    bytes_total,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms,
                } => SseProgressEvent::DownloadProgress {
                    filename,
                    file_index,
                    total_files: 0, // not available here but okay
                    bytes_downloaded,
                    bytes_total,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms,
                },
                DownloadProgressEvent::FileDone {
                    filename,
                    file_index,
                    total_files,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms,
                } => SseProgressEvent::DownloadDone {
                    filename,
                    file_index,
                    total_files,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms,
                },
                DownloadProgressEvent::Status { message } => SseProgressEvent::Info { message },
            };
            let _ = tx_dl.send(BackgroundEvent::Progress(sse));
        });

    match download::pull_and_configure_with_callback(model_name, callback, &PullOptions::default())
        .await
    {
        Ok((config, _paths)) => {
            let _ = tx.send(BackgroundEvent::Progress(SseProgressEvent::PullComplete {
                model: model_name.to_string(),
            }));
            Ok(config)
        }
        Err(e) => Err(format!("Failed to pull '{}': {}", model_name, e)),
    }
}

enum ServerResult {
    Done,
    FallbackLocal,
    Error(String),
}

/// Try generating via the server. If the server says the model isn't downloaded,
/// auto-pull it and retry once.
async fn try_server_generate(
    client: &MoldClient,
    req: &GenerateRequest,
    tx: &mpsc::UnboundedSender<BackgroundEvent>,
) -> ServerResult {
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<SseProgressEvent>();

    let tx_progress = tx.clone();
    tokio::spawn(async move {
        while let Some(event) = progress_rx.recv().await {
            let _ = tx_progress.send(BackgroundEvent::Progress(event));
        }
    });

    match try_server_generate_once(client, req, progress_tx).await {
        Ok(response) => {
            let _ = tx.send(BackgroundEvent::GenerationComplete {
                response: Box::new(response),
                from_local: false,
            });
            ServerResult::Done
        }
        Err(e) => match classify_generate_error(&e) {
            GenerateServerAction::FallbackLocal => ServerResult::FallbackLocal,
            GenerateServerAction::PullModelAndRetry => {
                // Auto-pull the model via the server, then retry
                let _ = tx.send(BackgroundEvent::Progress(SseProgressEvent::Info {
                    message: format!(
                        "Model '{}' not downloaded, pulling via server...",
                        req.model
                    ),
                }));

                let (pull_tx, mut pull_rx) = mpsc::unbounded_channel::<SseProgressEvent>();
                let tx_pull = tx.clone();
                tokio::spawn(async move {
                    while let Some(event) = pull_rx.recv().await {
                        let _ = tx_pull.send(BackgroundEvent::Progress(event));
                    }
                });

                if let Err(pull_err) = client.pull_model_stream(&req.model, pull_tx).await {
                    return ServerResult::Error(format!(
                        "Failed to pull '{}': {pull_err}",
                        req.model
                    ));
                }

                // Retry generation after pull
                let (retry_tx, mut retry_rx) = mpsc::unbounded_channel::<SseProgressEvent>();
                let tx_retry = tx.clone();
                tokio::spawn(async move {
                    while let Some(event) = retry_rx.recv().await {
                        let _ = tx_retry.send(BackgroundEvent::Progress(event));
                    }
                });

                match try_server_generate_once(client, req, retry_tx).await {
                    Ok(response) => {
                        let _ = tx.send(BackgroundEvent::GenerationComplete {
                            response: Box::new(response),
                            from_local: false,
                        });
                        ServerResult::Done
                    }
                    Err(retry_err) => match classify_generate_error(&retry_err) {
                        GenerateServerAction::FallbackLocal => ServerResult::FallbackLocal,
                        _ => ServerResult::Error(format!("Generation failed: {retry_err}")),
                    },
                }
            }
            GenerateServerAction::SurfaceError => {
                ServerResult::Error(format!("Generation failed: {e}"))
            }
        },
    }
}

/// Single attempt to generate via server (SSE with blocking fallback).
async fn try_server_generate_once(
    client: &MoldClient,
    req: &GenerateRequest,
    progress_tx: mpsc::UnboundedSender<SseProgressEvent>,
) -> Result<GenerateResponse, anyhow::Error> {
    match client.generate_stream(req, progress_tx).await {
        Ok(Some(response)) => Ok(response),
        Ok(None) => {
            // Server doesn't support SSE — try blocking API
            client.generate(req.clone()).await
        }
        Err(e) => Err(e),
    }
}

async fn run_local_generation(
    params: GenerateParams,
    prompt: String,
    negative_prompt: Option<String>,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    use mold_core::{Config, ModelPaths};
    use mold_inference::progress::ProgressEvent;
    use mold_inference::{create_engine, LoadStrategy};

    let mut config = Config::load_or_default();
    let model_name = params.model.clone();

    // Resolve model paths — auto-pull if not downloaded
    let model_paths = match ModelPaths::resolve(&model_name, &config) {
        Some(paths) => paths,
        None => {
            // Try auto-pull
            match auto_pull_model(&model_name, &tx).await {
                Ok(updated_config) => {
                    config = updated_config;
                    match ModelPaths::resolve(&model_name, &config) {
                        Some(paths) => paths,
                        None => {
                            let _ = tx.send(BackgroundEvent::Error(format!(
                                "Model '{}' was pulled but paths could not be resolved",
                                model_name
                            )));
                            return;
                        }
                    }
                }
                Err(msg) => {
                    let _ = tx.send(BackgroundEvent::Error(msg));
                    return;
                }
            }
        }
    };

    let offload = params.offload;
    let req = build_request(&params, &prompt, &negative_prompt);

    let tx_clone = tx.clone();

    let result = tokio::task::spawn_blocking(move || {
        let mut engine = create_engine(
            model_name,
            model_paths,
            &config,
            LoadStrategy::Sequential,
            0,
            offload,
        )?;

        let tx_progress = tx_clone.clone();
        engine.set_on_progress(Box::new(move |event: ProgressEvent| {
            let sse_event: SseProgressEvent = event.into();
            let _ = tx_progress.send(BackgroundEvent::Progress(sse_event));
        }));

        let response = engine.generate(&req)?;
        engine.clear_on_progress();
        Ok::<GenerateResponse, anyhow::Error>(response)
    })
    .await;

    match result {
        Ok(Ok(response)) => {
            // Local path — covers the forced `InferenceMode::Local` case
            // AND the Auto-mode fallback after a remote server becomes
            // unreachable. Marking `from_local: true` lets the TUI save
            // the file locally instead of deferring to a server that
            // never produced it.
            let _ = tx.send(BackgroundEvent::GenerationComplete {
                response: Box::new(response),
                from_local: true,
            });
        }
        Ok(Err(e)) => {
            let _ = tx.send(BackgroundEvent::Error(format!("Generation failed: {e}")));
        }
        Err(e) => {
            let _ = tx.send(BackgroundEvent::Error(format!("Task panicked: {e}")));
        }
    }
}

fn build_request(
    params: &GenerateParams,
    prompt: &str,
    negative_prompt: &Option<String>,
) -> GenerateRequest {
    let lora = params.lora_path.as_ref().map(|path| LoraWeight {
        path: path.clone(),
        scale: params.lora_scale,
    });

    let source_image = params
        .source_image_path
        .as_ref()
        .and_then(|p| std::fs::read(p).ok());

    let mask_image = params
        .mask_image_path
        .as_ref()
        .and_then(|p| std::fs::read(p).ok());

    let control_image = params
        .control_image_path
        .as_ref()
        .and_then(|p| std::fs::read(p).ok());

    let family = mold_core::manifest::find_manifest(&params.model)
        .map(|m| m.family.as_str().to_string())
        .unwrap_or_default();
    let (edit_images, source_image, strength, mask_image) = if family == "qwen-image-edit" {
        (
            source_image.clone().map(|image| vec![image]),
            None,
            0.75,
            None,
        )
    } else {
        (None, source_image, params.strength, mask_image)
    };

    GenerateRequest {
        prompt: prompt.to_string(),
        negative_prompt: negative_prompt.clone(),
        model: params.model.clone(),
        width: params.width,
        height: params.height,
        steps: params.steps,
        guidance: params.guidance,
        seed: params.seed,
        batch_size: params.batch,
        output_format: params.format,
        embed_metadata: Some(mold_core::Config::load_or_default().effective_embed_metadata(None)),
        scheduler: params.scheduler,
        edit_images,
        source_image,
        strength,
        mask_image,
        control_image,
        control_model: params.control_model.clone(),
        control_scale: params.control_scale,
        expand: if params.expand { Some(true) } else { None },
        original_prompt: None,
        lora,
        frames: Some(params.frames),
        fps: Some(params.fps),
        upscale_model: None,
        gif_preview: true,
        enable_audio: None,
        audio_file: None,
        source_video: None,
        keyframes: None,
        pipeline: None,
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: None,
    }
}

/// Build a map of file_path -> list of model names that reference it.
pub(crate) fn build_ref_counts(
    config: &mold_core::Config,
) -> std::collections::HashMap<String, Vec<String>> {
    let mut refs: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for (model_name, model_config) in &config.models {
        for path in model_config.all_file_paths() {
            refs.entry(path).or_default().push(model_name.clone());
        }
    }
    refs
}

/// Collect hf-hub cache blob paths for a model's unique files so we can delete them
/// to actually reclaim disk space (clean paths are hardlinked from blobs).
fn collect_hf_cache_blob_paths(
    model_name: &str,
    unique_clean_paths: &[(String, u64)],
) -> Vec<std::path::PathBuf> {
    use mold_core::manifest::{find_manifest, storage_path};

    let manifest = match find_manifest(model_name) {
        Some(m) => m,
        None => return Vec::new(),
    };

    let config = mold_core::Config::load_or_default();
    let models_dir = config.resolved_models_dir();
    let cache_dir = models_dir.join(".hf-cache");
    if !cache_dir.is_dir() {
        return Vec::new();
    }

    let unique_set: std::collections::HashSet<String> =
        unique_clean_paths.iter().map(|(p, _)| p.clone()).collect();

    let mut blobs = Vec::new();

    for file in &manifest.files {
        let clean_path = models_dir
            .join(storage_path(manifest, file))
            .to_string_lossy()
            .to_string();
        if !unique_set.contains(&clean_path) {
            continue;
        }

        let repo_dir_name = format!("models--{}", file.hf_repo.replace('/', "--"));
        let repo_dir = cache_dir.join(&repo_dir_name);
        if !repo_dir.is_dir() {
            continue;
        }

        let snapshots_dir = repo_dir.join("snapshots");
        if !snapshots_dir.is_dir() {
            continue;
        }

        if let Ok(revisions) = std::fs::read_dir(&snapshots_dir) {
            for rev in revisions.flatten() {
                let snap_file = rev.path().join(&file.hf_filename);
                if snap_file.symlink_metadata().is_ok() {
                    if let Ok(blob) = snap_file.canonicalize() {
                        blobs.push(blob);
                    }
                    blobs.push(snap_file);
                }
            }
        }
    }

    blobs
}

/// Remove a model's files and config entry. Runs on a blocking thread.
pub fn remove_model(model_name: String, tx: mpsc::UnboundedSender<BackgroundEvent>) {
    let mut config = mold_core::Config::load_or_default();

    if !config.models.contains_key(&model_name) {
        let _ = tx.send(BackgroundEvent::ModelRemoveFailed(format!(
            "Model '{}' is not installed",
            model_name
        )));
        return;
    }

    // Build reference counts to identify shared vs unique files
    let ref_counts = build_ref_counts(&config);
    let model_config = config.models.get(&model_name).unwrap();
    let all_paths = model_config.all_file_paths();

    let mut unique_files: Vec<(String, u64)> = Vec::new();

    for path in &all_paths {
        let refs = ref_counts.get(path).cloned().unwrap_or_default();
        let other_refs: Vec<String> = refs.into_iter().filter(|n| n != &model_name).collect();
        if other_refs.is_empty() {
            let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            unique_files.push((path.clone(), size));
        }
    }

    // Delete unique files
    let hf_cache_blobs = collect_hf_cache_blob_paths(&model_name, &unique_files);

    for (path, _) in &unique_files {
        let _ = std::fs::remove_file(path);
    }

    // Delete hf-cache blobs (where actual disk space lives due to hardlinks)
    for blob_path in &hf_cache_blobs {
        let _ = std::fs::remove_file(blob_path);
    }

    // Clean up empty directories left behind by deleted files.
    // Deduplicate parent dirs to avoid redundant remove_dir attempts.
    let mut tried_dirs = std::collections::HashSet::new();
    for (path, _) in &unique_files {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if tried_dirs.insert(parent.to_path_buf()) {
                let _ = std::fs::remove_dir(parent); // only succeeds if empty
            }
        }
    }

    // Remove from config
    config.remove_model(&model_name);
    mold_core::download::remove_pulling_marker(&model_name);

    // Reassign default model if needed
    if config.default_model == model_name {
        let new_default = config
            .models
            .keys()
            .min()
            .cloned()
            .unwrap_or_else(|| "flux2-klein".to_string());
        config.default_model = new_default;
    }

    if let Err(e) = config.save() {
        let _ = tx.send(BackgroundEvent::ModelRemoveFailed(format!(
            "Removed files but failed to save config: {e}"
        )));
        return;
    }

    let _ = tx.send(BackgroundEvent::ModelRemoveComplete(model_name));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_ref_counts_tracks_shared_files() {
        let mut config = mold_core::Config::default();

        let model_a = mold_core::ModelConfig {
            transformer: Some("/models/a/transformer.safetensors".to_string()),
            vae: Some("/models/shared/vae.safetensors".to_string()),
            ..Default::default()
        };

        let model_b = mold_core::ModelConfig {
            transformer: Some("/models/b/transformer.safetensors".to_string()),
            vae: Some("/models/shared/vae.safetensors".to_string()),
            ..Default::default()
        };

        config.models.insert("model-a".to_string(), model_a);
        config.models.insert("model-b".to_string(), model_b);

        let refs = build_ref_counts(&config);

        // Unique files should have exactly one reference
        let a_refs = refs.get("/models/a/transformer.safetensors").unwrap();
        assert_eq!(a_refs.len(), 1);
        assert!(a_refs.contains(&"model-a".to_string()));

        // Shared files should have both models
        let vae_refs = refs.get("/models/shared/vae.safetensors").unwrap();
        assert_eq!(vae_refs.len(), 2);
        assert!(vae_refs.contains(&"model-a".to_string()));
        assert!(vae_refs.contains(&"model-b".to_string()));
    }

    #[test]
    fn build_ref_counts_empty_config() {
        let config = mold_core::Config::default();
        let refs = build_ref_counts(&config);
        assert!(refs.is_empty());
    }

    #[test]
    fn build_request_uses_batch_from_params() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.batch = 4;
        let req = build_request(&params, "test prompt", &None);
        assert_eq!(req.batch_size, 4);
    }

    #[test]
    fn build_request_single_batch_default() {
        let config = mold_core::Config::load_or_default();
        let params = GenerateParams::from_config(&config);
        assert_eq!(params.batch, 1);
        let req = build_request(&params, "test prompt", &None);
        assert_eq!(req.batch_size, 1);
    }
}
