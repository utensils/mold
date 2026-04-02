use std::sync::Arc;

use mold_core::{
    classify_generate_error, download::DownloadProgressEvent, GenerateRequest, GenerateResponse,
    GenerateServerAction, LoraWeight, MoldClient, SseProgressEvent,
};
use tokio::sync::mpsc;

use crate::app::{BackgroundEvent, GenerateParams, InferenceMode};

/// Run a generation request — tries remote first, falls back to local on connection error.
pub async fn run_generation(
    server_url: Option<String>,
    params: GenerateParams,
    prompt: String,
    negative_prompt: Option<String>,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    if params.inference_mode == InferenceMode::Local {
        run_local_generation(params, prompt, negative_prompt, tx).await;
        return;
    }

    // Determine which server URL to use (params.host overrides server_url)
    let effective_url = params.host.clone().or(server_url);

    if let Some(url) = effective_url {
        let client = MoldClient::new(&url);
        let req = build_request(&params, &prompt, &negative_prompt);

        let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<SseProgressEvent>();

        let tx_progress = tx.clone();
        tokio::spawn(async move {
            while let Some(event) = progress_rx.recv().await {
                let _ = tx_progress.send(BackgroundEvent::Progress(event));
            }
        });

        match client.generate_stream(&req, progress_tx).await {
            Ok(Some(response)) => {
                let _ = tx.send(BackgroundEvent::GenerationComplete(Box::new(response)));
                return;
            }
            Ok(None) => {
                // Server doesn't support SSE — try blocking API
                match client.generate(req).await {
                    Ok(response) => {
                        let _ = tx.send(BackgroundEvent::GenerationComplete(Box::new(response)));
                        return;
                    }
                    Err(e) => match classify_generate_error(&e) {
                        GenerateServerAction::FallbackLocal => {
                            let _ = tx.send(BackgroundEvent::Progress(SseProgressEvent::Info {
                                message: "Server unreachable, using local inference".to_string(),
                            }));
                        }
                        _ => {
                            let _ =
                                tx.send(BackgroundEvent::Error(format!("Generation failed: {e}")));
                            return;
                        }
                    },
                }
            }
            Err(e) => match classify_generate_error(&e) {
                GenerateServerAction::FallbackLocal => {
                    let _ = tx.send(BackgroundEvent::Progress(SseProgressEvent::Info {
                        message: "Server unreachable, using local inference".to_string(),
                    }));
                }
                _ => {
                    let _ = tx.send(BackgroundEvent::Error(format!("Generation failed: {e}")));
                    return;
                }
            },
        }
    }

    // Fall through: no server URL or server unreachable
    if params.inference_mode == InferenceMode::Remote {
        let _ = tx.send(BackgroundEvent::Error(
            "Server unreachable and mode is set to 'remote'. Switch to 'auto' or 'local'."
                .to_string(),
        ));
        return;
    }
    run_local_generation(params, prompt, negative_prompt, tx).await;
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
                    size_bytes: _,
                } => SseProgressEvent::Info {
                    message: format!(
                        "[{}/{}] Downloading {}",
                        file_index + 1,
                        total_files,
                        filename
                    ),
                },
                DownloadProgressEvent::FileProgress {
                    filename,
                    file_index,
                    bytes_downloaded,
                    bytes_total,
                } => SseProgressEvent::DownloadProgress {
                    filename,
                    file_index,
                    total_files: 0, // not available here but okay
                    bytes_downloaded,
                    bytes_total,
                },
                DownloadProgressEvent::FileDone {
                    filename,
                    file_index,
                    total_files,
                } => SseProgressEvent::DownloadDone {
                    filename,
                    file_index,
                    total_files,
                },
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
            let _ = tx.send(BackgroundEvent::GenerationComplete(Box::new(response)));
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
        source_image,
        strength: params.strength,
        mask_image,
        control_image,
        control_model: params.control_model.clone(),
        control_scale: params.control_scale,
        expand: if params.expand { Some(true) } else { None },
        original_prompt: None,
        lora,
    }
}
