use mold_core::{
    classify_generate_error, GenerateRequest, GenerateResponse, GenerateServerAction, LoraWeight,
    MoldClient, SseProgressEvent,
};
use tokio::sync::mpsc;

use crate::app::{BackgroundEvent, GenerateParams};

/// Run a generation request — tries remote first, falls back to local on connection error.
pub async fn run_generation(
    server_url: Option<String>,
    params: GenerateParams,
    prompt: String,
    negative_prompt: Option<String>,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    if params.local_mode {
        run_local_generation(params, prompt, negative_prompt, tx).await;
        return;
    }

    if let Some(url) = server_url {
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
                    Err(e) => {
                        match classify_generate_error(&e) {
                            GenerateServerAction::FallbackLocal => {
                                let _ =
                                    tx.send(BackgroundEvent::Progress(SseProgressEvent::Info {
                                        message: "Server unreachable, using local inference"
                                            .to_string(),
                                    }));
                                // Fall through to local
                            }
                            _ => {
                                let _ = tx.send(BackgroundEvent::Error(format!(
                                    "Generation failed: {e}"
                                )));
                                return;
                            }
                        }
                    }
                }
            }
            Err(e) => {
                match classify_generate_error(&e) {
                    GenerateServerAction::FallbackLocal => {
                        let _ = tx.send(BackgroundEvent::Progress(SseProgressEvent::Info {
                            message: "Server unreachable, using local inference".to_string(),
                        }));
                        // Fall through to local
                    }
                    _ => {
                        let _ = tx.send(BackgroundEvent::Error(format!("Generation failed: {e}")));
                        return;
                    }
                }
            }
        }
    }

    // Fall through: no server URL or server unreachable — try local
    run_local_generation(params, prompt, negative_prompt, tx).await;
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

    let config = Config::load_or_default();
    let model_name = params.model.clone();
    let model_paths = match ModelPaths::resolve(&model_name, &config) {
        Some(paths) => paths,
        None => {
            let _ = tx.send(BackgroundEvent::Error(format!(
                "Model '{}' not downloaded. Run: mold pull {}",
                model_name, model_name
            )));
            return;
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
        embed_metadata: Some(true),
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
