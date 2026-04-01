use mold_core::{
    GenerateRequest, GenerateResponse, LoraWeight, MoldClient, SseProgressEvent,
};
use tokio::sync::mpsc;

use crate::app::{BackgroundEvent, GenerateParams};

/// Run a generation request (remote or local) and send progress/results to the channel.
pub async fn run_generation(
    server_url: Option<String>,
    params: GenerateParams,
    prompt: String,
    negative_prompt: Option<String>,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    if params.local_mode {
        run_local_generation(params, prompt, negative_prompt, tx).await;
    } else if let Some(url) = server_url {
        let client = MoldClient::new(&url);
        run_remote_generation(client, params, prompt, negative_prompt, tx).await;
    } else {
        // No server configured and not in local mode
        let _ = tx.send(BackgroundEvent::Error(
            "No server configured. Set MOLD_HOST or use local mode.".to_string(),
        ));
    }
}

async fn run_remote_generation(
    client: MoldClient,
    params: GenerateParams,
    prompt: String,
    negative_prompt: Option<String>,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    let req = build_request(params, prompt, negative_prompt);

    // Create a channel for SSE progress events
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<SseProgressEvent>();

    // Forward progress events to the main TUI channel
    let tx_progress = tx.clone();
    tokio::spawn(async move {
        while let Some(event) = progress_rx.recv().await {
            let _ = tx_progress.send(BackgroundEvent::Progress(event));
        }
    });

    // Run the streaming generation
    match client.generate_stream(&req, progress_tx).await {
        Ok(Some(response)) => {
            let _ = tx.send(BackgroundEvent::GenerationComplete(Box::new(response)));
        }
        Ok(None) => {
            let _ = tx.send(BackgroundEvent::Error(
                "Generation completed but no response received".to_string(),
            ));
        }
        Err(e) => {
            let _ = tx.send(BackgroundEvent::Error(format!("Generation failed: {e}")));
        }
    }
}

async fn run_local_generation(
    params: GenerateParams,
    prompt: String,
    negative_prompt: Option<String>,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    use mold_core::{Config, ModelPaths};
    use mold_inference::{LoadStrategy, create_engine};
    use mold_inference::progress::ProgressEvent;

    let config = Config::load_or_default();
    let model_name = params.model.clone();
    let model_paths = match ModelPaths::resolve(&model_name, &config) {
        Some(paths) => paths,
        None => {
            let _ = tx.send(BackgroundEvent::Error(format!(
                "Cannot resolve model paths for '{model_name}'. Run: mold pull {model_name}"
            )));
            return;
        }
    };

    let offload = params.offload;
    let req = build_request(params, prompt, negative_prompt);

    let tx_clone = tx.clone();

    let result = tokio::task::spawn_blocking(move || {
        let mut engine = create_engine(
            model_name,
            model_paths,
            &config,
            LoadStrategy::Sequential,
            offload,
        )?;

        // Install progress callback
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
    params: GenerateParams,
    prompt: String,
    negative_prompt: Option<String>,
) -> GenerateRequest {
    let lora = params.lora_path.map(|path| LoraWeight {
        path,
        scale: params.lora_scale,
    });

    // Read source image bytes if path provided
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
        prompt,
        negative_prompt,
        model: params.model,
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
        control_model: params.control_model,
        control_scale: params.control_scale,
        expand: if params.expand { Some(true) } else { None },
        original_prompt: None,
        lora,
    }
}
