use std::sync::Arc;

use base64::Engine as _;
use mold_core::{SseCompleteEvent, SseErrorEvent, SseProgressEvent};
use sha2::{Digest, Sha256};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::model_manager;
use crate::state::{
    ActiveGenerationSnapshot, AppState, GenerationJob, GenerationJobResult, SseMessage,
};

/// Convert an inference-crate progress event to an SSE wire event.
fn progress_to_sse(event: mold_inference::ProgressEvent) -> SseProgressEvent {
    event.into()
}

/// Strips backtrace frames from candle error messages.
pub(crate) fn clean_error_message(e: &anyhow::Error) -> String {
    let full = format!("{e}");
    let mut lines: Vec<&str> = Vec::new();
    for line in full.lines() {
        let trimmed = line.trim_start();
        if (trimmed.starts_with("0:") || trimmed.starts_with("1:"))
            && trimmed.len() > 3
            && trimmed
                .as_bytes()
                .first()
                .is_some_and(|b| b.is_ascii_digit())
        {
            break;
        }
        if trimmed.len() > 2
            && trimmed.as_bytes()[0].is_ascii_digit()
            && trimmed.contains("::")
            && trimmed.contains("at ")
        {
            break;
        }
        lines.push(line);
    }
    let msg = lines.join("\n").trim().to_string();
    if msg.is_empty() {
        format!("{}", e.root_cause())
    } else {
        msg
    }
}

fn set_active_generation(state: &AppState, model: &str, prompt: &str) {
    let prompt_sha256 = format!("{:x}", Sha256::digest(prompt.as_bytes()));
    let started_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let mut active = state
        .active_generation
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *active = Some(ActiveGenerationSnapshot {
        model: model.to_string(),
        prompt_sha256,
        started_at_unix_ms,
        started_at: Instant::now(),
    });
}

fn clear_active_generation(state: &AppState) {
    let mut active = state
        .active_generation
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *active = None;
}

fn save_image_to_dir(
    dir: &std::path::Path,
    img: &mold_core::ImageData,
    model: &str,
    batch_size: u32,
) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!("failed to create output dir {}: {e}", dir.display());
        return;
    }
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let ext = img.format.to_string();
    let filename =
        mold_core::default_output_filename(model, timestamp_ms, &ext, batch_size, img.index);
    let path = dir.join(&filename);
    match std::fs::write(&path, &img.data) {
        Ok(()) => tracing::info!("saved image to {}", path.display()),
        Err(e) => tracing::warn!("failed to save image to {}: {e}", path.display()),
    }
}

/// Runs the generation queue worker loop. Processes one job at a time (FIFO).
/// Exits when the sender half of the channel is dropped (server shutdown).
pub async fn run_queue_worker(
    mut job_rx: tokio::sync::mpsc::Receiver<GenerationJob>,
    state: AppState,
) {
    tracing::debug!("generation queue worker started");
    while let Some(job) = job_rx.recv().await {
        process_job(&state, job).await;
        state.queue.decrement();
    }
    tracing::info!("generation queue worker shutting down");
}

async fn process_job(state: &AppState, job: GenerationJob) {
    // Check if client already disconnected before doing any work
    if job.result_tx.is_closed() {
        tracing::debug!("skipping queued job — client disconnected");
        return;
    }

    // Send "now processing" event (position 0)
    if let Some(ref tx) = job.progress_tx {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::Queued {
            position: 0,
        }));
    }

    // 1. Ensure model is ready (with progress forwarding)
    let progress_callback = job.progress_tx.as_ref().map(|tx| {
        let tx = tx.clone();
        Arc::new(move |event: mold_inference::ProgressEvent| {
            let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
        }) as model_manager::EngineProgressCallback
    });

    if let Err(api_err) =
        model_manager::ensure_model_ready(state, &job.request.model, progress_callback).await
    {
        let err_msg = api_err.error.clone();
        if let Some(ref tx) = job.progress_tx {
            let _ = tx.send(SseMessage::Error(SseErrorEvent {
                message: err_msg.clone(),
            }));
        }
        let _ = job.result_tx.send(Err(err_msg));
        return;
    }

    // 2. Run inference in spawn_blocking
    let engine = state.engine.clone();
    let active_gen = state.active_generation.clone();
    let gen_state = state.clone();
    let gen_req = job.request.clone();
    let progress_tx = job.progress_tx.clone();

    let result = tokio::task::spawn_blocking(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = engine.blocking_lock();
            let e = guard.as_mut().ok_or_else(|| {
                anyhow::anyhow!("no engine available after model readiness check")
            })?;
            set_active_generation(&gen_state, &gen_req.model, &gen_req.prompt);

            // Install progress callback for the generate phase
            if let Some(ref ptx) = progress_tx {
                let ptx = ptx.clone();
                e.set_on_progress(Box::new(move |event| {
                    let _ = ptx.send(SseMessage::Progress(progress_to_sse(event)));
                }));
            } else {
                e.clear_on_progress();
            }

            let generate_result = e.generate(&gen_req);
            if progress_tx.is_some() {
                e.clear_on_progress();
            }
            clear_active_generation(&gen_state);
            generate_result
        }))
    })
    .await;

    match result {
        Ok(Ok(Ok(mut response))) => {
            if response.images.is_empty() {
                let err_msg = "generation error: engine returned no images".to_string();
                if let Some(ref tx) = job.progress_tx {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent {
                        message: err_msg.clone(),
                    }));
                }
                let _ = job.result_tx.send(Err(err_msg));
                return;
            }
            let img = response.images.remove(0);

            // Save to output directory if configured
            if let Some(ref dir) = job.output_dir {
                let dir = dir.clone();
                let img_clone = img.clone();
                let model = job.request.model.clone();
                let batch_size = job.request.batch_size;
                tokio::task::spawn_blocking(move || {
                    save_image_to_dir(&dir, &img_clone, &model, batch_size);
                });
            }

            // Send SSE complete event
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Complete(SseCompleteEvent {
                    image: base64::engine::general_purpose::STANDARD.encode(&img.data),
                    format: img.format,
                    width: img.width,
                    height: img.height,
                    seed_used: response.seed_used,
                    generation_time_ms: response.generation_time_ms,
                }));
            }

            // Send result through oneshot
            let _ = job.result_tx.send(Ok(GenerationJobResult {
                image: img,
                response,
            }));
        }
        Ok(Ok(Err(e))) => {
            *active_gen.write().unwrap_or_else(|e| e.into_inner()) = None;
            tracing::error!("generation error: {e:#}");
            let err_msg = format!("generation error: {}", clean_error_message(&e));
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: err_msg.clone(),
                }));
            }
            let _ = job.result_tx.send(Err(err_msg));
        }
        Ok(Err(panic_payload)) => {
            *active_gen.write().unwrap_or_else(|e| e.into_inner()) = None;
            let msg = panic_payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            tracing::error!("inference panicked: {msg}");
            let err_msg = format!("inference panicked: {msg}");
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: err_msg.clone(),
                }));
            }
            let _ = job.result_tx.send(Err(err_msg));
        }
        Err(join_err) => {
            *active_gen.write().unwrap_or_else(|e| e.into_inner()) = None;
            tracing::error!("inference task join error: {join_err:?}");
            let err_msg = "inference task failed".to_string();
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: err_msg.clone(),
                }));
            }
            let _ = job.result_tx.send(Err(err_msg));
        }
    }
}
