use std::sync::Arc;

use mold_core::{
    classify_generate_error, download::DownloadProgressEvent, ChainRequest, GenerateRequest,
    GenerateResponse, GenerateServerAction, LoraWeight, MoldClient, PromptExpander,
    PromptTransformOperation, RemixRequest, RemixResponse, RemixVariant, SseProgressEvent,
};
use tokio::sync::mpsc;

use crate::app::{
    BackgroundEvent, GenerateParams, GenerationMetadataSnapshot, InferenceMode,
    PromptTransformSnapshot,
};

/// Prepare reviewable Expand/Remix alternatives without queueing generation.
/// A remote target fails closed on `/api/remix`; it never falls back to
/// `/api/expand`, which older hosts could silently interpret incorrectly.
pub async fn run_prompt_transform(
    server_url: Option<String>,
    api_key: Option<String>,
    operation: PromptTransformOperation,
    request: RemixRequest,
    snapshot: PromptTransformSnapshot,
    token: u64,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    let result = if let Some(url) = server_url {
        let client = crate::hosts::client_for(&url, api_key.as_deref());
        match operation {
            PromptTransformOperation::Remix => client.remix_prompt(&request).await,
            PromptTransformOperation::Expand => client
                .expand_prompt(&mold_core::ExpandRequest {
                    prompt: request.source_prompt.clone(),
                    model_family: request.model_family.clone(),
                    variations: request.variations,
                    style: request.style.clone(),
                    task: request.task,
                })
                .await
                .map(|response| RemixResponse {
                    source_prompt: response.original,
                    root_prompt: request.root_prompt.clone(),
                    source_kind: request.source_kind,
                    task: request.task.unwrap_or_else(|| {
                        mold_core::ExpandTask::for_family(&request.model_family)
                    }),
                    variants: response
                        .expanded
                        .into_iter()
                        .map(|prompt| RemixVariant {
                            prompt,
                            dimensions: Vec::new(),
                        })
                        .collect(),
                }),
        }
    } else {
        run_local_prompt_transform(operation, &request).await
    };
    match result {
        Ok(response) => {
            let _ = tx.send(BackgroundEvent::PromptTransformComplete {
                token,
                operation,
                snapshot,
                response,
            });
        }
        Err(error) => {
            let _ = tx.send(BackgroundEvent::PromptTransformFailed {
                token,
                message: format!("Prompt transform failed: {error}"),
            });
        }
    }
}

async fn run_local_prompt_transform(
    operation: PromptTransformOperation,
    request: &RemixRequest,
) -> anyhow::Result<RemixResponse> {
    let config = mold_core::Config::load_or_default();
    let settings = config.expand.clone().with_env_overrides();
    let mut expand_config = settings.to_expand_config(&request.model_family, request.variations);
    expand_config.operation = operation;
    expand_config.task = request
        .task
        .unwrap_or_else(|| mold_core::ExpandTask::for_family(&request.model_family));
    expand_config.style = request.style.clone();
    if operation == PromptTransformOperation::Remix {
        expand_config.remix_dimensions = mold_core::expand::resolve_remix_dimensions(
            &request.dimensions,
            expand_config.task,
            request
                .style
                .as_ref()
                .is_some_and(|style| !style.trim().is_empty()),
        )?;
    }

    let expander: Box<dyn PromptExpander> = if let Some(api) = settings.create_api_expander()? {
        Box::new(api)
    } else {
        #[cfg(feature = "expand")]
        {
            Box::new(
                mold_inference::expand::LocalExpander::from_config(&config, Some(&settings.model))
                    .ok_or_else(|| anyhow::anyhow!("local expansion model is not installed"))?,
            )
        }
        #[cfg(not(feature = "expand"))]
        {
            anyhow::bail!(
                "local prompt transforms require the TUI expand feature or an API backend"
            )
        }
    };
    let prompt = request.source_prompt.clone();
    let result =
        tokio::task::spawn_blocking(move || expander.expand(&prompt, &expand_config)).await??;
    let task = request
        .task
        .unwrap_or_else(|| mold_core::ExpandTask::for_family(&request.model_family));
    let dimensions = if operation == PromptTransformOperation::Remix {
        mold_core::expand::resolve_remix_dimensions(
            &request.dimensions,
            task,
            request
                .style
                .as_ref()
                .is_some_and(|style| !style.trim().is_empty()),
        )?
    } else {
        Vec::new()
    };
    Ok(RemixResponse {
        source_prompt: request.source_prompt.clone(),
        root_prompt: request.root_prompt.clone(),
        source_kind: request.source_kind,
        task,
        variants: result
            .expanded
            .into_iter()
            .enumerate()
            .map(|(index, prompt)| RemixVariant {
                prompt,
                dimensions: if operation == PromptTransformOperation::Remix {
                    mold_core::expand::remix_dimensions_for_position(&dimensions, index + 1)
                } else {
                    Vec::new()
                },
            })
            .collect(),
    })
}

/// Run a generation request — tries remote first, falls back to local on connection error.
/// When batch > 1, loops client-side with `batch_size=1` per iteration (matching CLI behavior).
///
/// `api_key` carries the per-host key when the Machines generation
/// target routed this run at a specific registered host.
pub async fn run_generation(
    server_url: Option<String>,
    params: GenerateParams,
    prompt: String,
    negative_prompt: Option<String>,
    api_key: Option<String>,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    // Canonicalize before batching and provenance capture so metadata/history
    // describe the exact request that was sent, not hidden stale UI state.
    let config = mold_core::Config::load_or_default();
    let (params, negative_prompt) =
        canonicalize_generation_authority(params, negative_prompt, &config);
    let h3_task = mold_core::minimax_h3::task_for_model(&params.model);
    if h3_task == Some(mold_core::minimax_h3::Task::Ref2va)
        && (params.reference_paths.is_empty()
            || params.batch != 1
            || !params.prepared_prompts.is_empty())
    {
        let _ = tx.send(BackgroundEvent::Error(
            "MiniMax H3 ordered references require at least one reference and Batch 1".to_string(),
        ));
        return;
    }
    if h3_task != Some(mold_core::minimax_h3::Task::Ref2va) && !params.reference_paths.is_empty() {
        let _ = tx.send(BackgroundEvent::Error(
            "Ordered references require an explicitly authorized MiniMax H3 Ref2VA model"
                .to_string(),
        ));
        return;
    }
    let prepared_prompts = params.prepared_prompts.clone();
    let prepared_transforms = params.prepared_prompt_transforms.clone();
    let batch = if prepared_prompts.is_empty() {
        params.batch
    } else {
        prepared_prompts.len() as u32
    };
    let base_seed = params.seed;
    let prepared_batch_id =
        (!prepared_prompts.is_empty()).then(|| format!("remix-{:032x}", rand::random::<u128>()));

    for i in 0..batch {
        let mut iter_params = params.clone();
        iter_params.batch = 1;
        let iter_prompt = prepared_prompts
            .get(i as usize)
            .cloned()
            .unwrap_or_else(|| prompt.clone());
        if let Some(transform) = prepared_transforms.get(i as usize) {
            iter_params.prompt_transform = Some(transform.clone());
        }
        if let Some(batch_id) = &prepared_batch_id {
            iter_params.batch_id = Some(batch_id.clone());
            iter_params.batch_index = Some(i + 1);
            iter_params.batch_count = Some(batch);
        }
        iter_params.prepared_prompts.clear();
        iter_params.prepared_prompt_transforms.clear();

        // Increment seed for each batch iteration (first uses original seed)
        if i > 0 {
            iter_params.seed = base_seed.map(|s| s.wrapping_add(i as u64));
        }

        if batch > 1 {
            let _ = tx.send(BackgroundEvent::Progress(SseProgressEvent::Info {
                message: format!("Generating image {}/{batch}...", i + 1),
            }));
        }

        let metadata_snapshot = GenerationMetadataSnapshot::new(
            iter_params.clone(),
            iter_prompt.clone(),
            negative_prompt.clone(),
        );

        if iter_params.inference_mode == InferenceMode::Local && h3_task.is_some() {
            let _ = tx.send(BackgroundEvent::Error(h3_runtime_unavailable_message(None)));
            return;
        } else if iter_params.inference_mode == InferenceMode::Local {
            run_local_generation_single(
                iter_params,
                iter_prompt.clone(),
                negative_prompt.clone(),
                metadata_snapshot,
                &tx,
            )
            .await;
        } else {
            let effective_url = iter_params.host.clone().or_else(|| server_url.clone());

            let mut fell_through = false;
            if let Some(ref url) = effective_url {
                let client = crate::hosts::client_for(url, api_key.as_deref());
                let mut req = build_request(&iter_params, &iter_prompt, &negative_prompt);
                let mut reference_session = if iter_params.reference_paths.is_empty() {
                    None
                } else {
                    let reference_paths = iter_params.reference_paths.clone();
                    let reference_client = client.clone();
                    let prepared = match tokio::task::spawn_blocking(move || {
                        crate::h3_references::prepare_references(
                            &reference_client,
                            &reference_paths,
                        )
                    })
                    .await
                    {
                        Ok(Ok(prepared)) => prepared,
                        Ok(Err(error)) => {
                            let _ = tx.send(BackgroundEvent::Error(format!(
                                "MiniMax H3 reference preparation failed: {error}"
                            )));
                            return;
                        }
                        Err(error) => {
                            let _ = tx.send(BackgroundEvent::Error(format!(
                                "MiniMax H3 reference preparation task failed: {error}"
                            )));
                            return;
                        }
                    };
                    match crate::h3_references::bind_remote_references(&client, &mut req, prepared)
                        .await
                    {
                        Ok(session) => Some(session),
                        Err(error) => {
                            let _ = tx.send(BackgroundEvent::Error(format!(
                                "MiniMax H3 reference authorization/upload failed: {error}"
                            )));
                            return;
                        }
                    }
                };

                let result = try_server_generate(&client, &req, &metadata_snapshot, &tx).await;
                if matches!(&result, ServerResult::Done) {
                    if let Some(lease) = reference_session.as_mut() {
                        lease.mark_consumed();
                    }
                } else if let Some(lease) = reference_session.as_mut() {
                    let _ = lease.cancel().await;
                }
                match result {
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
                if h3_task.is_some() {
                    let _ = tx.send(BackgroundEvent::Error(h3_runtime_unavailable_message(
                        effective_url.as_deref(),
                    )));
                    return;
                }
                if iter_params.inference_mode == InferenceMode::Remote {
                    // An explicit target that's down is an error naming the
                    // host + a concrete fix — never a silent local fallback.
                    let _ = tx.send(BackgroundEvent::Error(remote_unreachable_message(
                        iter_params.target_host_name.as_deref(),
                        effective_url.as_deref(),
                    )));
                    return;
                }
                run_local_generation_single(
                    iter_params,
                    iter_prompt.clone(),
                    negative_prompt.clone(),
                    metadata_snapshot,
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
///
/// This is the one funnel for both the forced `InferenceMode::Local` case
/// and the Auto-mode fallback, so it is where local runs mirror server
/// admission's negative materialization (#787 round 3): an absent
/// `negative_prompt` on a family with a tuned engine fallback (wan) is
/// resolved into the request *and* the provenance snapshot before the engine
/// runs, so locally saved metadata records the uncond that actually
/// conditioned the render instead of omitting it.
async fn run_local_generation_single(
    params: GenerateParams,
    prompt: String,
    mut negative_prompt: Option<String>,
    mut metadata_snapshot: GenerationMetadataSnapshot,
    tx: &mpsc::UnboundedSender<BackgroundEvent>,
) {
    let config = mold_core::Config::load_or_default();
    let family = crate::model_info::family_for_model(&params.model, &config);
    materialize_local_negative_authority(&mut negative_prompt, &mut metadata_snapshot, &family);
    run_local_generation(
        params,
        prompt,
        negative_prompt,
        metadata_snapshot,
        tx.clone(),
    )
    .await;
}

/// Resolve the engine's absence-fallback negative into a local request and
/// its metadata snapshot — the TUI-local mirror of the server's
/// `materialize_default_negative_prompt` (#787). The wan engine substitutes
/// its tuned default whenever `negative_prompt` is absent, so recording the
/// request as-received would save provenance claiming no negative while the
/// long tuned uncond conditioned the render. An explicit value — the `""`
/// opt-out above all — is authority and passes through untouched.
fn materialize_local_negative_authority(
    negative_prompt: &mut Option<String>,
    metadata_snapshot: &mut GenerationMetadataSnapshot,
    family: &str,
) {
    if negative_prompt.is_none() {
        if let Some(default) = mold_core::manifest::default_negative_prompt_for_family(family) {
            *negative_prompt = Some(default.to_string());
        }
    }
    metadata_snapshot.negative_prompt = negative_prompt.clone();
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

/// Compose the no-fallback error for `InferenceMode::Remote` (spec §11
/// error pattern: name the thing, name the fix). A Machines-targeted run
/// carries the host's display name in `GenerateParams.target_host_name`.
pub(crate) fn remote_unreachable_message(host_name: Option<&str>, url: Option<&str>) -> String {
    match (host_name, url) {
        (Some(name), Some(url)) => {
            format!("Can't reach {name} ({url}). Check the host in Machines.")
        }
        (None, Some(url)) => {
            format!("Can't reach {url}. Check the host in Machines, or switch the mode to 'auto' or 'local'.")
        }
        _ => "Server unreachable and mode is set to 'remote'. Switch to 'auto' or 'local'."
            .to_string(),
    }
}

enum ServerResult {
    Done,
    FallbackLocal,
    Error(String),
}

fn requires_secure_generation_stream(req: &GenerateRequest) -> bool {
    mold_core::minimax_h3::task_for_model(&req.model).is_some()
        || req.references.as_ref().is_some_and(|refs| !refs.is_empty())
}

fn h3_runtime_unavailable_message(url: Option<&str>) -> String {
    let target = url
        .map(|url| format!(" on {url}"))
        .unwrap_or_else(|| " in the local TUI runtime".to_string());
    format!(
        "MiniMax H3 runtime is unavailable{target}; select a reachable, authorized mold server. {}",
        mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED
    )
}

/// Try generating via the server. If the server says the model isn't downloaded,
/// auto-pull it and retry once.
async fn try_server_generate(
    client: &MoldClient,
    req: &GenerateRequest,
    metadata_snapshot: &GenerationMetadataSnapshot,
    tx: &mpsc::UnboundedSender<BackgroundEvent>,
) -> ServerResult {
    let is_h3 = mold_core::minimax_h3::task_for_model(&req.model).is_some();
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
                metadata_snapshot: Box::new(metadata_snapshot.clone()),
            });
            ServerResult::Done
        }
        Err(e) if is_h3 => ServerResult::Error(format!(
            "MiniMax H3 runtime/authorization unavailable: {e} ({})",
            mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED
        )),
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
                            metadata_snapshot: Box::new(metadata_snapshot.clone()),
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
        Ok(None) if requires_secure_generation_stream(req) => {
            anyhow::bail!(
                "server lacks secure streaming generation required for one-use MiniMax H3 references; update the server"
            )
        }
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
    metadata_snapshot: GenerationMetadataSnapshot,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    use mold_core::{Config, ModelPaths};
    use mold_inference::progress::ProgressEvent;
    use mold_inference::{create_engine, LoadStrategy};

    let mut config = Config::load_or_default();
    let model_name = params.model.clone();
    if mold_core::minimax_h3::task_for_model(&model_name).is_some() {
        let _ = tx.send(BackgroundEvent::Error(h3_runtime_unavailable_message(None)));
        return;
    }

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
                metadata_snapshot: Box::new(metadata_snapshot),
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

fn canonicalize_generation_authority(
    mut params: GenerateParams,
    mut negative_prompt: Option<String>,
    config: &mold_core::Config,
) -> (GenerateParams, Option<String>) {
    let family = crate::model_info::family_for_model(&params.model, config);
    crate::app::normalize_generate_params_for_family(&mut params, &family);
    if mold_core::minimax_h3::is_family(&family) {
        negative_prompt = None;
    }
    (params, negative_prompt)
}

fn build_request(
    params: &GenerateParams,
    prompt: &str,
    negative_prompt: &Option<String>,
) -> GenerateRequest {
    let config = mold_core::Config::load_or_default();
    let (normalized_params, normalized_negative_prompt) =
        canonicalize_generation_authority(params.clone(), negative_prompt.clone(), &config);
    let params = &normalized_params;
    let family = crate::model_info::family_for_model(&params.model, &config);

    let lora = params.lora_path.as_ref().map(|path| LoraWeight {
        path: path.clone(),
        scale: params.lora_scale,

        expert: None,
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
    // Provenance label: the picked file's name, so clients (and the desktop's
    // Reuse settings) can attempt to restore the input image. The label
    // rides ONLY with an actual `source_image` — qwen-edit moves the image
    // into `edit_images` and must not ship a dangling name.
    let source_image_name = source_image.as_ref().and_then(|_| {
        params.source_image_path.as_ref().and_then(|p| {
            std::path::Path::new(p)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
        })
    });

    GenerateRequest {
        hdr_exr_dir: None,
        hdr_exr_full_float: false,
        guidance_overrides: params.guidance_overrides.clone().into_option(),
        sample_shift: params.sample_shift,
        distill_strength_high: None,
        distill_strength_low: None,
        prompt: prompt.to_string(),
        negative_prompt: normalized_negative_prompt,
        model: params.model.clone(),
        width: params.width,
        height: params.height,
        steps: params.steps,
        guidance: params.guidance,
        seed: params.seed,
        batch_size: params.batch,
        output_format: Some(params.format),
        embed_metadata: Some(config.effective_embed_metadata(None)),
        scheduler: params.scheduler,
        cfg_plus: None,
        edit_images,
        references: None,
        source_image,
        source_image_name,
        strength,
        mask_image,
        control_image,
        control_model: params.control_model.clone(),
        control_scale: params.control_scale,
        expand: if params.expand { Some(true) } else { None },
        original_prompt: params.original_prompt.clone(),
        prompt_transform: params.prompt_transform.clone(),
        batch_id: params.batch_id.clone(),
        batch_index: params.batch_index,
        batch_count: params.batch_count,
        lora,
        frames: Some(params.frames),
        fps: Some(params.fps),
        upscale_model: params.upscale_model.clone(),
        gif_preview: true,
        enable_audio: params.enable_audio,
        audio_file: None,
        audio_file_path: None,
        source_video: None,
        source_video_path: None,
        extend_video: None,
        extend_video_path: None,
        extend_overlap_frames: None,
        keyframes: None,
        pipeline: params.pipeline,
        ic_lora_control: None,
        loras: None,
        retake_range: None,
        spatial_upscale: params.spatial_upscale,
        temporal_upscale: params.temporal_upscale,
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

    /// #787 round 3: a Local-target (or Auto-fallback) run with an untouched
    /// wan editor used to snapshot `None` even though the engine substitutes
    /// the tuned default — locally saved provenance omitted the negative that
    /// conditioned the render. The local funnel now mirrors server
    /// admission's `materialize_default_negative_prompt`: absence resolves
    /// into both the request and the metadata snapshot, while the explicit
    /// `""` opt-out and typed text pass through untouched.
    #[test]
    fn local_runs_materialize_the_wan_default_negative_into_request_and_snapshot() {
        let config = mold_core::Config::default();
        let params = GenerateParams::from_config(&config);
        let mut snapshot = GenerationMetadataSnapshot::new(params, "p".into(), None);

        let mut absent = None;
        materialize_local_negative_authority(&mut absent, &mut snapshot, "wan");
        assert_eq!(
            absent.as_deref(),
            Some(mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT)
        );
        assert_eq!(snapshot.negative_prompt, absent);

        let mut cleared = Some(String::new());
        materialize_local_negative_authority(&mut cleared, &mut snapshot, "wan");
        assert_eq!(cleared.as_deref(), Some(""));
        assert_eq!(snapshot.negative_prompt.as_deref(), Some(""));

        let mut typed = Some("blurry".to_string());
        materialize_local_negative_authority(&mut typed, &mut snapshot, "wan");
        assert_eq!(typed.as_deref(), Some("blurry"));
        assert_eq!(snapshot.negative_prompt.as_deref(), Some("blurry"));

        // Families without an engine fallback keep truthful absence.
        let mut other = None;
        materialize_local_negative_authority(&mut other, &mut snapshot, "flux");
        assert_eq!(other, None);
        assert_eq!(snapshot.negative_prompt, None);
    }

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
    fn remote_unreachable_message_names_host_and_fix() {
        // Machines-targeted run: name + URL + the concrete fix (spec §11).
        assert_eq!(
            remote_unreachable_message(Some("hal9000"), Some("http://hal9000:7680")),
            "Can't reach hal9000 (http://hal9000:7680). Check the host in Machines."
        );
        // Legacy Remote mode without a Machines target still names the URL.
        let msg = remote_unreachable_message(None, Some("http://hal9000:7680"));
        assert!(msg.contains("http://hal9000:7680"), "{msg}");
        assert!(msg.contains("Machines"), "{msg}");
        // No URL at all — the old copy survives.
        assert!(remote_unreachable_message(None, None).contains("Server unreachable"));
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

    #[test]
    fn build_request_carries_upscale_model() {
        // Create → Advanced → Upscale: the picked upscaler must ride the
        // existing wire field (this used to be hardcoded to None).
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        assert_eq!(
            build_request(&params, "p", &None).upscale_model,
            None,
            "off by default"
        );
        params.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        let req = build_request(&params, "p", &None);
        assert_eq!(
            req.upscale_model.as_deref(),
            Some("real-esrgan-x4plus:fp16")
        );
    }

    #[test]
    fn build_request_preserves_explicit_audio_choice() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        assert_eq!(
            build_request(&params, "p", &None).enable_audio,
            None,
            "untouched TUI state must preserve the server's pipeline default"
        );

        params.enable_audio = Some(true);
        assert_eq!(build_request(&params, "p", &None).enable_audio, Some(true));

        params.enable_audio = Some(false);
        assert_eq!(build_request(&params, "p", &None).enable_audio, Some(false));
    }

    #[test]
    fn build_request_preserves_explicit_ltx2_pipeline_choice() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        assert_eq!(
            build_request(&params, "p", &None).pipeline,
            None,
            "untouched TUI state must preserve server pipeline selection"
        );

        params.pipeline = Some(mold_core::Ltx2PipelineMode::TwoStageHq);
        assert_eq!(
            build_request(&params, "p", &None).pipeline,
            Some(mold_core::Ltx2PipelineMode::TwoStageHq)
        );
    }

    #[test]
    fn build_request_preserves_ltx2_upscale_choices() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        let default_request = build_request(&params, "p", &None);
        assert_eq!(default_request.spatial_upscale, None);
        assert_eq!(default_request.temporal_upscale, None);

        params.spatial_upscale = Some(mold_core::Ltx2SpatialUpscale::X1_5);
        params.temporal_upscale = Some(mold_core::Ltx2TemporalUpscale::X2);
        let request = build_request(&params, "p", &None);
        assert_eq!(
            request.spatial_upscale,
            Some(mold_core::Ltx2SpatialUpscale::X1_5)
        );
        assert_eq!(
            request.temporal_upscale,
            Some(mold_core::Ltx2TemporalUpscale::X2)
        );
    }

    #[test]
    fn build_request_preserves_only_explicit_ltx2_guidance_overrides() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        assert_eq!(
            build_request(&params, "p", &None).guidance_overrides,
            None,
            "untouched TUI state must preserve pipeline guidance defaults"
        );

        params.guidance_overrides = mold_core::Ltx2GuidanceOverrides {
            stg_scale: Some(1.5),
            stg_blocks: Some(vec![28, 29]),
            rescale_scale: None,
            modality_scale: Some(3.0),
            skip_step: Some(2),
        };
        assert_eq!(
            build_request(&params, "p", &None).guidance_overrides,
            Some(params.guidance_overrides.clone())
        );
    }

    #[test]
    fn build_request_reasserts_h3_authority_over_stale_shared_fields() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        let readable_path = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        params.model = mold_core::minimax_h3::FL2VA_COMFY.into();
        params.frames = 25;
        params.fps = 30;
        params.format = mold_core::OutputFormat::Png;
        params.enable_audio = Some(false);
        params.guidance = 7.5;
        params.strength = 0.25;
        params.scheduler = Some(mold_core::Scheduler::Ddim);
        params.lora_path = Some("stale.safetensors".into());
        params.source_image_path = Some(readable_path.clone());
        params.mask_image_path = Some(readable_path.clone());
        params.control_image_path = Some(readable_path);
        params.control_model = Some("stale-control".into());
        params.pipeline = Some(mold_core::Ltx2PipelineMode::TwoStage);
        params.spatial_upscale = Some(mold_core::Ltx2SpatialUpscale::X1_5);
        params.temporal_upscale = Some(mold_core::Ltx2TemporalUpscale::X2);
        params.guidance_overrides = mold_core::Ltx2GuidanceOverrides {
            stg_scale: Some(1.5),
            ..Default::default()
        };
        params.upscale_model = Some("stale-upscaler".into());

        let (canonical_params, canonical_negative) = canonicalize_generation_authority(
            params.clone(),
            Some("stale negative".into()),
            &config,
        );
        let snapshot = GenerationMetadataSnapshot::new(
            canonical_params.clone(),
            "p".into(),
            canonical_negative.clone(),
        );
        let request = build_request(&canonical_params, "p", &canonical_negative);

        assert_eq!(request.frames, Some(mold_core::minimax_h3::MIN_FRAMES));
        assert_eq!(request.fps, Some(mold_core::minimax_h3::FIXED_FPS));
        assert_eq!(request.output_format, Some(mold_core::OutputFormat::Mp4));
        assert_eq!(request.enable_audio, Some(true));
        assert_eq!(request.guidance, 0.0);
        assert_eq!(request.strength, 1.0);
        assert_eq!(request.negative_prompt, None);
        assert_eq!(request.scheduler, None);
        assert_eq!(request.lora, None);
        assert_eq!(request.source_image, None);
        assert_eq!(request.mask_image, None);
        assert_eq!(request.control_image, None);
        assert_eq!(request.control_model, None);
        assert_eq!(request.pipeline, None);
        assert_eq!(request.spatial_upscale, None);
        assert_eq!(request.temporal_upscale, None);
        assert_eq!(request.guidance_overrides, None);
        assert_eq!(request.upscale_model, None);
        assert_eq!(snapshot.negative_prompt, request.negative_prompt);
        assert_eq!(snapshot.params.frames, request.frames.unwrap());
        assert_eq!(snapshot.params.fps, request.fps.unwrap());
        assert_eq!(Some(snapshot.params.format), request.output_format);
        assert_eq!(snapshot.params.enable_audio, request.enable_audio);
        assert_eq!(snapshot.params.guidance, request.guidance);
        assert_eq!(snapshot.params.strength, request.strength);
        assert!(requires_secure_generation_stream(&request));
    }

    #[test]
    fn build_request_propagates_prepared_batch_identity() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.batch = 1;
        params.batch_id = Some("remix-0123".into());
        params.batch_index = Some(2);
        params.batch_count = Some(3);

        let request = build_request(&params, "reviewed sibling", &None);
        assert_eq!(request.batch_id.as_deref(), Some("remix-0123"));
        assert_eq!(request.batch_index, Some(2));
        assert_eq!(request.batch_count, Some(3));
        assert_eq!(request.batch_size, 1);
    }
}
