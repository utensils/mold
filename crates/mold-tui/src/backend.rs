use std::sync::Arc;

#[cfg(test)]
use mold_core::ServerCapabilities;
use mold_core::{
    classify_generate_error, download::DownloadProgressEvent, GenerateRequest, GenerateResponse,
    GenerateServerAction, GenerationBatchChildState, LoraWeight, MoldClient, PromptExpander,
    PromptTransformOperation, RemixRequest, RemixResponse, RemixVariant, SseProgressEvent,
};
use tokio::sync::mpsc;

use crate::app::{
    BackgroundEvent, DurableGenerationChildOutcome, GenerateParams, GenerationMetadataSnapshot,
    InferenceMode, PromptTransformSnapshot,
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
    // A family whose profile ignores the prompt (no text encoder) is answered
    // from the guide on every host, without a round trip or a model: the
    // same one-variant answer `/api/expand`, `/api/remix`, and the CLI give.
    let result = if let Some(advice) = mold_core::ignored_prompt_advice(&request.model_family) {
        Ok(RemixResponse {
            source_prompt: request.source_prompt.clone(),
            root_prompt: request.root_prompt.clone(),
            source_kind: request.source_kind,
            task: request
                .task
                .unwrap_or_else(|| mold_core::ExpandTask::for_family(&request.model_family)),
            variants: vec![RemixVariant {
                prompt: advice.text(),
                dimensions: Vec::new(),
            }],
        })
    } else if let Some(url) = server_url {
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
                    context: request.context.clone(),
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

    // Batch N goes through `POST /api/generation-batches`, matching the CLI.
    // A singleton keeps the `/api/generate/stream` facade. Both feed the same
    // `SseProgressEvent::Preview` frames into the centered fixed-protocol
    // preview sink CLAUDE.md protects by name: the batch path polls each
    // running child's progress snapshot and unfolds it into those events.
    // Both are the same durable admission underneath.
    if batch > 1 && params.inference_mode != InferenceMode::Local {
        let effective_url = params.host.clone().or_else(|| server_url.clone());
        if let Some(url) = effective_url {
            let client = crate::hosts::client_for(&url, api_key.as_deref());
            match try_canonical_remote_batch(CanonicalBatchInput {
                client: &client,
                host: crate::app::HeldHost {
                    url: url.clone(),
                    api_key: api_key.clone(),
                },
                params: &params,
                prompt: &prompt,
                negative_prompt: &negative_prompt,
                prepared_prompts: &prepared_prompts,
                prepared_transforms: &prepared_transforms,
                batch,
                base_seed,
                tx: &tx,
            })
            .await
            {
                CanonicalBatchResult::Done => return,
                CanonicalBatchResult::Error(error) => {
                    let _ = tx.send(BackgroundEvent::Error(error));
                    return;
                }
                // Fall through to the per-item loop below, which is the
                // singleton path and owns the local fallback.
                CanonicalBatchResult::FallbackLocal => {}
            }
        }
    }

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
                let mut req = match build_request(&iter_params, &iter_prompt, &negative_prompt) {
                    Ok(req) => req,
                    Err(message) => {
                        // The identity photo went away between picking it
                        // and pressing Generate. Refusing is the whole
                        // point: an ordinary render would look fine and
                        // simply not be that person.
                        let _ = tx.send(BackgroundEvent::Error(message));
                        return;
                    }
                };
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

                let host_label = iter_params
                    .target_host_name
                    .clone()
                    .unwrap_or_else(|| url.clone());
                let result =
                    try_server_generate(&client, &host_label, &req, &metadata_snapshot, &tx).await;
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum CanonicalBatchResult {
    Done,
    Error(String),
    /// The host could not be reached at all. The singleton path answers this
    /// by rendering locally, and a batch must not be the one submission that
    /// fails outright on an unreachable server.
    FallbackLocal,
}

fn new_client_batch_id() -> String {
    let mut bytes: [u8; 16] = rand::random();
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

fn build_batch_requests(
    params: &GenerateParams,
    prompt: &str,
    negative_prompt: &Option<String>,
    prepared_prompts: &[String],
    prepared_transforms: &[mold_core::PromptTransformProvenance],
    batch: u32,
    base_seed: Option<u64>,
) -> Result<Vec<GenerateRequest>, String> {
    // Admission ids belong to transport chunks; this id belongs to the user's
    // logical Batch N and must remain stable when a host advertises a smaller
    // canonical admission limit. A restored singleton keeps its exact saved
    // provenance, while an ordinary or prepared multi-output submission gets
    // one fresh identity and global positions rather than inheriting a source
    // print's older batch group.
    let preserve_exact_provenance = batch == 1
        && params.batch_id.is_some()
        && params.batch_index.is_some()
        && params.batch_count.is_some();
    let logical_batch_id = if preserve_exact_provenance {
        params
            .batch_id
            .clone()
            .expect("complete provenance has an id")
    } else {
        new_client_batch_id()
    };
    let mut requests = Vec::with_capacity(batch as usize);
    for index in 0..batch {
        let mut child = params.clone();
        child.batch = 1;
        child.prepared_prompts.clear();
        child.prepared_prompt_transforms.clear();
        child.seed = base_seed.map(|seed| seed.wrapping_add(index as u64));
        let child_prompt = prepared_prompts
            .get(index as usize)
            .map(String::as_str)
            .unwrap_or(prompt);
        if let Some(transform) = prepared_transforms.get(index as usize) {
            child.prompt_transform = Some(transform.clone());
        }
        let mut request = build_request(&child, child_prompt, negative_prompt)?;
        request.batch_size = 1;
        request.batch_id = Some(logical_batch_id.clone());
        if !preserve_exact_provenance {
            request.batch_index = Some(index + 1);
            request.batch_count = Some(batch);
        }
        requests.push(request);
    }
    Ok(requests)
}

fn batch_child_state_label(state: &GenerationBatchChildState) -> &'static str {
    match state {
        GenerationBatchChildState::Accepted => "accepted",
        GenerationBatchChildState::Paused => "paused",
        GenerationBatchChildState::Cancelling => "cancelling",
        GenerationBatchChildState::Running => "running",
        GenerationBatchChildState::Complete => "complete",
        GenerationBatchChildState::Failed => "failed",
        GenerationBatchChildState::Cancelled => "cancelled",
        GenerationBatchChildState::Held => "held",
    }
}

struct CanonicalBatchInput<'a> {
    client: &'a MoldClient,
    /// The route `client` was built from, kept beside it so a held child can
    /// be retried on the host that admitted it.
    host: crate::app::HeldHost,
    params: &'a GenerateParams,
    prompt: &'a str,
    negative_prompt: &'a Option<String>,
    prepared_prompts: &'a [String],
    prepared_transforms: &'a [mold_core::PromptTransformProvenance],
    batch: u32,
    base_seed: Option<u64>,
    tx: &'a mpsc::UnboundedSender<BackgroundEvent>,
}

async fn try_canonical_remote_batch(input: CanonicalBatchInput<'_>) -> CanonicalBatchResult {
    let requests = match build_batch_requests(
        input.params,
        input.prompt,
        input.negative_prompt,
        input.prepared_prompts,
        input.prepared_transforms,
        input.batch,
        input.base_seed,
    ) {
        Ok(requests) => requests,
        Err(error) => return CanonicalBatchResult::Error(error),
    };

    let observed = std::sync::Mutex::new(std::collections::HashMap::<
        String,
        GenerationBatchChildState,
    >::new());
    // Unfolds each polled progress snapshot back into the same
    // `SseProgressEvent`s the singleton stream delivers, so the protected
    // centered denoise preview and the step counter work for a batch too.
    let progress = std::sync::Mutex::new(mold_core::queue_progress::ProgressEventStream::new());
    let observer = |event: mold_core::durable_generation::CanonicalGenerationEvent| match event {
        mold_core::durable_generation::CanonicalGenerationEvent::Progress {
            job_id,
            progress: snapshot,
            ..
        } => {
            let events = progress
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .events(&job_id, &snapshot);
            for event in events {
                let _ = input.tx.send(BackgroundEvent::Progress(event));
            }
        }
        mold_core::durable_generation::CanonicalGenerationEvent::Admitted {
            status,
            request_offset,
            ..
        } => {
            let first = request_offset + 1;
            let last = request_offset + status.children.len() as u32;
            let _ = input
                .tx
                .send(BackgroundEvent::Progress(SseProgressEvent::Info {
                    message: format!("Durably accepted batch {first}-{last}"),
                }));
        }
        mold_core::durable_generation::CanonicalGenerationEvent::Snapshot {
            status,
            request_offset,
            ..
        } => {
            let mut observed = observed.lock().unwrap_or_else(|poison| poison.into_inner());
            for child in status.children {
                if observed.get(&child.job_id) != Some(&child.state) {
                    observed.insert(child.job_id, child.state.clone());
                    let _ = input
                        .tx
                        .send(BackgroundEvent::Progress(SseProgressEvent::Info {
                            message: format!(
                                "Batch {} {}",
                                request_offset + child.index,
                                batch_child_state_label(&child.state)
                            ),
                        }));
                }
            }
        }
        mold_core::durable_generation::CanonicalGenerationEvent::ReconcileDelayed {
            error, ..
        } => {
            let _ = input
                .tx
                .send(BackgroundEvent::Progress(SseProgressEvent::Info {
                    message: format!("Durable queue reconciliation delayed: {error}"),
                }));
        }
    };

    let report = match mold_core::durable_generation::canonical_generation_observed(
        input.client,
        &requests,
        Some(&observer),
    )
    .await
    {
        Ok(report) => report,
        Err(error) if mold_core::MoldClient::is_connection_error(&error) => {
            return CanonicalBatchResult::FallbackLocal
        }
        Err(error) => return CanonicalBatchResult::Error(format!("{error:#}")),
    };

    finish_canonical_batch(
        report,
        input.client,
        input.host.clone(),
        BatchSubmission {
            prompt: input.prompt,
            negative_prompt: input.negative_prompt.as_deref(),
            model: &input.params.model,
        },
        input.tx,
    )
    .await
}

/// Retry every held child this client still holds authority for, then wait
/// for their batches to settle again.
///
/// `POST /api/queue/{id}/retry` fences on the whole admission authority, so
/// this is the only place a TUI retry can come from: the Machines queue lanes
/// list rows with no batch identity at all.
pub async fn retry_held_prints(
    held: crate::app::HeldBatch,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    let crate::app::HeldBatch {
        host,
        submission,
        retries: held,
    } = held;
    let client = crate::hosts::client_for(&host.url, host.api_key.as_deref());
    let mut authorities: Vec<mold_core::GenerationBatchAuthority> = Vec::new();
    let mut failures = Vec::new();
    for entry in &held {
        let _ = tx.send(BackgroundEvent::Progress(SseProgressEvent::Info {
            message: format!("Retrying batch {}", entry.index),
        }));
        match mold_core::durable_generation::retry_canonical_child(
            &client,
            &entry.authority,
            &entry.job_id,
            0,
        )
        .await
        {
            Ok(_) => {
                if !authorities
                    .iter()
                    .any(|known| known.batch_id == entry.authority.batch_id)
                {
                    authorities.push(entry.authority.clone());
                }
            }
            Err(error) => failures.push(format!("Batch {}: {error:#}", entry.index)),
        }
    }
    if authorities.is_empty() {
        let _ = tx.send(BackgroundEvent::Error(if failures.is_empty() {
            "No held prints to retry".to_string()
        } else {
            failures.join("; ")
        }));
        return;
    }
    let mut outcomes = Vec::new();
    for authority in &authorities {
        match mold_core::durable_generation::wait_for_settled_batch(&client, authority).await {
            Ok(status) => {
                for child in status.children {
                    let Some(entry) = held.iter().find(|entry| entry.job_id == child.job_id) else {
                        continue;
                    };
                    outcomes.push(child_outcome(entry.index, authority.clone(), child));
                }
            }
            Err(error) => failures.push(format!("{error:#}")),
        }
    }
    outcomes.sort_by_key(|outcome| outcome.index);
    hydrate_last_completed(&client, &mut outcomes).await;
    let _ = tx.send(BackgroundEvent::DurableGenerationBatchComplete {
        outcomes,
        prompt: submission.prompt,
        negative_prompt: submission.negative_prompt,
        model: submission.model,
        host,
    });
    if !failures.is_empty() {
        let _ = tx.send(BackgroundEvent::Error(failures.join("; ")));
    }
}

/// [`BatchSubmission`] with owned strings, for the retry task that outlives
/// the form it was read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedBatchSubmission {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub model: String,
}

/// Project one settled child onto the pane's outcome row.
fn child_outcome(
    index: u32,
    authority: mold_core::GenerationBatchAuthority,
    child: mold_core::GenerationBatchChild,
) -> DurableGenerationChildOutcome {
    let result = child.result.unwrap_or_default();
    let error =
        match child.state {
            GenerationBatchChildState::Complete
                if result.filename.is_none() && result.original_filename.is_none() =>
            {
                Some("completed without a durable gallery filename".to_string())
            }
            GenerationBatchChildState::Complete => None,
            _ => Some(child.error.unwrap_or_else(|| {
                format!("generation {}", batch_child_state_label(&child.state))
            })),
        };
    DurableGenerationChildOutcome {
        index,
        job_id: child.job_id,
        authority,
        filename: result.filename,
        original_filename: result.original_filename,
        error,
        retryable: child.retryable.unwrap_or(false),
        seed: result.seed,
        generation_time_ms: result.generation_time_ms,
        preview_bytes: None,
        mesh: None,
    }
}

/// The pane shows one print, so only the last completed sibling's bytes are
/// worth a round trip.
async fn hydrate_last_completed(
    client: &MoldClient,
    outcomes: &mut [DurableGenerationChildOutcome],
) {
    let Some(last) = outcomes
        .iter_mut()
        .rev()
        .find(|outcome| outcome.error.is_none())
    else {
        return;
    };
    let Some(filename) = last
        .filename
        .as_deref()
        .or(last.original_filename.as_deref())
    else {
        return;
    };
    // A mesh has no raster to show: its `.glb` would only fail the pane's
    // decode after a multi-megabyte download, so the poster the server
    // rendered at save time (served by the thumbnail route) is fetched
    // instead — the same picture the grid shows.
    if crate::gallery_scan::is_mesh_filename(filename) {
        last.preview_bytes = client.get_gallery_thumbnail(filename).await.ok();
        // The caption's tris · verts · extent: a singleton reads them off
        // `MeshData`, but neither the durable child result nor the gallery
        // record carries them, so the stored GLB is read back and counted
        // — one download for the print the pane is about to show, the same
        // round trip a raster batch makes for its preview bytes.
        last.mesh = client
            .get_gallery_image(filename)
            .await
            .ok()
            .and_then(|glb| crate::app::DurableMeshFacts::from_glb(&glb));
    } else {
        last.preview_bytes = client.get_gallery_image(filename).await.ok();
    }
}

/// What the batch was submitted with. The Create form may have moved on
/// while it rendered, and prompt history records what was developed.
struct BatchSubmission<'a> {
    prompt: &'a str,
    negative_prompt: Option<&'a str>,
    model: &'a str,
}

async fn finish_canonical_batch(
    report: mold_core::durable_generation::CanonicalGenerationReport,
    client: &MoldClient,
    host: crate::app::HeldHost,
    submission: BatchSubmission<'_>,
    tx: &mpsc::UnboundedSender<BackgroundEvent>,
) -> CanonicalBatchResult {
    let mut outcomes = report
        .outcomes
        .into_iter()
        .map(|outcome| {
            let index = outcome.request_offset + outcome.child.index;
            child_outcome(index, outcome.authority, outcome.child)
        })
        .collect::<Vec<_>>();
    outcomes.sort_by_key(|outcome| outcome.index);
    hydrate_last_completed(client, &mut outcomes).await;
    let _ = tx.send(BackgroundEvent::DurableGenerationBatchComplete {
        outcomes,
        prompt: submission.prompt.to_string(),
        negative_prompt: submission.negative_prompt.map(str::to_string),
        model: submission.model.to_string(),
        host,
    });

    if report.orchestration_failures.is_empty() {
        CanonicalBatchResult::Done
    } else {
        CanonicalBatchResult::Error(report.orchestration_failures.join("; "))
    }
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
    host_label: &str,
    req: &GenerateRequest,
    metadata_snapshot: &GenerationMetadataSnapshot,
    tx: &mpsc::UnboundedSender<BackgroundEvent>,
) -> ServerResult {
    if let Err(error) = prepare_remote_licensed_dependencies(client, host_label, req, tx).await {
        return ServerResult::Error(error);
    }
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

async fn request_license_consent(
    host_label: String,
    requirements: Vec<crate::app::LicenseDownloadRequirement>,
    tx: &mpsc::UnboundedSender<BackgroundEvent>,
) -> Result<bool, String> {
    if requirements.is_empty() {
        return Ok(true);
    }
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    tx.send(BackgroundEvent::LicenseRequired {
        host_label,
        requirements,
        response: response_tx,
    })
    .map_err(|_| "the license review UI is unavailable".to_string())?;
    response_rx
        .await
        .map_err(|_| "the license review was closed".to_string())
}

fn grouped_license_requirements(
    pending: &[mold_core::PendingModelDownload],
) -> Vec<crate::app::LicenseDownloadRequirement> {
    let mut by_model = std::collections::BTreeMap::<
        String,
        std::collections::BTreeMap<String, mold_core::LicenseRefusal>,
    >::new();
    for download in pending {
        let Some(model) = download.install_model.as_ref() else {
            continue;
        };
        let licenses = by_model.entry(model.clone()).or_default();
        for license in &download.licenses {
            licenses.insert(
                format!("{}\0{}\0{}", license.id, license.url, license.sha256),
                license.clone(),
            );
        }
    }
    by_model
        .into_iter()
        .filter_map(|(install_model, licenses)| {
            (!licenses.is_empty()).then(|| crate::app::LicenseDownloadRequirement {
                install_model,
                licenses: licenses.into_values().collect(),
            })
        })
        .collect()
}

async fn prepare_remote_licensed_dependencies(
    client: &MoldClient,
    host_label: &str,
    req: &GenerateRequest,
    tx: &mpsc::UnboundedSender<BackgroundEvent>,
) -> Result<(), String> {
    let preview = match client.preview_generation_placement(req.clone(), 1).await {
        Ok(preview) => preview,
        // Compatibility: an older server has no structured preflight. Its
        // existing admission error remains visible, but cannot be accepted in
        // place because it exposes no safe pinned-terms contract.
        Err(error) if mold_core::client::is_missing_endpoint_error(&error) => return Ok(()),
        Err(error) => {
            return Err(format!(
                "Could not verify licensed dependencies on {host_label}: {error}"
            ));
        }
    };
    let requirements = grouped_license_requirements(&preview.pending_downloads);
    if !request_license_consent(host_label.to_string(), requirements.clone(), tx).await? {
        return Err("License acceptance cancelled; nothing was queued.".to_string());
    }
    for requirement in requirements {
        let acceptances = requirement
            .licenses
            .iter()
            .map(|license| mold_core::LicenseAcceptance {
                id: license.id.clone(),
                url: license.url.clone(),
                sha256: license.sha256.clone(),
            })
            .collect::<Vec<_>>();
        let (progress_tx, mut progress_rx) = mpsc::unbounded_channel();
        let progress_events = tx.clone();
        tokio::spawn(async move {
            while let Some(event) = progress_rx.recv().await {
                let _ = progress_events.send(BackgroundEvent::Progress(event));
            }
        });
        client
            .pull_model_stream_accepting(&requirement.install_model, &acceptances, progress_tx)
            .await
            .map_err(|error| {
                format!(
                    "Failed to download '{}' on {host_label}: {error}",
                    requirement.install_model
                )
            })?;
    }
    Ok(())
}

pub async fn pull_remote_model_with_consent(
    client: &MoldClient,
    host_label: String,
    model: String,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) -> Result<(), String> {
    let statuses = match client.list_licenses().await {
        Ok(statuses) => statuses,
        Err(error) if mold_core::client::is_missing_endpoint_error(&error) => Vec::new(),
        Err(error) => {
            return Err(format!("Could not read licenses on {host_label}: {error}"));
        }
    };
    let licenses = statuses
        .into_iter()
        .filter(|license| {
            !license.accepted && license.required_by.iter().any(|name| name == &model)
        })
        .map(|license| mold_core::LicenseRefusal {
            id: license.id,
            name: license.name,
            url: license.url,
            canonical: license.canonical,
            sha256: license.sha256,
            summary: license.summary,
        })
        .collect::<Vec<_>>();
    let requirements = (!licenses.is_empty())
        .then(|| crate::app::LicenseDownloadRequirement {
            install_model: model.clone(),
            licenses,
        })
        .into_iter()
        .collect::<Vec<_>>();
    if !request_license_consent(host_label, requirements.clone(), &tx).await? {
        return Err("License acceptance cancelled; nothing was downloaded.".to_string());
    }
    let acceptances = requirements
        .iter()
        .flat_map(|requirement| &requirement.licenses)
        .map(|license| mold_core::LicenseAcceptance {
            id: license.id.clone(),
            url: license.url.clone(),
            sha256: license.sha256.clone(),
        })
        .collect::<Vec<_>>();
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel();
    let progress_events = tx.clone();
    tokio::spawn(async move {
        while let Some(event) = progress_rx.recv().await {
            let _ = progress_events.send(BackgroundEvent::Progress(event));
        }
    });
    client
        .pull_model_stream_accepting(&model, &acceptances, progress_tx)
        .await
        .map_err(|error| format!("Server pull failed: {error}"))
}

pub async fn pull_local_model_with_consent(
    model: String,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) -> Result<(), String> {
    let home = mold_core::Config::mold_dir()
        .ok_or_else(|| "Could not resolve the Mold data directory".to_string())?;
    let config = mold_core::Config::load_or_default();
    let missing_files = mold_core::manifest::find_manifest(&model)
        .map(|manifest| {
            manifest
                .files
                .iter()
                .filter(|file| config.complete_manifest_file_path(manifest, file).is_none())
                .map(|file| file.hf_filename.as_str())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let licenses =
        mold_core::license_acceptance::unaccepted_for_manifest_files(&model, missing_files, &home);
    let requirements = (!licenses.is_empty())
        .then(|| crate::app::LicenseDownloadRequirement {
            install_model: model.clone(),
            licenses,
        })
        .into_iter()
        .collect::<Vec<_>>();
    if !request_license_consent("This device".to_string(), requirements.clone(), &tx).await? {
        return Err("License acceptance cancelled; nothing was downloaded.".to_string());
    }
    let acceptances = requirements
        .iter()
        .flat_map(|requirement| &requirement.licenses)
        .map(|license| mold_core::LicenseAcceptance {
            id: license.id.clone(),
            url: license.url.clone(),
            sha256: license.sha256.clone(),
        })
        .collect::<Vec<_>>();
    mold_core::license_acceptance::record_acceptances(&home, &acceptances)
        .map_err(|error| format!("Could not record license acceptance: {error}"))?;
    auto_pull_model(&model, &tx).await.map(|_| ())
}

/// Materialize auxiliary bundles the in-process TUI engine needs before it
/// starts. The consent/download loop is bundle-generic; request policy merely
/// names which registered bundles apply.
async fn prepare_local_licensed_dependencies(
    request: &GenerateRequest,
    config: &mold_core::Config,
    tx: &mpsc::UnboundedSender<BackgroundEvent>,
) -> Result<(), String> {
    let family = config.resolved_model_config(&request.model).family;
    for bundle in
        mold_core::manifest::auxiliary_manifests_for_request_with_family(request, family.as_deref())
    {
        if config.manifest_model_needs_download(bundle) {
            pull_local_model_with_consent(bundle.to_string(), tx.clone()).await?;
        }
    }
    Ok(())
}

/// Single attempt to generate via the durable `/api/generate/stream` facade.
async fn try_server_generate_once(
    client: &MoldClient,
    req: &GenerateRequest,
    progress_tx: mpsc::UnboundedSender<SseProgressEvent>,
) -> Result<GenerateResponse, anyhow::Error> {
    client.generate_stream(req, progress_tx).await
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

    let offload = params.offload;
    let req = match build_request(&params, &prompt, &negative_prompt) {
        Ok(req) => req,
        Err(message) => {
            let _ = tx.send(BackgroundEvent::Error(message));
            return;
        }
    };
    if let Err(message) = prepare_local_licensed_dependencies(&req, &config, &tx).await {
        let _ = tx.send(BackgroundEvent::Error(message));
        return;
    }
    // A bundle pull may have updated model paths or their configured root.
    config = Config::load_or_default();

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

/// Build the wire request for one generation.
///
/// Fallible because of the identity photo alone: every other conditioning
/// input degrades to "absent" when it cannot be read, but an identity
/// reference that silently vanishes turns the run into an ordinary render
/// that produces a plausible print with the wrong face and says nothing. A
/// file accepted at entry can still be deleted, truncated, or swapped for a
/// symlink before Generate is pressed, so the load is re-checked here and a
/// failure aborts dispatch with `mold_core::identity`'s own wording.
pub(crate) fn build_request(
    params: &GenerateParams,
    prompt: &str,
    negative_prompt: &Option<String>,
) -> Result<GenerateRequest, String> {
    let config = mold_core::Config::load_or_default();
    let (normalized_params, normalized_negative_prompt) =
        canonicalize_generation_authority(params.clone(), negative_prompt.clone(), &config);
    let params = &normalized_params;
    let family = crate::model_info::family_for_model(&params.model, &config);
    let supports_video = crate::model_info::capabilities_for_family(&family).supports_video;

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

    // The ordered reference group, resolved from the recipe's own contract
    // rather than from the family name — the sniff here is what left
    // flux2-dev falling through to `source_image` and being refused at
    // admission for a request the form had no way to author.
    let reference_images =
        mold_core::generation_profile::reference_images_for_recipe(&family, &params.model);
    // Every reference is read or the dispatch fails. `edit_images` is
    // ORDERED — each entry is packed at its own time coordinate — so
    // skipping the one file that could not be read renumbers every reference
    // after it and renders something the user did not ask for, silently; and
    // if they all fail the request quietly degrades to img2img. The path is
    // the user's own, shown back to them in their own TUI, so it is quoted
    // whole rather than redacted.
    //
    // Read only on a recipe that HAS a References row: on a target-first
    // recipe the arm below never looks at these paths, and failing a render
    // over a group that recipe cannot carry would be a refusal with no
    // control on screen to clear it.
    let reference_bytes: Vec<Vec<u8>> = if reference_images.primary_is_target {
        Vec::new()
    } else {
        params
            .edit_image_paths
            .iter()
            .enumerate()
            .map(|(index, path)| {
                std::fs::read(path).map_err(|err| {
                    format!(
                        "Reference image {} ({path}) could not be read: {err}",
                        index + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, String>>()?
    };
    let (edit_images, source_image, strength, mask_image) = if reference_images.primary_is_target {
        // The FIRST image is the thing being edited, so the Source row IS
        // the reference group's head and there is no denoise pass to weight.
        (
            source_image.clone().map(|image| vec![image]),
            None,
            0.75,
            None,
        )
    } else if !reference_bytes.is_empty() {
        // References attached: `Replaces` and `Exclusive` both refuse the
        // img2img fields alongside them, so the request carries the group
        // alone.
        (Some(reference_bytes), None, params.strength, None)
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

    // Identity conditioning ships as a group or not at all: the two knobs
    // without a photo are exactly the incomplete form
    // `mold_core::identity::validate_identity_conditioning` refuses. The
    // photo is re-read here (it was validated at entry) so the request
    // carries the file as it is right now, and the label is the basename —
    // never the local path.
    let identity_image = match params.identity_image_path.as_deref() {
        Some(path) => Some(crate::identity::load_identity_image(path)?),
        None => None,
    };
    let identity_image_name = identity_image.as_ref().and_then(|_| {
        params
            .identity_image_path
            .as_deref()
            .and_then(crate::identity::identity_image_name)
    });

    // Creation-time filing is a client decision (see
    // `mold_core::compose_client_tags`): the server never auto-tags, so the
    // title's slug is composed here, exactly as the CLI composes it. Both
    // File-under editors and `start_generation`'s guard already refuse a
    // list the auto tag would push past the cap, so the error arm is
    // defensive — it keeps the tags the user typed rather than dropping them.
    let composed = crate::ui::create_form::compose_filing_tags(
        &params.tags,
        params.title.as_deref(),
        params.auto_tag_title,
    )
    .unwrap_or_else(|_| mold_core::ComposedClientTags {
        tags: params.tags.clone(),
        auto_tagged: None,
    });

    Ok(GenerateRequest {
        offload: None,
        // Absent-until-touched, like every optional block: an untouched form
        // ships no `mesh` at all (the recipe's defaults apply), and the
        // capability sync already cleared the block on any recipe whose
        // profile has no `mesh` — where the server refuses it rather than
        // ignoring it.
        mesh: (params.mesh != mold_core::MeshRequestOptions::default())
            .then(|| params.mesh.clone()),
        video_only: None,
        collection: params
            .collection
            .as_deref()
            .map(mold_core::CollectionRef::by_name),
        tags: (!composed.tags.is_empty()).then_some(composed.tags),
        title: params.title.clone(),
        // The TUI has no multi-photograph or true-CFG control yet (#1226); its
        // single `--id-image` equivalent rides the singular fields below.
        id_images: None,
        id_image_names: None,
        true_cfg: None,
        cfg_start_step: None,
        source_fit: None,
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
        save_to_gallery: None,
        original_prompt: params.original_prompt.clone(),
        prompt_transform: params.prompt_transform.clone(),
        batch_id: params.batch_id.clone(),
        batch_index: params.batch_index,
        batch_count: params.batch_count,
        lora,
        // Image recipes reject video timing even when the values happen to
        // match the TUI's hidden defaults. Only put these fields on the wire
        // when the selected family exposes the Frames/FPS controls.
        frames: (supports_video
            && !(params.predict_duration && params.duration_prediction_supported))
            .then_some(params.frames),
        fps: supports_video.then_some(params.fps),
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
        id_weight: identity_image.as_ref().map(|_| params.id_weight),
        id_start_step: identity_image.as_ref().map(|_| params.id_start_step),
        id_image_name: identity_image_name,
        id_image: identity_image,
    })
}

/// Build a map of file_path -> list of model names that reference it.
///
/// Delegates to `mold_core::removal::build_ref_counts` so the TUI's removal
/// flow shares the CLI/server ownership rules: manifest-backed installs
/// without a config entry still count as owners, and a configured model
/// missing files on disk does not.
pub(crate) fn build_ref_counts(
    config: &mold_core::Config,
) -> std::collections::HashMap<String, Vec<String>> {
    mold_core::removal::build_ref_counts(config)
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

    #[tokio::test]
    async fn prompt_transform_answers_a_prompt_ignored_family_without_a_host_or_a_model() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let request = RemixRequest {
            source_prompt: "a dining chair".to_string(),
            root_prompt: None,
            source_kind: mold_core::RemixSourceKind::Direct,
            model_family: "hunyuan3d".to_string(),
            variations: 3,
            style: None,
            task: None,
            dimensions: Vec::new(),
            context: None,
        };
        let snapshot = PromptTransformSnapshot {
            operation: PromptTransformOperation::Expand,
            model: "hunyuan3d-mini-turbo:fp16".to_string(),
            target: crate::hosts::GenTarget::Host("nothing-listens".to_string()),
            task: mold_core::ExpandTask::TextToImage,
            reference_fingerprint: String::new(),
            source_prompt: "a dining chair".to_string(),
            current_prompt: "a dining chair".to_string(),
            root_prompt: None,
            source_kind: mold_core::RemixSourceKind::Direct,
        };
        // An unreachable host: a round trip would fail, so a completed
        // transform proves none was made.
        run_prompt_transform(
            Some("http://127.0.0.1:9".to_string()),
            None,
            PromptTransformOperation::Expand,
            request,
            snapshot,
            7,
            tx,
        )
        .await;
        let advice = mold_core::ignored_prompt_advice("hunyuan3d").unwrap();
        match rx.recv().await.unwrap() {
            BackgroundEvent::PromptTransformComplete {
                token, response, ..
            } => {
                assert_eq!(token, 7);
                assert_eq!(response.variants.len(), 1);
                assert_eq!(response.variants[0].prompt, advice.text());
                assert!(response.variants[0].dimensions.is_empty());
            }
            _ => panic!("unexpected event"),
        }
    }

    fn canonical_batch_capabilities(limit: u32) -> ServerCapabilities {
        let mut capabilities = ServerCapabilities::default();
        capabilities.queue.heterogeneous_batch_max_outputs = Some(limit);
        capabilities.durable_media = Some(mold_core::DurableMediaCapabilities::v2(false));
        capabilities
    }

    fn ordinary_request() -> GenerateRequest {
        let config = mold_core::Config::default();
        build_request(&GenerateParams::from_config(&config), "print", &None).unwrap()
    }

    /// The Source row's path is read at dispatch and rides the request as
    /// `source_image` bytes with its file name — on a mesh recipe exactly as
    /// on a raster one, with the GLB pin untouched.
    #[test]
    fn the_source_row_path_rides_the_request_as_bytes() {
        let config = mold_core::Config::default();
        let mut params = GenerateParams::from_config(&config);
        params.model = mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL.to_string();
        let dir = tempfile::tempdir().unwrap();
        let cat = dir.path().join("armchair.png");
        std::fs::write(&cat, b"\x89PNG\r\n\x1a\nnot really a png").unwrap();
        params.source_image_path = Some(cat.to_string_lossy().to_string());

        let request = build_request(&params, "", &None).unwrap();
        assert_eq!(
            request.source_image.as_deref(),
            Some(&b"\x89PNG\r\n\x1a\nnot really a png"[..])
        );
        assert_eq!(request.source_image_name.as_deref(), Some("armchair.png"));
        assert_eq!(request.output_format, Some(mold_core::OutputFormat::Glb));

        params.source_image_path = None;
        let request = build_request(&params, "", &None).unwrap();
        assert_eq!(request.source_image, None);
        assert_eq!(request.source_image_name, None);
    }

    /// The mesh block is absent-until-touched: an untouched form ships no
    /// `mesh` (the recipe's defaults apply, and a raster recipe would refuse
    /// the block), and a touched knob ships exactly as the row shows it.
    #[test]
    fn the_mesh_block_rides_the_request_only_once_touched() {
        let config = mold_core::Config::default();
        let mut params = GenerateParams::from_config(&config);
        params.model = mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL.to_string();
        let untouched = build_request(&params, "", &None).unwrap();
        assert!(untouched.mesh.is_none());
        // The family normalizer pins the container regardless of the form.
        assert_eq!(untouched.output_format, Some(mold_core::OutputFormat::Glb));

        params.mesh.octree_resolution = Some(320);
        params.mesh.threshold = Some(0.55);
        let touched = build_request(&params, "", &None).unwrap();
        let mesh = touched.mesh.expect("touched knobs ship");
        assert_eq!(mesh.octree_resolution, Some(320));
        assert_eq!(mesh.threshold, Some(0.55));
        assert_eq!(mesh.target_faces, None);
        assert_eq!(mesh.texture, None, "no texture stage is ever requested");
    }

    /// A host that advertises no batch limit advertises no generation at all,
    /// and the refusal must say so rather than degrade to a second path.
    #[test]
    fn a_host_with_no_advertised_limit_refuses_generation_by_name() {
        let request = ordinary_request();
        let mut capabilities = canonical_batch_capabilities(17);
        assert_eq!(
            capabilities.canonical_generation_batch_limit(std::slice::from_ref(&request)),
            Ok(17)
        );

        capabilities.queue.heterogeneous_batch_max_outputs = None;
        assert_eq!(
            capabilities.canonical_generation_batch_limit(&[request]),
            Err(mold_core::CanonicalRefusal::GenerationUnavailable)
        );
    }

    #[test]
    fn canonical_request_builder_covers_a_remote_singleton() {
        let config = mold_core::Config::default();
        let params = GenerateParams::from_config(&config);
        let requests = build_batch_requests(&params, "one print", &None, &[], &[], 1, Some(42))
            .expect("singleton request");

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].batch_size, 1);
        assert_eq!(requests[0].seed, Some(42));
        assert_eq!(
            canonical_batch_capabilities(64).canonical_generation_batch_limit(&requests),
            Ok(64)
        );
    }

    /// The singleton path renders locally when the server cannot be reached.
    /// A batch must not be the one submission that fails outright instead.
    #[tokio::test]
    async fn an_unreachable_host_hands_the_batch_to_the_local_path() {
        let config = mold_core::Config::default();
        let params = GenerateParams::from_config(&config);
        let (tx, _rx) = mpsc::unbounded_channel();
        // Port 1 is not listening: the capability read fails to connect.
        let client = MoldClient::new("http://127.0.0.1:1");
        let result = try_canonical_remote_batch(CanonicalBatchInput {
            host: crate::app::HeldHost {
                url: "http://127.0.0.1:1".into(),
                api_key: None,
            },
            client: &client,
            params: &params,
            prompt: "two prints",
            negative_prompt: &None,
            prepared_prompts: &[],
            prepared_transforms: &[],
            batch: 2,
            base_seed: Some(1),
            tx: &tx,
        })
        .await;
        assert_eq!(result, CanonicalBatchResult::FallbackLocal);
    }

    #[tokio::test]
    async fn held_child_emits_one_structured_completion_without_global_error() {
        let authority = mold_core::GenerationBatchAuthority {
            instance_id: "instance-1".into(),
            batch_id: "batch-1".into(),
            client_batch_id: "client-1".into(),
        };
        let report = mold_core::durable_generation::CanonicalGenerationReport {
            authorities: vec![authority.clone()],
            admitted_client_ids: vec!["client-1".into()],
            outcomes: vec![mold_core::durable_generation::CanonicalGenerationOutcome {
                authority,
                client_batch_id: "client-1".into(),
                request_offset: 0,
                request: ordinary_request(),
                child: mold_core::GenerationBatchChild {
                    index: 1,
                    job_id: "job-1".into(),
                    state: GenerationBatchChildState::Held,
                    error: Some("dependency unavailable".into()),
                    error_code: None,
                    retryable: Some(true),
                    created_at_ms: 0,
                    updated_at_ms: 0,
                    revision: 1,
                    completed_at_ms: None,
                    terminal_error: None,
                    result: None,
                },
            }],
            orchestration_failures: Vec::new(),
            failures: vec!["terminal held child".into()],
        };
        let (tx, mut rx) = mpsc::unbounded_channel();

        assert_eq!(
            finish_canonical_batch(
                report,
                &MoldClient::new("http://127.0.0.1:1"),
                crate::app::HeldHost {
                    url: "http://127.0.0.1:1".into(),
                    api_key: None,
                },
                BatchSubmission {
                    prompt: "a held print",
                    negative_prompt: None,
                    model: "flux-dev:q8",
                },
                &tx,
            )
            .await,
            CanonicalBatchResult::Done
        );
        match rx.try_recv().unwrap() {
            BackgroundEvent::DurableGenerationBatchComplete {
                outcomes,
                prompt,
                model,
                ..
            } => {
                assert_eq!(outcomes.len(), 1);
                assert_eq!(outcomes[0].error.as_deref(), Some("dependency unavailable"));
                assert!(outcomes[0].retryable);
                // A held child produced nothing, so there is nothing to
                // hydrate and no terminal facts to report.
                assert_eq!(outcomes[0].seed, None);
                assert_eq!(outcomes[0].generation_time_ms, None);
                assert!(outcomes[0].preview_bytes.is_none());
                assert_eq!(prompt, "a held print");
                assert_eq!(model, "flux-dev:q8");
            }
            _ => panic!("unexpected non-completion event"),
        }
        assert!(rx.try_recv().is_err());
    }

    /// A completed child carries the seed and elapsed time its settlement
    /// recorded, which is what the pane restores.
    #[tokio::test]
    async fn a_completed_child_reports_the_terminal_facts_it_settled_with() {
        let authority = mold_core::GenerationBatchAuthority {
            instance_id: "instance-1".into(),
            batch_id: "batch-1".into(),
            client_batch_id: "client-1".into(),
        };
        let report = mold_core::durable_generation::CanonicalGenerationReport {
            authorities: vec![authority.clone()],
            admitted_client_ids: vec!["client-1".into()],
            outcomes: vec![mold_core::durable_generation::CanonicalGenerationOutcome {
                authority,
                client_batch_id: "client-1".into(),
                request_offset: 0,
                request: ordinary_request(),
                child: mold_core::GenerationBatchChild {
                    index: 1,
                    job_id: "job-1".into(),
                    state: GenerationBatchChildState::Complete,
                    error: None,
                    error_code: None,
                    retryable: None,
                    created_at_ms: 0,
                    updated_at_ms: 0,
                    revision: 2,
                    completed_at_ms: Some(9),
                    terminal_error: None,
                    result: Some(mold_core::GenerationBatchResult {
                        filename: Some("print.png".into()),
                        original_filename: None,
                        seed: Some(4242),
                        generation_time_ms: Some(7_500),
                        gpu: Some(0),
                    }),
                },
            }],
            orchestration_failures: Vec::new(),
            failures: Vec::new(),
        };
        let (tx, mut rx) = mpsc::unbounded_channel();

        assert_eq!(
            finish_canonical_batch(
                report,
                // Unreachable on purpose: hydration is best effort and must
                // never turn a completed print into a failure.
                &MoldClient::new("http://127.0.0.1:1"),
                crate::app::HeldHost {
                    url: "http://127.0.0.1:1".into(),
                    api_key: None,
                },
                BatchSubmission {
                    prompt: "a finished print",
                    negative_prompt: Some("blurry"),
                    model: "flux-dev:q8",
                },
                &tx,
            )
            .await,
            CanonicalBatchResult::Done
        );
        match rx.try_recv().unwrap() {
            BackgroundEvent::DurableGenerationBatchComplete {
                outcomes,
                negative_prompt,
                ..
            } => {
                assert_eq!(outcomes[0].seed, Some(4242));
                assert_eq!(outcomes[0].generation_time_ms, Some(7_500));
                assert_eq!(outcomes[0].error, None);
                assert_eq!(negative_prompt.as_deref(), Some("blurry"));
            }
            _ => panic!("unexpected non-completion event"),
        }
    }

    /// The client gates on the MACHINE alone: a host that advertises the
    /// durable queue admits every request shape — media, a LoRA beside it,
    /// identity — and the server's typed refusal answers for anything it
    /// cannot take. A host with no durable queue is refused by name.
    #[test]
    fn the_client_gate_reads_only_the_machines_durable_queue() {
        let capabilities = canonical_batch_capabilities(64);
        let mut source = ordinary_request();
        source.source_image = Some(vec![1, 2, 3]);
        source.lora = Some(LoraWeight {
            path: "adapter.safetensors".into(),
            scale: 1.0,
            expert: None,
        });
        assert_eq!(
            capabilities.canonical_generation_batch_limit(std::slice::from_ref(&source)),
            Ok(64)
        );
        let mut no_media = capabilities.clone();
        no_media.durable_media = None;
        assert_eq!(
            no_media.canonical_generation_batch_limit(&[source.clone()]),
            Ok(64),
            "durable media is the server's per-request refusal, not a client fence"
        );
        let mut no_queue = capabilities;
        no_queue.queue.heterogeneous_batch_max_outputs = None;
        assert_eq!(
            no_queue.canonical_generation_batch_limit(&[source]),
            Err(mold_core::CanonicalRefusal::GenerationUnavailable)
        );
    }

    #[test]
    fn batch_request_builder_freezes_order_prompts_and_singleton_seeds() {
        let config = mold_core::Config::default();
        let mut params = GenerateParams::from_config(&config);
        params.batch = 3;
        params.seed = Some(u64::MAX - 1);
        let prompts = vec![
            "first".to_string(),
            "second".to_string(),
            "third".to_string(),
        ];

        let requests =
            build_batch_requests(&params, "unused", &None, &prompts, &[], 3, params.seed).unwrap();

        assert_eq!(
            requests
                .iter()
                .map(|request| request.prompt.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(
            requests
                .iter()
                .map(|request| request.seed)
                .collect::<Vec<_>>(),
            vec![Some(u64::MAX - 1), Some(u64::MAX), Some(0)]
        );
        assert!(requests.iter().all(|request| request.batch_size == 1));
        assert!(requests
            .iter()
            .all(|request| request.batch_id == requests[0].batch_id));
        assert_eq!(requests[0].batch_index, Some(1));
        assert_eq!(requests[2].batch_index, Some(3));
        assert!(requests
            .iter()
            .all(|request| request.batch_count == Some(3)));
        let chunks = requests.chunks(2).collect::<Vec<_>>();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0][0].batch_id, chunks[1][0].batch_id);
        assert_eq!(chunks[1][0].batch_index, Some(3));
    }

    #[test]
    fn client_batch_ids_are_uuid_v4_shaped_and_unique() {
        let first = new_client_batch_id();
        let second = new_client_batch_id();
        assert_ne!(first, second);
        assert_eq!(first.len(), 36);
        assert_eq!(&first[14..15], "4");
        assert!(matches!(&first[19..20], "8" | "9" | "a" | "b"));
        assert_eq!(
            first
                .chars()
                .enumerate()
                .filter(|(_, ch)| *ch == '-')
                .map(|(i, _)| i)
                .collect::<Vec<_>>(),
            vec![8, 13, 18, 23]
        );
    }

    #[test]
    fn license_requirements_group_future_terms_by_install_bundle() {
        let terms = mold_core::LicenseRefusal {
            id: "future-research-weights".into(),
            name: "Future research weights".into(),
            url: "https://example.test/pinned".into(),
            canonical: "https://example.test/project".into(),
            sha256: "a".repeat(64),
            summary: "Research use only.".into(),
        };
        let dependency = |name: &str| mold_core::PendingModelDownload {
            kind: "identity_encoder".into(),
            name: name.into(),
            repo: "example/future".into(),
            bytes: 42,
            install_model: Some("future-face-adapter".into()),
            licenses: vec![terms.clone()],
        };

        let grouped = grouped_license_requirements(&[dependency("one"), dependency("two")]);

        assert_eq!(grouped.len(), 1);
        assert_eq!(grouped[0].install_model, "future-face-adapter");
        assert_eq!(grouped[0].licenses, vec![terms]);
    }

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
        // Ownership requires a complete install (the core rule this
        // delegates to), so the referenced files must exist on disk.
        let tmp = std::env::temp_dir().join(format!("mold-tui-refs-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let a_transformer = tmp.join("a-transformer.safetensors");
        let b_transformer = tmp.join("b-transformer.safetensors");
        let shared_vae = tmp.join("shared-vae.safetensors");
        for path in [&a_transformer, &b_transformer, &shared_vae] {
            std::fs::write(path, b"weights").unwrap();
        }

        let mut config = mold_core::Config::default();

        let model_a = mold_core::ModelConfig {
            transformer: Some(a_transformer.to_string_lossy().into_owned()),
            vae: Some(shared_vae.to_string_lossy().into_owned()),
            ..Default::default()
        };

        let model_b = mold_core::ModelConfig {
            transformer: Some(b_transformer.to_string_lossy().into_owned()),
            vae: Some(shared_vae.to_string_lossy().into_owned()),
            ..Default::default()
        };

        config.models.insert("model-a".to_string(), model_a);
        config.models.insert("model-b".to_string(), model_b);

        let refs = build_ref_counts(&config);

        // Unique files should have exactly one reference
        let a_refs = refs
            .get(&a_transformer.to_string_lossy().into_owned())
            .unwrap();
        assert_eq!(a_refs.len(), 1);
        assert!(a_refs.contains(&"model-a".to_string()));

        // Shared files should have both models
        let vae_refs = refs
            .get(&shared_vae.to_string_lossy().into_owned())
            .unwrap();
        assert_eq!(vae_refs.len(), 2);
        assert!(vae_refs.contains(&"model-a".to_string()));
        assert!(vae_refs.contains(&"model-b".to_string()));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn build_ref_counts_empty_config() {
        let config = mold_core::Config::default();
        let refs = build_ref_counts(&config);
        // No configured entries: any owners present come from manifest-backed
        // installs discovered on this machine's disk, and every listed path
        // must name at least one owner.
        assert!(refs.values().all(|owners| !owners.is_empty()));
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
        let req = build_request(&params, "test prompt", &None).unwrap();
        assert_eq!(req.batch_size, 4);
    }

    /// Klein's References row ships `edit_images` and drops the img2img
    /// fields the `Exclusive` contract refuses beside them. Before the
    /// recipe answered this question, flux2-dev fell through to
    /// `source_image` and was refused at admission for a request the form
    /// had no way to author.
    #[test]
    fn build_request_ships_the_reference_group_as_edit_images() {
        let dir = tempfile::tempdir().unwrap();
        let mut paths = Vec::new();
        for (name, bytes) in [("a.png", b"first" as &[u8]), ("b.png", b"second")] {
            let file = dir.path().join(name);
            std::fs::write(&file, bytes).unwrap();
            paths.push(file.to_string_lossy().into_owned());
        }
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.model = "flux2-klein:bf16".to_string();
        params.edit_image_paths = paths;
        params.mask_image_path = Some("/tmp/does-not-matter.png".to_string());
        let req = build_request(&params, "put sunglasses on the person", &None).unwrap();
        assert_eq!(
            req.edit_images,
            Some(vec![b"first".to_vec(), b"second".to_vec()]),
            "order is semantic"
        );
        assert!(req.source_image.is_none());
        assert!(req.mask_image.is_none());
    }

    /// An unreadable reference is a hard failure, never a silent gap.
    ///
    /// `edit_images` is ORDERED — each entry is packed at its own time
    /// coordinate — so dropping the one file that could not be read
    /// renumbers every reference after it and renders a different picture
    /// than the one asked for, with nothing on screen to say so. Worse, if
    /// every path fails, the request quietly reverts to img2img. The refusal
    /// names the one-based position and the path, unredacted: it is the
    /// user's own path, in their own TUI.
    #[test]
    fn build_request_refuses_an_unreadable_reference_instead_of_dropping_it() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("a.png");
        std::fs::write(&good, b"first").unwrap();
        let missing = dir.path().join("gone.png");
        let missing = missing.to_string_lossy().into_owned();
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.model = "flux2-klein:bf16".to_string();
        params.edit_image_paths = vec![good.to_string_lossy().into_owned(), missing.clone()];
        let error = build_request(&params, "a cat", &None)
            .expect_err("a reference that cannot be read must fail the dispatch");
        assert!(
            error.contains('2'),
            "the refusal names the one-based position: {error}"
        );
        assert!(
            error.contains(&missing),
            "the refusal names the path: {error}"
        );
    }

    /// With no references attached, Klein is an ordinary img2img recipe —
    /// the whole reason its relation is `Exclusive` rather than `Replaces`.
    #[test]
    fn build_request_leaves_klein_img2img_alone_without_references() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.png");
        std::fs::write(&source, b"source").unwrap();
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.model = "flux2-klein:bf16".to_string();
        params.source_image_path = Some(source.to_string_lossy().into_owned());
        let req = build_request(&params, "a cat", &None).unwrap();
        assert_eq!(req.source_image, Some(b"source".to_vec()));
        assert!(req.edit_images.is_none());
    }

    #[test]
    fn build_request_single_batch_default() {
        let config = mold_core::Config::load_or_default();
        let params = GenerateParams::from_config(&config);
        assert_eq!(params.batch, 1);
        let req = build_request(&params, "test prompt", &None).unwrap();
        assert_eq!(req.batch_size, 1);
    }

    // ── creation-time filing ("File under") ────────────────────

    /// Absent-until-touched: an untouched Create form submits exactly the
    /// request it always did, with no filing fields on the wire.
    #[test]
    fn build_request_omits_filing_until_the_form_is_touched() {
        let config = mold_core::Config::load_or_default();
        let params = GenerateParams::from_config(&config);
        let req = build_request(&params, "p", &None).unwrap();
        assert_eq!(req.title, None);
        assert_eq!(req.tags, None);
        assert!(req.collection.is_none());
    }

    #[test]
    fn build_request_carries_the_title_tags_and_collection() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.auto_tag_title = false;
        params.title = Some("Smurf Village".to_string());
        params.tags = vec!["village".to_string(), "blue".to_string()];
        params.collection = Some("Blue Period".to_string());

        let req = build_request(&params, "p", &None).unwrap();
        assert_eq!(req.title.as_deref(), Some("Smurf Village"));
        assert_eq!(
            req.tags,
            Some(vec!["village".to_string(), "blue".to_string()])
        );
        // Collections travel by display name — the portable, cross-host
        // form; ids belong to one host's rows.
        let collection = req.collection.expect("collection rides the request");
        assert_eq!(collection.name.as_deref(), Some("Blue Period"));
        assert_eq!(collection.id, None);
    }

    /// The server never auto-tags, so the title's slug is composed here —
    /// the same `mold_core::compose_client_tags` decision the CLI makes.
    #[test]
    fn build_request_auto_tags_a_titled_print_only_while_the_preference_is_on() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.title = Some("Smurf Village".to_string());
        params.tags = vec!["village".to_string()];

        params.auto_tag_title = true;
        assert_eq!(
            build_request(&params, "p", &None).unwrap().tags,
            Some(vec!["village".to_string(), "smurf-village".to_string()])
        );

        params.auto_tag_title = false;
        assert_eq!(
            build_request(&params, "p", &None).unwrap().tags,
            Some(vec!["village".to_string()])
        );

        // A title alone still files the print under its own slug.
        params.auto_tag_title = true;
        params.tags.clear();
        assert_eq!(
            build_request(&params, "p", &None).unwrap().tags,
            Some(vec!["smurf-village".to_string()])
        );

        // …and an untitled form with the preference on carries nothing.
        params.title = None;
        assert_eq!(build_request(&params, "p", &None).unwrap().tags, None);
    }

    #[test]
    fn build_request_carries_filing_on_every_family() {
        // Filing is not a generation parameter: a video request carries it
        // exactly as an image request does.
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.model = "ltx2".to_string();
        params.auto_tag_title = false;
        params.title = Some("Smurf Village".to_string());
        params.collection = Some("Blue Period".to_string());
        let req = build_request(&params, "p", &None).unwrap();
        assert_eq!(req.title.as_deref(), Some("Smurf Village"));
        assert_eq!(
            req.collection.and_then(|c| c.name).as_deref(),
            Some("Blue Period")
        );
    }

    #[test]
    fn build_request_carries_upscale_model() {
        // Create → Advanced → Upscale: the picked upscaler must ride the
        // existing wire field (this used to be hardcoded to None).
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        assert_eq!(
            build_request(&params, "p", &None).unwrap().upscale_model,
            None,
            "off by default"
        );
        params.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        let req = build_request(&params, "p", &None).unwrap();
        assert_eq!(
            req.upscale_model.as_deref(),
            Some("real-esrgan-x4plus:fp16")
        );
    }

    #[test]
    fn build_request_omits_video_timing_for_image_models() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);

        params.model = "sdxl-base:fp16".to_string();
        params.frames = 25;
        params.fps = 24;
        let image = build_request(&params, "p", &None).unwrap();
        assert_eq!(image.frames, None);
        assert_eq!(image.fps, None);
    }

    #[test]
    fn build_request_omits_frames_only_for_qualified_duration_prediction() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.model = mold_core::ltx25_manifest::DISTILLED_INT8_CONV.to_string();
        params.frames = 97;
        params.predict_duration = true;

        let unqualified = build_request(&params, "p", &None).unwrap();
        assert_eq!(unqualified.frames, Some(97));

        params.duration_prediction_supported = true;
        let qualified = build_request(&params, "p", &None).unwrap();
        assert_eq!(qualified.frames, None);
        assert_eq!(qualified.fps, Some(params.fps));
    }

    /// The four identity fields ship as one group. A request carrying a knob
    /// but no photo is exactly the incomplete form
    /// `mold_core::identity::validate_identity_conditioning` refuses, so the
    /// builder must never produce it — not even from a form whose knobs were
    /// moved before a photo was picked.
    #[test]
    fn build_request_ships_identity_as_a_group_or_not_at_all() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);

        let req = build_request(&params, "p", &None).unwrap();
        assert!(req.id_image.is_none());
        assert!(req.id_image_name.is_none());
        assert!(req.id_weight.is_none());
        assert!(req.id_start_step.is_none());
        assert!(!mold_core::identity::request_mentions_identity(&req));

        // Knobs alone never reach the wire.
        params.id_weight = 2.0;
        params.id_start_step = 2;
        let req = build_request(&params, "p", &None).unwrap();
        assert!(
            !mold_core::identity::request_mentions_identity(&req),
            "a knob without a photo must stay absent"
        );

        // A path that no longer resolves builds NO request at all. Dropping
        // the group and rendering anyway would produce a plausible print with
        // the wrong face and say nothing about it.
        params.identity_image_path = Some("/nonexistent/face.png".into());
        assert!(build_request(&params, "p", &None).is_err());

        let dir = tempfile::tempdir().unwrap();
        let photo = dir.path().join("ada.png");
        std::fs::write(&photo, PNG_1X1).unwrap();
        params.identity_image_path = Some(photo.to_string_lossy().to_string());
        let req = build_request(&params, "p", &None).unwrap();
        assert_eq!(req.id_image.as_deref(), Some(&PNG_1X1[..]));
        assert_eq!(
            req.id_image_name.as_deref(),
            Some("ada.png"),
            "the label is the basename, never the local path"
        );
        assert_eq!(req.id_weight, Some(2.0));
        assert_eq!(req.id_start_step, Some(2));
        assert_eq!(mold_core::identity::effective_id_weight(&req), 2.0);
        assert_eq!(mold_core::identity::effective_id_start_step(&req), 2);
    }

    /// The photo is validated when it is picked, but the filesystem keeps
    /// moving: between accepting it and pressing Generate the file can be
    /// deleted, truncated, or swapped for a symlink. Every such case must
    /// abort dispatch with `mold_core::identity`'s wording rather than
    /// quietly building an ordinary request.
    #[test]
    fn a_photo_that_changed_after_it_was_accepted_refuses_to_build_a_request() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        let dir = tempfile::tempdir().unwrap();
        let photo = dir.path().join("ada.png");
        std::fs::write(&photo, PNG_1X1).unwrap();
        params.identity_image_path = Some(photo.to_string_lossy().to_string());
        // Accepted, and it builds.
        assert!(build_request(&params, "p", &None).is_ok());

        // Replaced by a symlink to an equally valid photo: still refused,
        // because the bytes that would be sent are no longer the file that
        // passed the check.
        let real = dir.path().join("someone-else.png");
        std::fs::write(&real, PNG_1X1).unwrap();
        std::fs::remove_file(&photo).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&real, &photo).unwrap();
            let error = build_request(&params, "p", &None).unwrap_err();
            assert!(
                error.starts_with("Identity photo could not be opened"),
                "{error}"
            );
            std::fs::remove_file(&photo).unwrap();
        }

        // Deleted outright.
        assert!(build_request(&params, "p", &None).is_err());

        // Replaced by something that is no longer an image: mold-core's own
        // refusal, not a restatement.
        std::fs::write(&photo, b"not an image").unwrap();
        assert_eq!(
            build_request(&params, "p", &None).unwrap_err(),
            mold_core::identity::validate_id_image_bytes(b"not an image").unwrap_err()
        );
    }

    /// A genuine 1x1 RGBA PNG — the smallest payload
    /// `identity::validate_id_image_bytes` accepts.
    const PNG_1X1: [u8; 67] = [
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    #[test]
    fn build_request_preserves_explicit_audio_choice() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        assert_eq!(
            build_request(&params, "p", &None).unwrap().enable_audio,
            None,
            "untouched TUI state must preserve the server's pipeline default"
        );

        params.enable_audio = Some(true);
        assert_eq!(
            build_request(&params, "p", &None).unwrap().enable_audio,
            Some(true)
        );

        params.enable_audio = Some(false);
        assert_eq!(
            build_request(&params, "p", &None).unwrap().enable_audio,
            Some(false)
        );
    }

    #[test]
    fn build_request_preserves_explicit_ltx2_pipeline_choice() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        assert_eq!(
            build_request(&params, "p", &None).unwrap().pipeline,
            None,
            "untouched TUI state must preserve server pipeline selection"
        );

        params.pipeline = Some(mold_core::Ltx2PipelineMode::TwoStageHq);
        assert_eq!(
            build_request(&params, "p", &None).unwrap().pipeline,
            Some(mold_core::Ltx2PipelineMode::TwoStageHq)
        );
    }

    #[test]
    fn build_request_preserves_ltx2_upscale_choices() {
        let config = mold_core::Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        let default_request = build_request(&params, "p", &None).unwrap();
        assert_eq!(default_request.spatial_upscale, None);
        assert_eq!(default_request.temporal_upscale, None);

        params.spatial_upscale = Some(mold_core::Ltx2SpatialUpscale::X1_5);
        params.temporal_upscale = Some(mold_core::Ltx2TemporalUpscale::X2);
        let request = build_request(&params, "p", &None).unwrap();
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
            build_request(&params, "p", &None)
                .unwrap()
                .guidance_overrides,
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
            build_request(&params, "p", &None)
                .unwrap()
                .guidance_overrides,
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
        let request = build_request(&canonical_params, "p", &canonical_negative).unwrap();

        // The stale count was off the grid, so H3 authority snapped it to
        // the nearest valid clip length.
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
        assert_eq!(request.upscale_model.as_deref(), Some("stale-upscaler"));
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

        let request = build_request(&params, "reviewed sibling", &None).unwrap();
        assert_eq!(request.batch_id.as_deref(), Some("remix-0123"));
        assert_eq!(request.batch_index, Some(2));
        assert_eq!(request.batch_count, Some(3));
        assert_eq!(request.batch_size, 1);
    }
}
