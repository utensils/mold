use anyhow::Result;
use base64::{engine::general_purpose, Engine as _};
use image::GenericImageView;
use mold_core::{
    client::is_transient_request_error, Config, GalleryImage, GenerateRequest, GenerateResponse,
    GenerationBatchAuthority, GenerationBatchChild, GenerationBatchChildState, LoraInfo,
    LoraWeight, MoldClient, OutputFormat, SseProgressEvent,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Weak,
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::sync::Mutex;
use tokio::time::Instant;

use crate::commands::durable_generation::{
    canonical_generation_observed, canonical_singleton_artifact, hydrate_canonical_artifact,
    retry_canonical_child, CanonicalGenerationEvent, CanonicalGenerationReport,
    CanonicalRetrySubmission,
};

const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
const MAX_ASYNC_JOBS: usize = 32;
const RECONCILE_INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const RECONCILE_MAX_BACKOFF: Duration = Duration::from_secs(30);
const RECONCILE_READ_TIMEOUT: Duration = Duration::from_secs(4);
const AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS: usize = 5;
const MAX_LIST_STATUS_RECONCILIATIONS: usize = 4;

fn reconciliation_initial_backoff() -> Duration {
    if cfg!(test) {
        Duration::from_millis(10)
    } else {
        RECONCILE_INITIAL_BACKOFF
    }
}

fn reconciliation_max_backoff() -> Duration {
    if cfg!(test) {
        Duration::from_millis(80)
    } else {
        RECONCILE_MAX_BACKOFF
    }
}

fn reconciliation_read_timeout() -> Duration {
    if cfg!(test) {
        Duration::from_millis(100)
    } else {
        RECONCILE_READ_TIMEOUT
    }
}

fn reconciliation_key(authority: &GenerationBatchAuthority, job_id: &str) -> String {
    [
        &authority.instance_id,
        &authority.batch_id,
        &authority.client_batch_id,
        job_id,
    ]
    .into_iter()
    .map(|part| format!("{}:{part}", part.len()))
    .collect::<Vec<_>>()
    .join("|")
}

async fn generate_canonically(
    client: &MoldClient,
    req: GenerateRequest,
) -> std::result::Result<GenerateResponse, String> {
    let artifact = canonical_singleton_artifact(client, &req)
        .await
        .map_err(|error| format!("mold durable generation failed: {error:#}"))?;
    let image = image::load_from_memory(&artifact.bytes)
        .map_err(|error| format!("could not decode durable output: {error}"))?;
    let (width, height) = image.dimensions();
    let format = artifact
        .filename
        .rsplit_once('.')
        .and_then(|(_, extension)| extension.parse::<OutputFormat>().ok())
        .unwrap_or_else(|| artifact.request.resolved_output_format());
    Ok(GenerateResponse {
        images: vec![mold_core::ImageData {
            data: artifact.bytes,
            format,
            width,
            height,
            index: 0,
        }],
        video: None,
        audio: None,
        generation_time_ms: artifact.result.generation_time_ms.unwrap_or_default(),
        model: artifact.metadata.model,
        seed_used: artifact.metadata.seed,
        gpu: artifact.result.gpu,
        request_warnings: Vec::new(),
    })
}

async fn hydrate_canonical_outcome(
    client: &MoldClient,
    child: &GenerationBatchChild,
) -> std::result::Result<GenerateResponse, String> {
    let terminal = child.result.clone().unwrap_or_default();
    let artifact = hydrate_canonical_artifact(client, child)
        .await
        .map_err(|error| format!("{error:#}"))?;
    let image = image::load_from_memory(&artifact.bytes)
        .map_err(|error| format!("could not decode durable output: {error}"))?;
    let (width, height) = image.dimensions();
    let format = artifact
        .filename
        .rsplit_once('.')
        .and_then(|(_, extension)| extension.parse::<OutputFormat>().ok())
        .ok_or_else(|| {
            format!(
                "accepted output {} has no supported file extension",
                artifact.filename
            )
        })?;
    Ok(GenerateResponse {
        images: vec![mold_core::ImageData {
            data: artifact.bytes,
            format,
            width,
            height,
            index: 0,
        }],
        video: None,
        audio: None,
        generation_time_ms: terminal.generation_time_ms.unwrap_or_default(),
        model: artifact.metadata.model,
        seed_used: artifact.metadata.seed,
        gpu: terminal.gpu,
        request_warnings: Vec::new(),
    })
}

async fn run_async_generation(
    client: MoldClient,
    jobs: AsyncJobRegistry,
    local_job_id: String,
    req: GenerateRequest,
) {
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let event_jobs = jobs.clone();
    let event_job_id = local_job_id.clone();
    let event_task = tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            event_jobs.apply_canonical_event(&event_job_id, event).await;
        }
    });
    let canonical =
        canonical_generation_observed(&client, std::slice::from_ref(&req), Some(&event_tx)).await;
    drop(event_tx);
    let _ = event_task.await;
    match canonical {
        Ok(report) => {
            // The shared canonical helper currently returns only after an
            // authoritative terminal child. Keep the local row truthfully
            // queued while it waits rather than fabricating Running.
            finish_canonical_async(&jobs, &local_job_id, report).await;
        }
        Err(error) => {
            jobs.finish(
                &local_job_id,
                Err(format!("mold durable generation failed: {error:#}")),
            )
            .await;
        }
    }
}

async fn finish_canonical_async(
    jobs: &AsyncJobRegistry,
    local_job_id: &str,
    report: CanonicalGenerationReport,
) {
    let failure = (!report.failures.is_empty()).then(|| report.failures.join("; "));
    let Some(outcome) = report.outcomes.into_iter().next() else {
        if !report.authorities.is_empty() {
            jobs.record_reconciliation_error(
                local_job_id,
                failure.unwrap_or_else(|| "canonical generation reconciliation stopped".into()),
            )
            .await;
            return;
        }
        jobs.finish_canonical(
            local_job_id,
            CanonicalAsyncSettlement {
                status: AsyncJobStatus::Failed,
                error: failure.or_else(|| Some("canonical generation returned no child".into())),
                ..Default::default()
            },
        )
        .await;
        return;
    };
    finish_canonical_child(
        jobs,
        local_job_id,
        outcome.authority,
        outcome.child,
        failure,
    )
    .await;
}

async fn finish_canonical_child(
    jobs: &AsyncJobRegistry,
    local_job_id: &str,
    authority: GenerationBatchAuthority,
    child: GenerationBatchChild,
    failure: Option<String>,
) {
    let status = match child.state {
        GenerationBatchChildState::Complete => AsyncJobStatus::Succeeded,
        GenerationBatchChildState::Held => AsyncJobStatus::Held,
        GenerationBatchChildState::Cancelled => AsyncJobStatus::Cancelled,
        GenerationBatchChildState::Failed => AsyncJobStatus::Failed,
        GenerationBatchChildState::Accepted
        | GenerationBatchChildState::Running
        | GenerationBatchChildState::Cancelling => AsyncJobStatus::Failed,
    };
    let durable_job_id = child.job_id.clone();
    jobs.finish_canonical(
        local_job_id,
        CanonicalAsyncSettlement {
            authority: Some(authority),
            durable_job_id: Some(durable_job_id),
            retryable: child.retryable,
            status,
            error: child.error.clone().or(failure),
            result: None,
            canonical_child: Some(child),
        },
    )
    .await;
}

pub async fn run(host: Option<String>) -> Result<()> {
    let server = McpServer::new(host);
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = BufWriter::new(tokio::io::stdout());

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<Value>(&line) {
            Ok(message) => server.handle_message(message).await,
            Err(e) => Some(error_response(
                Value::Null,
                -32700,
                format!("parse error: {e}"),
            )),
        };

        if let Some(response) = response {
            let encoded = serde_json::to_string(&response)?;
            stdout.write_all(encoded.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }

    Ok(())
}

struct McpServer {
    client: MoldClient,
    jobs: AsyncJobRegistry,
}

fn append_queue_plan_status_lines(lines: &mut Vec<String>, queue: &mold_core::QueueListingWire) {
    let Some(plan) = &queue.plan else {
        return;
    };
    lines.push(format!(
        "queue plan: v{} state={} optimizer={} next_replan={}",
        plan.plan_version,
        plan.state_version,
        plan.optimizer_state,
        plan.next_replan_at_unix_ms
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".into())
    ));
    for item in &plan.work_items {
        let lane = match item.planned_lane_kind.as_ref() {
            Some(mold_core::QueuePlannedLaneKind::HostUtility) => "host-utility".into(),
            Some(mold_core::QueuePlannedLaneKind::Device) => item
                .planned_device_id
                .clone()
                .unwrap_or_else(|| "device".into()),
            Some(mold_core::QueuePlannedLaneKind::Unknown(_)) => "assigned".into(),
            None if item.is_host_utility_lane()
                || item.activity_phase == mold_core::QueueActivityPhase::Cpu =>
            {
                "host-utility".into()
            }
            None => item
                .planned_device_id
                .clone()
                .unwrap_or_else(|| "unassigned".into()),
        };
        lines.push(format!(
            "work {}: phase={} lane={}/{} blocked={} finish={} confidence={}",
            item.work_id,
            item.activity_phase,
            lane,
            item.lane_order
                .map(|value| value.to_string())
                .unwrap_or_else(|| "—".into()),
            item.blocked_reason
                .as_ref()
                .map(|reason| reason.as_str())
                .unwrap_or("none"),
            item.estimated_finish_unix_ms
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".into()),
            item.estimate_confidence,
        ));
    }
}

fn append_queue_summary_status_lines(
    lines: &mut Vec<String>,
    status: &mold_core::ServerStatus,
    queue: Option<&mold_core::QueueListingWire>,
) {
    match (
        status.queue_depth,
        queue.and_then(|listing| listing.page.as_ref()),
    ) {
        (Some(depth), Some(_)) => {
            lines.push(format!("queue backlog: {depth} total nonterminal jobs"));
        }
        (Some(depth), None) => {
            lines.push(format!("queue load (server-reported): {depth} jobs"));
        }
        (None, _) => {}
    }
    if let Some(queue) = queue {
        let label = if queue.page.is_some() {
            "hydrated queue page"
        } else {
            "queue response"
        };
        lines.push(format!("{label}: {} rows", queue.entries.len()));
    }
    if let Some(capacity) = status.queue_capacity {
        lines.push(format!("runtime window capacity: {capacity} jobs"));
    }
}

impl McpServer {
    fn new(host: Option<String>) -> Self {
        let client = match host {
            Some(host) => match std::env::var("MOLD_API_KEY").ok().filter(|k| !k.is_empty()) {
                Some(api_key) => MoldClient::with_api_key(&host, api_key),
                None => MoldClient::new(&host),
            },
            None => MoldClient::from_env(),
        };
        Self {
            client,
            jobs: AsyncJobRegistry::default(),
        }
    }

    async fn handle_message(&self, message: Value) -> Option<Value> {
        match message {
            Value::Array(items) => {
                if items.is_empty() {
                    return Some(error_response(
                        Value::Null,
                        -32600,
                        "invalid request: empty batch",
                    ));
                }
                let mut responses = Vec::new();
                for item in items {
                    if let Some(response) = self.handle_single(item).await {
                        responses.push(response);
                    }
                }
                (!responses.is_empty()).then_some(Value::Array(responses))
            }
            other => self.handle_single(other).await,
        }
    }

    async fn handle_single(&self, message: Value) -> Option<Value> {
        let method = match message.get("method").and_then(Value::as_str) {
            Some(method) => method,
            None => {
                return Some(error_response(
                    message.get("id").cloned().unwrap_or(Value::Null),
                    -32600,
                    "invalid request: missing method",
                ));
            }
        };

        if method == "tools/call" {
            return self.handle_tool_call(&message).await;
        }

        handle_protocol_message(message)
    }

    async fn handle_tool_call(&self, message: &Value) -> Option<Value> {
        let id = message.get("id").cloned()?;

        let params = message.get("params").cloned().unwrap_or_else(|| json!({}));
        let Some(name) = params.get("name").and_then(Value::as_str) else {
            return Some(error_response(id, -32602, "tools/call missing params.name"));
        };
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));

        let result = match name {
            "generate_image" => self.tool_generate_image(arguments).await,
            "generate_image_async" => self.tool_generate_image_async(arguments).await,
            "generation_status" => self.tool_generation_status(arguments).await,
            "generation_retry" => self.tool_generation_retry(arguments).await,
            "list_gallery" => self.tool_list_gallery(arguments).await,
            "get_gallery_image" => self.tool_get_gallery_image(arguments).await,
            "list_models" => self.tool_list_models(arguments).await,
            "list_loras" => self.tool_list_loras(arguments).await,
            "server_status" => self.tool_server_status().await,
            other => Err(format!("unknown tool: {other}")),
        };

        Some(match result {
            Ok(result) => response(id, result),
            Err(err) => response(
                id,
                json!({
                    "content": [{ "type": "text", "text": err }],
                    "isError": true
                }),
            ),
        })
    }

    async fn tool_generate_image(&self, arguments: Value) -> std::result::Result<Value, String> {
        let mut args: GenerateImageArgs =
            serde_json::from_value(arguments).map_err(|e| format!("invalid arguments: {e}"))?;
        let loras = self.resolve_loras(args.loras.take()).await?;
        let req = build_generate_request(args, loras)?;
        let response = generate_canonically(&self.client, req).await?;
        let image = response
            .images
            .first()
            .ok_or_else(|| "mold did not return an image".to_string())?;
        let encoded = general_purpose::STANDARD.encode(&image.data);
        let mut details = format!(
            "Generated {}x{} {} image with {}; seed {}",
            image.width,
            image.height,
            image.format.extension(),
            response.model,
            response.seed_used
        );
        if response.generation_time_ms > 0 {
            details.push_str(&format!(
                "; generation time {:.1}s",
                response.generation_time_ms as f64 / 1000.0
            ));
        }
        if let Some(gpu) = response.gpu {
            details.push_str(&format!("; gpu {gpu}"));
        }

        Ok(json!({
            "content": [
                { "type": "text", "text": details },
                {
                    "type": "image",
                    "data": encoded,
                    "mimeType": image.format.content_type()
                }
            ]
        }))
    }

    async fn tool_generate_image_async(
        &self,
        arguments: Value,
    ) -> std::result::Result<Value, String> {
        let mut args: GenerateImageArgs =
            serde_json::from_value(arguments).map_err(|e| format!("invalid arguments: {e}"))?;
        let loras = self.resolve_loras(args.loras.take()).await?;
        let req = build_generate_request(args, loras)?;
        let job_id = self.jobs.create(&req).await?;
        let client = self.client.clone();
        let jobs = self.jobs.clone();
        let task_job_id = job_id.clone();

        tokio::spawn(async move {
            run_async_generation(client, jobs, task_job_id, req).await;
        });

        Ok(json!({
            "content": [{
                "type": "text",
                "text": format!(
                    "Started async generation job {job_id}. Poll generation_status with {{\"job_id\":\"{job_id}\"}}."
                )
            }],
            "structuredContent": {
                "job_id": job_id,
                "status": "queued"
            }
        }))
    }

    async fn tool_generation_status(&self, arguments: Value) -> std::result::Result<Value, String> {
        let args: GenerationStatusArgs =
            serde_json::from_value(arguments).map_err(|e| format!("invalid arguments: {e}"))?;
        // A caller asking about ONE job wants the print: the default is to
        // hydrate it, and `include_result: false` is the explicit opt-out for
        // a poll that only wants the state.
        let include_result = args.include_result.unwrap_or(true);

        if let Some(job_id) = args.job_id {
            self.jobs.reconcile_pending_job(&self.client, &job_id).await;
            if include_result {
                self.jobs.hydrate_result(&self.client, &job_id).await;
            }
            let Some(job) = self.jobs.get(&job_id).await else {
                return Err(format!("unknown async generation job: {job_id}"));
            };
            return Ok(job_tool_result(&job, include_result));
        }

        for job_id in self
            .jobs
            .eligible_pending_reconciliations(MAX_LIST_STATUS_RECONCILIATIONS)
            .await
        {
            self.jobs.reconcile_pending_job(&self.client, &job_id).await;
        }
        let jobs = self.jobs.list().await;
        if jobs.is_empty() {
            return Ok(text_result("No async generation jobs are tracked."));
        }

        let lines = jobs.iter().map(job_status_line).collect::<Vec<_>>();
        Ok(json!({
            "content": [{ "type": "text", "text": lines.join("\n") }],
            "structuredContent": {
                "jobs": jobs.iter().map(job_summary_json).collect::<Vec<_>>()
            }
        }))
    }

    async fn tool_generation_retry(&self, arguments: Value) -> std::result::Result<Value, String> {
        let args: GenerationRetryArgs =
            serde_json::from_value(arguments).map_err(|e| format!("invalid arguments: {e}"))?;
        let retry = self.jobs.begin_retry(&args.job_id).await?;

        let retry_result = match retry_canonical_child(
            &self.client,
            &retry.authority,
            &retry.durable_job_id,
            retry.observed_revision,
        )
        .await
        {
            Ok(result) => result,
            Err(error) => {
                let error = format!("could not retry durable generation: {error}");
                self.jobs.abort_retry(&args.job_id, error.clone()).await;
                return Err(error);
            }
        };

        let (retry_was_ambiguous, observed_revision) = match &retry_result {
            CanonicalRetrySubmission::Ambiguous {
                observed_revision, ..
            } => (true, *observed_revision),
            CanonicalRetrySubmission::Accepted => (false, 0),
        };
        self.jobs
            .mark_retry_queued(&args.job_id, retry_was_ambiguous, observed_revision)
            .await;
        if let CanonicalRetrySubmission::Ambiguous { error, .. } = &retry_result {
            self.jobs
                .record_reconciliation_error(
                    &args.job_id,
                    format!("retry response was interrupted; reconciling durable state: {error}"),
                )
                .await;
        }
        if !retry_was_ambiguous {
            self.jobs
                .record_reconciliation_error(
                    &args.job_id,
                    "retry accepted; exact status will reconcile on generation_status".into(),
                )
                .await;
        }

        let job = self
            .jobs
            .get(&args.job_id)
            .await
            .ok_or_else(|| format!("async generation job {} disappeared", args.job_id))?;
        Ok(job_tool_result(&job, false))
    }

    async fn tool_list_gallery(&self, arguments: Value) -> std::result::Result<Value, String> {
        let args: ListGalleryArgs =
            serde_json::from_value(arguments).map_err(|e| format!("invalid arguments: {e}"))?;
        let images = self
            .client
            .list_gallery()
            .await
            .map_err(|e| format!("failed to list gallery: {e}"))?;
        let images = filter_sort_gallery(images, &args.filter())?;
        let total = images.len();
        let limit = args.limit.unwrap_or(25).clamp(1, 200);
        let offset = args.offset.unwrap_or(0);
        let page = images
            .into_iter()
            .skip(offset)
            .take(limit)
            .collect::<Vec<_>>();

        if page.is_empty() {
            return Ok(text_result("No gallery images matched."));
        }

        let lines = page.iter().map(gallery_line).collect::<Vec<_>>();
        Ok(json!({
            "content": [{
                "type": "text",
                "text": format!(
                    "Showing {} of {total} matched gallery images:\n{}",
                    lines.len(),
                    lines.join("\n")
                )
            }],
            "structuredContent": {
                "total": total,
                "offset": offset,
                "limit": limit,
                "images": page.iter().map(gallery_summary_json).collect::<Vec<_>>()
            }
        }))
    }

    async fn tool_get_gallery_image(&self, arguments: Value) -> std::result::Result<Value, String> {
        let args: GetGalleryImageArgs =
            serde_json::from_value(arguments).map_err(|e| format!("invalid arguments: {e}"))?;
        let gallery = self
            .client
            .list_gallery()
            .await
            .map_err(|e| format!("failed to list gallery: {e}"))?;

        let selected = if let Some(filename) = args.filename.as_deref() {
            gallery
                .iter()
                .find(|image| image.filename == filename)
                .cloned()
                .ok_or_else(|| format!("gallery image not found: {filename}"))?
        } else {
            let matches = filter_sort_gallery(gallery, &args.filter())?;
            matches
                .into_iter()
                .next()
                .ok_or_else(|| "no gallery image matched".to_string())?
        };

        let format = gallery_format(&selected);
        let use_thumbnail = args.thumbnail.unwrap_or(format == Some(OutputFormat::Mp4));
        let (data, mime_type, returned_thumbnail) = if use_thumbnail {
            (
                self.client
                    .get_gallery_thumbnail(&selected.filename)
                    .await
                    .map_err(|e| format!("failed to fetch gallery thumbnail: {e}"))?,
                OutputFormat::Png.content_type(),
                true,
            )
        } else {
            if format == Some(OutputFormat::Mp4) {
                return Err(
                    "selected gallery item is an MP4 video; set thumbnail=true to fetch a preview"
                        .to_string(),
                );
            }
            let data = self
                .client
                .get_gallery_image(&selected.filename)
                .await
                .map_err(|e| format!("failed to fetch gallery image: {e}"))?;
            let mime_type = format
                .map(|f| f.content_type())
                .unwrap_or_else(|| content_type_for_filename(&selected.filename));
            (data, mime_type, false)
        };

        Ok(json!({
            "content": [
                {
                    "type": "text",
                    "text": if returned_thumbnail {
                        format!("Fetched thumbnail for {}", gallery_line(&selected))
                    } else {
                        format!("Fetched {}", gallery_line(&selected))
                    }
                },
                {
                    "type": "image",
                    "data": general_purpose::STANDARD.encode(data),
                    "mimeType": mime_type
                }
            ],
            "structuredContent": {
                "image": gallery_summary_json(&selected),
                "thumbnail": returned_thumbnail
            }
        }))
    }

    async fn tool_list_models(&self, arguments: Value) -> std::result::Result<Value, String> {
        let args: ListModelsArgs =
            serde_json::from_value(arguments).map_err(|e| format!("invalid arguments: {e}"))?;
        let models = self
            .client
            .list_models_extended()
            .await
            .map_err(|e| format!("failed to list models: {e}"))?;

        let limit = args.limit.unwrap_or(50).clamp(1, 200);
        let mut lines = Vec::new();
        for model in models
            .into_iter()
            .filter(|m| !args.downloaded_only.unwrap_or(true) || m.downloaded)
            .filter(|m| !args.generation_only.unwrap_or(true) || m.is_generation_model())
            .take(limit)
        {
            let status = if model.downloaded {
                "downloaded"
            } else {
                "not downloaded"
            };
            lines.push(format!(
                "- {} [{}] {status}; default {}x{}, steps {}, guidance {}",
                model.name,
                model.family,
                model.defaults.default_width,
                model.defaults.default_height,
                model.defaults.default_steps,
                model.defaults.default_guidance
            ));
        }

        if lines.is_empty() {
            lines.push("No matching models found.".to_string());
        }

        Ok(text_result(lines.join("\n")))
    }

    async fn tool_list_loras(&self, arguments: Value) -> std::result::Result<Value, String> {
        let args: ListLorasArgs =
            serde_json::from_value(arguments).map_err(|e| format!("invalid arguments: {e}"))?;
        let loras = self
            .client
            .list_loras(args.model.as_deref())
            .await
            .map_err(|e| format!("failed to list LoRAs: {e}"))?;

        if loras.is_empty() {
            let scope = args
                .model
                .as_deref()
                .map(|model| format!(" compatible with {model}"))
                .unwrap_or_default();
            return Ok(json!({
                "content": [{ "type": "text", "text": format!("No installed LoRAs{scope} found.") }],
                "structuredContent": { "loras": [] }
            }));
        }

        let scope = args
            .model
            .as_deref()
            .map(|model| format!(" compatible with {model}"))
            .unwrap_or_default();
        let lines = loras.iter().map(lora_line).collect::<Vec<_>>();
        Ok(json!({
            "content": [{
                "type": "text",
                "text": format!(
                    "Showing {} installed LoRAs{scope}:\n{}",
                    loras.len(),
                    lines.join("\n")
                )
            }],
            "structuredContent": { "loras": loras }
        }))
    }

    async fn tool_server_status(&self) -> std::result::Result<Value, String> {
        let status = self
            .client
            .server_status()
            .await
            .map_err(|e| format!("failed to read server status: {e}"))?;
        let devices = self.client.devices().await.ok();
        let queue = self
            .client
            .list_queue_for_capacity(status.queue_capacity)
            .await
            .ok()
            .map(|mut queue| {
                queue.normalize_planned_lanes_for_presentation();
                queue
            });

        let mut lines = vec![
            format!("mold server {}", status.version),
            format!("busy: {}", status.busy),
            format!("loaded models: {}", status.models_loaded.join(", ")),
            format!("uptime: {}s", status.uptime_secs),
        ];
        if let Some(hostname) = &status.hostname {
            lines.push(format!("host: {hostname}"));
        }
        if let Some(memory) = &status.memory_status {
            lines.push(memory.clone());
        }
        if let Some(device_state) = devices.as_ref().filter(|state| !state.devices.is_empty()) {
            for device in &device_state.devices {
                lines.push(format!(
                    "device {} (gpu {}): {} ({:?}/{:?}){}",
                    device.id,
                    device
                        .ordinal
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "—".into()),
                    device.name,
                    device.admin_state,
                    device.health,
                    device
                        .loaded_models
                        .first()
                        .map(|model| format!(", loaded {model}"))
                        .unwrap_or_default()
                ));
            }
        } else if let Some(gpus) = &status.gpus {
            for gpu in gpus {
                lines.push(format!(
                    "gpu {}: {} ({:?}){}",
                    gpu.ordinal,
                    gpu.name,
                    gpu.state,
                    gpu.loaded_model
                        .as_ref()
                        .map(|m| format!(", loaded {m}"))
                        .unwrap_or_default()
                ));
            }
        }
        append_queue_summary_status_lines(&mut lines, &status, queue.as_ref());
        if let Some(queue) = &queue {
            append_queue_plan_status_lines(&mut lines, queue);
        }

        Ok(json!({
            "content": [{ "type": "text", "text": lines.join("\n") }],
            "structuredContent": {
                "status": status,
                "devices": devices.map(|state| state.devices).unwrap_or_default(),
                "queue": queue,
            }
        }))
    }

    async fn resolve_loras(
        &self,
        loras: Option<Vec<McpLoraArg>>,
    ) -> std::result::Result<Option<Vec<LoraWeight>>, String> {
        let Some(loras) = loras else {
            return Ok(None);
        };
        if loras.is_empty() {
            return Ok(None);
        }

        let refs = loras
            .into_iter()
            .map(normalise_mcp_lora_arg)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let catalog = if refs.iter().any(|lora| is_lora_catalog_id(&lora.value)) {
            self.client
                .list_loras(None)
                .await
                .map_err(|e| format!("failed to resolve LoRA ids: {e}"))?
        } else {
            Vec::new()
        };
        Ok(Some(resolve_mcp_lora_refs(&refs, &catalog)?))
    }
}

#[derive(Debug, Deserialize)]
struct GenerateImageArgs {
    prompt: String,
    model: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    guidance: Option<f64>,
    seed: Option<u64>,
    negative_prompt: Option<String>,
    output_format: Option<String>,
    expand: Option<bool>,
    loras: Option<Vec<McpLoraArg>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum McpLoraArg {
    Ref(String),
    Object(McpLoraObject),
}

#[derive(Debug, Clone, Deserialize)]
struct McpLoraObject {
    id: Option<String>,
    path: Option<String>,
    scale: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct McpLoraRef {
    value: String,
    scale: f64,
}

#[derive(Debug, Deserialize)]
struct ListModelsArgs {
    downloaded_only: Option<bool>,
    generation_only: Option<bool>,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ListLorasArgs {
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GenerationStatusArgs {
    job_id: Option<String>,
    include_result: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct GenerationRetryArgs {
    job_id: String,
}

#[derive(Debug, Deserialize, Default)]
struct ListGalleryArgs {
    query: Option<String>,
    filename: Option<String>,
    model: Option<String>,
    format: Option<String>,
    sort_by: Option<String>,
    sort_order: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

impl ListGalleryArgs {
    fn filter(&self) -> GalleryFilter {
        GalleryFilter {
            query: self.query.clone(),
            filename: self.filename.clone(),
            model: self.model.clone(),
            format: self.format.clone(),
            sort_by: self.sort_by.clone(),
            sort_order: self.sort_order.clone(),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct GetGalleryImageArgs {
    filename: Option<String>,
    latest: Option<bool>,
    query: Option<String>,
    model: Option<String>,
    format: Option<String>,
    sort_by: Option<String>,
    sort_order: Option<String>,
    thumbnail: Option<bool>,
}

impl GetGalleryImageArgs {
    fn filter(&self) -> GalleryFilter {
        GalleryFilter {
            query: self.query.clone(),
            filename: None,
            model: self.model.clone(),
            format: self.format.clone(),
            sort_by: self.sort_by.clone().or_else(|| {
                if self.latest.unwrap_or(true) {
                    Some("timestamp".to_string())
                } else {
                    None
                }
            }),
            sort_order: self.sort_order.clone().or_else(|| {
                if self.latest.unwrap_or(true) {
                    Some("desc".to_string())
                } else {
                    None
                }
            }),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct GalleryFilter {
    query: Option<String>,
    filename: Option<String>,
    model: Option<String>,
    format: Option<String>,
    sort_by: Option<String>,
    sort_order: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AsyncJobStatus {
    Queued,
    Running,
    Held,
    Cancelled,
    Succeeded,
    Failed,
}

impl AsyncJobStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Held => "held",
            Self::Cancelled => "cancelled",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        }
    }

    fn is_terminal(self) -> bool {
        matches!(self, Self::Cancelled | Self::Succeeded | Self::Failed)
    }
}

struct CanonicalAsyncSettlement {
    authority: Option<GenerationBatchAuthority>,
    durable_job_id: Option<String>,
    retryable: Option<bool>,
    status: AsyncJobStatus,
    error: Option<String>,
    result: Option<GenerateResponse>,
    canonical_child: Option<GenerationBatchChild>,
}

impl Default for CanonicalAsyncSettlement {
    fn default() -> Self {
        Self {
            authority: None,
            durable_job_id: None,
            retryable: None,
            status: AsyncJobStatus::Failed,
            error: None,
            result: None,
            canonical_child: None,
        }
    }
}

#[derive(Debug, Clone)]
struct AsyncGenerationJob {
    id: String,
    status: AsyncJobStatus,
    model: String,
    prompt_preview: String,
    created_at_ms: u64,
    updated_at_ms: u64,
    started_at_ms: Option<u64>,
    finished_at_ms: Option<u64>,
    latest_progress: Option<SseProgressEvent>,
    /// Baseline for unfolding the next polled progress snapshot. Never
    /// serialized: it holds a denoise preview image, which has no MCP
    /// consumer and would put a base64 PNG in every status response.
    progress_snapshot: Option<mold_core::queue_progress::QueueJobProgress>,
    error: Option<String>,
    result: Option<GenerateResponse>,
    canonical_child: Option<GenerationBatchChild>,
    result_error: Option<String>,
    authority: Option<GenerationBatchAuthority>,
    durable_job_id: Option<String>,
    retryable: Option<bool>,
    retry_in_flight: bool,
    reconciliation_pending: bool,
}

impl AsyncGenerationJob {
    fn new(id: String, req: &GenerateRequest) -> Self {
        let now = now_ms();
        Self {
            id,
            status: AsyncJobStatus::Queued,
            model: req.model.clone(),
            prompt_preview: prompt_preview(&req.prompt),
            created_at_ms: now,
            updated_at_ms: now,
            started_at_ms: None,
            finished_at_ms: None,
            latest_progress: None,
            progress_snapshot: None,
            error: None,
            result: None,
            canonical_child: None,
            result_error: None,
            authority: None,
            durable_job_id: None,
            retryable: None,
            retry_in_flight: false,
            reconciliation_pending: false,
        }
    }
}

#[derive(Debug, Clone)]
struct CanonicalRetryClaim {
    authority: GenerationBatchAuthority,
    durable_job_id: String,
    /// The child revision this retry is submitted against. `0` when the host
    /// predates the column, which reconciliation reads as "no authority".
    observed_revision: u64,
}

#[derive(Debug)]
struct ReconciliationAttempt {
    next_attempt: Instant,
    backoff: Duration,
    ambiguous_confirmation_attempts: Option<usize>,
    /// The child's `revision` before the retry POST whose response was lost.
    /// A child observed ABOVE it was retried, whatever state it is in now;
    /// one still AT it was not. `0` means the host has no revision
    /// authority, and reconciliation stays state-only as it always was.
    observed_revision: u64,
}

#[derive(Debug, Default)]
struct AsyncJobRegistryInner {
    next_id: AtomicU64,
    jobs: Mutex<HashMap<String, AsyncGenerationJob>>,
    reconciliations: Mutex<HashMap<String, Arc<Mutex<ReconciliationAttempt>>>>,
    hydrations: Mutex<HashMap<String, Weak<Mutex<()>>>>,
}

#[derive(Debug, Clone, Default)]
struct AsyncJobRegistry {
    inner: Arc<AsyncJobRegistryInner>,
}

impl AsyncJobRegistry {
    async fn create(&self, req: &GenerateRequest) -> std::result::Result<String, String> {
        let id = format!(
            "gen-{}",
            self.inner.next_id.fetch_add(1, Ordering::Relaxed) + 1
        );
        let mut jobs = self.inner.jobs.lock().await;
        jobs.insert(id.clone(), AsyncGenerationJob::new(id.clone(), req));
        prune_completed_jobs(&mut jobs);
        if jobs.len() > MAX_ASYNC_JOBS {
            jobs.remove(&id);
            return Err(format!(
                "async generation registry is full ({MAX_ASYNC_JOBS} unresolved jobs); wait for existing work to settle"
            ));
        }
        Ok(id)
    }

    async fn finish(&self, id: &str, result: std::result::Result<GenerateResponse, String>) {
        let mut jobs = self.inner.jobs.lock().await;
        if let Some(job) = jobs.get_mut(id) {
            let now = now_ms();
            job.finished_at_ms = Some(now);
            job.updated_at_ms = now;
            match result {
                Ok(response) => {
                    job.status = AsyncJobStatus::Succeeded;
                    job.result = Some(response);
                    job.error = None;
                }
                Err(err) => {
                    job.status = AsyncJobStatus::Failed;
                    job.error = Some(err);
                    job.result = None;
                }
            }
        }
        prune_completed_jobs(&mut jobs);
    }

    async fn finish_canonical(&self, id: &str, settlement: CanonicalAsyncSettlement) {
        let mut jobs = self.inner.jobs.lock().await;
        if let Some(job) = jobs.get_mut(id) {
            let now = now_ms();
            job.authority = settlement.authority;
            job.durable_job_id = settlement.durable_job_id;
            job.retryable = settlement.retryable;
            job.retry_in_flight = false;
            job.reconciliation_pending = false;
            job.status = settlement.status;
            job.updated_at_ms = now;
            job.finished_at_ms = settlement.status.is_terminal().then_some(now);
            job.error = settlement.error;
            job.result = settlement.result;
            job.canonical_child = settlement.canonical_child;
            job.result_error = None;
        }
        prune_completed_jobs(&mut jobs);
    }

    async fn apply_canonical_event(&self, id: &str, event: CanonicalGenerationEvent) {
        let mut jobs = self.inner.jobs.lock().await;
        let Some(job) = jobs.get_mut(id) else {
            return;
        };
        let (authority, status) = match event {
            CanonicalGenerationEvent::Admitted {
                authority, status, ..
            }
            | CanonicalGenerationEvent::Snapshot {
                authority, status, ..
            } => (authority, status),
            CanonicalGenerationEvent::ReconcileDelayed { authority, error } => {
                job.authority = Some(authority);
                job.error = Some(format!("reconciliation delayed: {error}"));
                job.updated_at_ms = now_ms();
                return;
            }
            CanonicalGenerationEvent::Progress { progress, .. } => {
                let events = progress.events_since(job.progress_snapshot.as_ref());
                job.progress_snapshot = Some(progress);
                // The newest line that is not a preview: an MCP client reads
                // `progress_summary`, and the preview's payload is an image
                // it has nowhere to put.
                if let Some(event) = events
                    .into_iter()
                    .rev()
                    .find(|event| !matches!(event, SseProgressEvent::Preview { .. }))
                {
                    job.latest_progress = Some(event);
                }
                job.updated_at_ms = now_ms();
                return;
            }
        };
        let Some(child) = status.children.first() else {
            job.error = Some("canonical generation status returned no child".into());
            job.updated_at_ms = now_ms();
            return;
        };
        apply_canonical_child_to_job(job, authority, child);
    }

    async fn begin_retry(&self, id: &str) -> std::result::Result<CanonicalRetryClaim, String> {
        let mut jobs = self.inner.jobs.lock().await;
        let job = jobs
            .get_mut(id)
            .ok_or_else(|| format!("unknown async generation job: {id}"))?;
        if job.status != AsyncJobStatus::Held || job.retryable != Some(true) {
            return Err(format!("async generation job {id} is not retryable"));
        }
        if job.retry_in_flight {
            return Err(format!(
                "async generation job {id} already has a retry in progress"
            ));
        }
        let authority = job
            .authority
            .clone()
            .ok_or_else(|| format!("async generation job {id} has no durable batch authority"))?;
        let durable_job_id = job
            .durable_job_id
            .clone()
            .ok_or_else(|| format!("async generation job {id} has no durable job identity"))?;
        job.retry_in_flight = true;
        job.updated_at_ms = now_ms();
        // The registry already holds the child; its revision is the fence
        // an interrupted retry response reconciles against.
        let observed_revision = job
            .canonical_child
            .as_ref()
            .map(|child| child.revision)
            .unwrap_or(0);
        Ok(CanonicalRetryClaim {
            authority,
            durable_job_id,
            observed_revision,
        })
    }

    async fn abort_retry(&self, id: &str, error: String) {
        let mut jobs = self.inner.jobs.lock().await;
        if let Some(job) = jobs.get_mut(id) {
            job.retry_in_flight = false;
            job.error = Some(error);
            job.updated_at_ms = now_ms();
        }
    }

    async fn mark_retry_queued(&self, id: &str, ambiguous: bool, observed_revision: u64) {
        let reconciliation = {
            let mut jobs = self.inner.jobs.lock().await;
            let Some(job) = jobs.get_mut(id) else {
                return;
            };
            let now = now_ms();
            job.status = AsyncJobStatus::Queued;
            job.retryable = Some(false);
            job.retry_in_flight = false;
            job.started_at_ms = None;
            job.finished_at_ms = None;
            job.latest_progress = None;
            job.progress_snapshot = None;
            job.error = None;
            job.result = None;
            job.canonical_child = None;
            job.result_error = None;
            job.reconciliation_pending = true;
            job.updated_at_ms = now;
            job.authority
                .as_ref()
                .zip(job.durable_job_id.as_deref())
                .map(|(authority, durable_job_id)| reconciliation_key(authority, durable_job_id))
        };
        if let Some(key) = reconciliation {
            self.inner.reconciliations.lock().await.insert(
                key,
                Arc::new(Mutex::new(ReconciliationAttempt {
                    next_attempt: Instant::now(),
                    backoff: reconciliation_initial_backoff(),
                    ambiguous_confirmation_attempts: ambiguous.then_some(0),
                    observed_revision,
                })),
            );
        }
    }

    async fn reconcile_pending_job(&self, client: &MoldClient, id: &str) {
        let pending = {
            let jobs = self.inner.jobs.lock().await;
            let Some(job) = jobs.get(id) else {
                return;
            };
            if !job.reconciliation_pending {
                return;
            }
            job.authority.clone().zip(job.durable_job_id.clone())
        };
        let Some((authority, durable_job_id)) = pending else {
            return;
        };
        let key = reconciliation_key(&authority, &durable_job_id);
        let attempt = {
            let mut attempts = self.inner.reconciliations.lock().await;
            Arc::clone(attempts.entry(key.clone()).or_insert_with(|| {
                Arc::new(Mutex::new(ReconciliationAttempt {
                    next_attempt: Instant::now(),
                    backoff: reconciliation_initial_backoff(),
                    ambiguous_confirmation_attempts: None,
                    observed_revision: 0,
                }))
            }))
        };
        let attempt_identity = Arc::clone(&attempt);
        let Ok(mut attempt) = attempt.try_lock_owned() else {
            return;
        };
        if Instant::now() < attempt.next_attempt {
            return;
        }
        let confirmation_attempt = attempt
            .ambiguous_confirmation_attempts
            .map(|count| count + 1);
        let observed_revision = attempt.observed_revision;

        enum ReadOutcome {
            Found(Box<GenerationBatchChild>),
            Retry(String),
            Stop(String),
        }
        let read = tokio::time::timeout(
            reconciliation_read_timeout(),
            client.generation_batch(&authority.batch_id),
        )
        .await;
        let outcome = match read {
            Err(_) => ReadOutcome::Retry(format!(
                "generation batch {} status read timed out",
                authority.batch_id
            )),
            Ok(result) => match result {
            Ok(Some(status)) => match authority.validate_status(&status) {
                Ok(()) => match status
                    .children
                    .iter()
                    .find(|child| child.job_id == durable_job_id)
                    .cloned()
                {
                    Some(child) => ReadOutcome::Found(Box::new(child)),
                    None => ReadOutcome::Stop(format!(
                        "generation batch lost durable job {durable_job_id} during retry reconciliation"
                    )),
                },
                Err(error) => ReadOutcome::Stop(error),
            },
            Ok(None) => ReadOutcome::Retry(format!(
                "generation batch {} is not visible yet",
                authority.batch_id
            )),
            Err(error) if is_transient_request_error(&error) => {
                ReadOutcome::Retry(error.to_string())
            }
            Err(error) => ReadOutcome::Stop(error.to_string()),
            },
        };
        if let Some(confirmation_attempt) = confirmation_attempt {
            attempt.ambiguous_confirmation_attempts = Some(confirmation_attempt);
        }

        let mut confirmation_finished = false;
        let keep_polling = match outcome {
            ReadOutcome::Found(child) => {
                let child = *child;
                // A revision past the pre-POST one proves the retry landed,
                // whatever state the child is in now — a re-held child at a
                // higher revision was retried and held again for a fresh
                // reason. Short-circuit rather than burning five polls
                // waiting for a state change that already happened.
                let retry_landed = observed_revision > 0 && child.revision > observed_revision;
                if child.state == GenerationBatchChildState::Held
                    && !retry_landed
                    && confirmation_attempt
                        .is_some_and(|attempt| attempt < AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS)
                {
                    self.record_reconciliation_error(
                        id,
                        format!(
                            "retry remains locked; confirmation attempt {}/{} still sees the pre-commit held state",
                            confirmation_attempt.expect("checked above"),
                            AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS
                        ),
                    )
                    .await;
                    true
                } else if matches!(
                    child.state,
                    GenerationBatchChildState::Accepted
                        | GenerationBatchChildState::Running
                        | GenerationBatchChildState::Cancelling
                ) {
                    confirmation_finished = confirmation_attempt.is_some();
                    let mut jobs = self.inner.jobs.lock().await;
                    if let Some(job) = jobs.get_mut(id) {
                        apply_canonical_child_to_job(job, authority.clone(), &child);
                        job.reconciliation_pending = true;
                    }
                    true
                } else {
                    confirmation_finished = confirmation_attempt.is_some();
                    // Settling a still-held child at the SAME revision we
                    // submitted against is not the retry's outcome: that
                    // retry never landed, and the original POST may yet
                    // commit. The job stays retryable either way, so the
                    // settled error must say which of the two happened
                    // rather than presenting the pre-retry snapshot as the
                    // retry's result. It is composed onto the child's own
                    // error because `finish_canonical_child` lets the
                    // child's error win over any separate note.
                    let mut child = child;
                    if confirmation_attempt.is_some()
                        && child.state == GenerationBatchChildState::Held
                        && observed_revision > 0
                        && !retry_landed
                    {
                        let unlanded = format!(
                            "the retry did not reach durable job {durable_job_id}: it remains held at revision {} after {AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS} bounded attempts; retry it again once the cause is corrected",
                            child.revision
                        );
                        child.error = Some(match child.error.take() {
                            Some(existing) => format!("{existing}; {unlanded}"),
                            None => unlanded,
                        });
                    }
                    finish_canonical_child(self, id, authority, child, None).await;
                    false
                }
            }
            ReadOutcome::Retry(error) => {
                let confirmation_expired = confirmation_attempt
                    .is_some_and(|attempt| attempt >= AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS);
                confirmation_finished = confirmation_expired;
                self.record_reconciliation_error(
                    id,
                    if confirmation_expired {
                        format!(
                            "ambiguous retry remains unconfirmed after {AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS} bounded attempts; retry lock retained while exact reconciliation continues: {error}"
                        )
                    } else {
                        format!("retry remains locked; exact reconciliation delayed: {error}")
                    },
                )
                .await;
                true
            }
            ReadOutcome::Stop(error) => {
                self.record_reconciliation_error(
                    id,
                    format!("retry remains locked; exact reconciliation stopped: {error}"),
                )
                .await;
                let mut jobs = self.inner.jobs.lock().await;
                if let Some(job) = jobs.get_mut(id) {
                    job.reconciliation_pending = false;
                }
                false
            }
        };

        if keep_polling {
            if confirmation_attempt.is_some() && !confirmation_finished {
                attempt.next_attempt = Instant::now() + reconciliation_initial_backoff();
            } else {
                if confirmation_finished {
                    attempt.ambiguous_confirmation_attempts = None;
                    attempt.backoff = reconciliation_initial_backoff();
                }
                attempt.next_attempt = Instant::now() + attempt.backoff;
                attempt.backoff = attempt
                    .backoff
                    .saturating_mul(2)
                    .min(reconciliation_max_backoff());
            }
        } else {
            drop(attempt);
            let mut attempts = self.inner.reconciliations.lock().await;
            if attempts
                .get(&key)
                .is_some_and(|current| Arc::ptr_eq(current, &attempt_identity))
            {
                attempts.remove(&key);
            }
        }
    }

    async fn record_reconciliation_error(&self, id: &str, error: String) {
        let mut jobs = self.inner.jobs.lock().await;
        if let Some(job) = jobs.get_mut(id) {
            job.error = Some(error);
            job.updated_at_ms = now_ms();
        }
    }

    async fn hydrate_result(&self, client: &MoldClient, id: &str) {
        let pending = {
            let jobs = self.inner.jobs.lock().await;
            let Some(job) = jobs.get(id) else {
                return;
            };
            if job.status != AsyncJobStatus::Succeeded || job.result.is_some() {
                return;
            }
            job.authority
                .clone()
                .zip(job.durable_job_id.clone())
                .zip(job.canonical_child.clone())
        };
        let Some(((authority, durable_job_id), child)) = pending else {
            return;
        };
        let key = reconciliation_key(&authority, &durable_job_id);
        let hydration = {
            let mut hydrations = self.inner.hydrations.lock().await;
            hydrations.retain(|_, hydration| hydration.strong_count() > 0);
            match hydrations.get(&key).and_then(Weak::upgrade) {
                Some(hydration) => hydration,
                None => {
                    let hydration = Arc::new(Mutex::new(()));
                    hydrations.insert(key.clone(), Arc::downgrade(&hydration));
                    hydration
                }
            }
        };
        let hydration_identity = Arc::clone(&hydration);
        let Ok(hydration) = hydration.try_lock_owned() else {
            return;
        };

        let result = tokio::time::timeout(
            reconciliation_read_timeout(),
            hydrate_canonical_outcome(client, &child),
        )
        .await;
        {
            let mut jobs = self.inner.jobs.lock().await;
            if let Some(job) = jobs.get_mut(id) {
                if job.status == AsyncJobStatus::Succeeded
                    && job.durable_job_id.as_deref() == Some(durable_job_id.as_str())
                {
                    match result {
                        Ok(Ok(response)) => {
                            job.result = Some(response);
                            job.result_error = None;
                        }
                        Ok(Err(error)) => {
                            job.result_error = Some(format!(
                                "result hydration delayed; request the result again to retry: {error}"
                            ));
                        }
                        Err(_) => {
                            job.result_error = Some(
                                "result hydration timed out; request the result again to retry"
                                    .into(),
                            );
                        }
                    }
                    job.updated_at_ms = now_ms();
                }
            }
        }
        drop(hydration);
        let mut hydrations = self.inner.hydrations.lock().await;
        if hydrations
            .get(&key)
            .and_then(Weak::upgrade)
            .is_some_and(|current| Arc::ptr_eq(&current, &hydration_identity))
        {
            hydrations.remove(&key);
        }
    }

    async fn eligible_pending_reconciliations(&self, limit: usize) -> Vec<String> {
        let mut candidates = {
            let jobs = self.inner.jobs.lock().await;
            jobs.values()
                .filter(|job| job.reconciliation_pending)
                .filter_map(|job| {
                    let authority = job.authority.as_ref()?;
                    let durable_job_id = job.durable_job_id.as_deref()?;
                    Some((
                        job.updated_at_ms,
                        job.id.clone(),
                        reconciliation_key(authority, durable_job_id),
                    ))
                })
                .collect::<Vec<_>>()
        };
        candidates.sort_by_key(|(updated_at_ms, _, _)| *updated_at_ms);
        let now = Instant::now();
        let attempts = self.inner.reconciliations.lock().await;
        candidates
            .into_iter()
            .filter(|(_, _, key)| {
                attempts.get(key).is_none_or(|attempt| {
                    attempt
                        .try_lock()
                        .is_ok_and(|attempt| now >= attempt.next_attempt)
                })
            })
            .take(limit)
            .map(|(_, id, _)| id)
            .collect()
    }

    async fn get(&self, id: &str) -> Option<AsyncGenerationJob> {
        self.inner.jobs.lock().await.get(id).cloned()
    }

    async fn list(&self) -> Vec<AsyncGenerationJob> {
        let mut jobs = self
            .inner
            .jobs
            .lock()
            .await
            .values()
            .cloned()
            .collect::<Vec<_>>();
        jobs.sort_by_key(|job| job.created_at_ms);
        jobs
    }
}

fn async_status_from_child(state: &GenerationBatchChildState) -> AsyncJobStatus {
    match state {
        GenerationBatchChildState::Accepted => AsyncJobStatus::Queued,
        GenerationBatchChildState::Running | GenerationBatchChildState::Cancelling => {
            AsyncJobStatus::Running
        }
        GenerationBatchChildState::Held => AsyncJobStatus::Held,
        GenerationBatchChildState::Cancelled => AsyncJobStatus::Cancelled,
        GenerationBatchChildState::Complete => AsyncJobStatus::Succeeded,
        GenerationBatchChildState::Failed => AsyncJobStatus::Failed,
    }
}

fn apply_canonical_child_to_job(
    job: &mut AsyncGenerationJob,
    authority: GenerationBatchAuthority,
    child: &GenerationBatchChild,
) {
    let now = now_ms();
    job.authority = Some(authority);
    job.durable_job_id = Some(child.job_id.clone());
    job.retryable = child.retryable;
    job.status = async_status_from_child(&child.state);
    job.error = child.error.clone();
    job.canonical_child = Some(child.clone());
    if job.status != AsyncJobStatus::Succeeded {
        job.result = None;
        job.result_error = None;
    }
    job.updated_at_ms = now;
    if job.status == AsyncJobStatus::Running {
        job.started_at_ms.get_or_insert(now);
    }
    job.finished_at_ms = job.status.is_terminal().then_some(now);
}

fn prune_completed_jobs(jobs: &mut HashMap<String, AsyncGenerationJob>) {
    if jobs.len() <= MAX_ASYNC_JOBS {
        return;
    }

    let mut removable = jobs
        .values()
        .filter(|job| job.status.is_terminal())
        .map(|job| {
            (
                job.finished_at_ms.unwrap_or(job.updated_at_ms),
                job.id.clone(),
            )
        })
        .collect::<Vec<_>>();
    removable.sort_by_key(|(ts, _)| *ts);

    let remove_count = jobs.len().saturating_sub(MAX_ASYNC_JOBS);
    for (_, id) in removable.into_iter().take(remove_count) {
        jobs.remove(&id);
    }
}

fn job_tool_result(job: &AsyncGenerationJob, include_result: bool) -> Value {
    let mut content = vec![json!({ "type": "text", "text": job_status_line(job) })];
    if include_result {
        if let Some(response) = &job.result {
            if let Some(image) = response.images.first() {
                content.push(json!({
                    "type": "image",
                    "data": general_purpose::STANDARD.encode(&image.data),
                    "mimeType": image.format.content_type()
                }));
            }
        }
    }

    json!({
        "content": content,
        "structuredContent": job_summary_json(job)
    })
}

fn job_status_line(job: &AsyncGenerationJob) -> String {
    let mut line = format!("{}: {} model {}", job.id, job.status.as_str(), job.model);

    if let Some(durable_job_id) = &job.durable_job_id {
        line.push_str(&format!("; durable job {durable_job_id}"));
    }
    if job.status == AsyncJobStatus::Held && job.retryable == Some(true) {
        line.push_str("; retryable after correcting the cause");
    }
    if let Some(progress) = &job.latest_progress {
        line.push_str(&format!("; {}", progress_summary(progress)));
    }
    if let Some(error) = &job.error {
        line.push_str(&format!("; error: {error}"));
    }
    if let Some(error) = &job.result_error {
        line.push_str(&format!("; {error}"));
    }
    if let Some(response) = &job.result {
        if let Some(image) = response.images.first() {
            line.push_str(&format!(
                "; generated {}x{} {}; seed {}",
                image.width,
                image.height,
                image.format.extension(),
                response.seed_used
            ));
            if response.generation_time_ms > 0 {
                line.push_str(&format!(
                    "; generation time {:.1}s",
                    response.generation_time_ms as f64 / 1000.0
                ));
            }
            if let Some(gpu) = response.gpu {
                line.push_str(&format!("; gpu {gpu}"));
            }
        }
    }

    line
}

fn job_summary_json(job: &AsyncGenerationJob) -> Value {
    let result = job.result.as_ref().map(|response| {
        let image = response.images.first();
        json!({
            "model": response.model,
            "seed_used": response.seed_used,
            "generation_time_ms": (response.generation_time_ms > 0)
                .then_some(response.generation_time_ms),
            "gpu": response.gpu,
            "image": image.map(|image| json!({
                "width": image.width,
                "height": image.height,
                "format": image.format.extension(),
                "index": image.index
            }))
        })
    });

    json!({
        "job_id": job.id,
        "status": job.status.as_str(),
        "model": job.model,
        "prompt_preview": job.prompt_preview,
        "created_at_ms": job.created_at_ms,
        "updated_at_ms": job.updated_at_ms,
        "started_at_ms": job.started_at_ms,
        "finished_at_ms": job.finished_at_ms,
        "latest_progress": job.latest_progress,
        "expected_instance_id": job.authority.as_ref().map(|authority| &authority.instance_id),
        "server_batch_id": job.authority.as_ref().map(|authority| &authority.batch_id),
        "client_batch_id": job.authority.as_ref().map(|authority| &authority.client_batch_id),
        "durable_job_id": job.durable_job_id,
        "retryable": job.retryable,
        "retry_in_flight": job.retry_in_flight,
        "error": job.error,
        "result_error": job.result_error,
        "result": result,
    })
}

fn progress_summary(event: &SseProgressEvent) -> String {
    match event {
        SseProgressEvent::DependencyWait { dependency, reason } => {
            format!("waiting for dependency {dependency}: {reason}")
        }
        SseProgressEvent::Preview { step, total, .. } => format!("denoise preview {step}/{total}"),
        SseProgressEvent::StageStart { name } => format!("stage started: {name}"),
        SseProgressEvent::StageDone { name, elapsed_ms } => {
            format!("stage done: {name} in {:.1}s", *elapsed_ms as f64 / 1000.0)
        }
        SseProgressEvent::StageProgress {
            name,
            current,
            total,
        } => format!("{name} {current}/{total}"),
        SseProgressEvent::Info { message } => message.clone(),
        SseProgressEvent::CacheHit { resource } => format!("cache hit: {resource}"),
        SseProgressEvent::DenoiseStep {
            step,
            total,
            elapsed_ms,
        } => format!(
            "denoise step {step}/{total} after {:.1}s",
            *elapsed_ms as f64 / 1000.0
        ),
        SseProgressEvent::DownloadProgress {
            filename,
            file_index,
            total_files,
            bytes_downloaded,
            bytes_total,
            ..
        } => format!(
            "downloading {filename} ({file_index}/{total_files}, {} / {})",
            human_bytes(*bytes_downloaded),
            human_bytes(*bytes_total)
        ),
        SseProgressEvent::DownloadDone {
            filename,
            file_index,
            total_files,
            ..
        } => format!("downloaded {filename} ({file_index}/{total_files})"),
        SseProgressEvent::PullComplete { model } => format!("pull complete: {model}"),
        SseProgressEvent::Queued { position, id } => {
            if id.is_empty() {
                format!("queued at position {position}")
            } else {
                format!("queued at position {position}; server job {id}")
            }
        }
        SseProgressEvent::WeightLoad {
            bytes_loaded,
            bytes_total,
            component,
        } => format!(
            "loading {component} ({} / {})",
            human_bytes(*bytes_loaded),
            human_bytes(*bytes_total)
        ),
    }
}

fn human_bytes(bytes: u64) -> String {
    mold_core::format::human_bytes(bytes)
}

fn prompt_preview(prompt: &str) -> String {
    const MAX_CHARS: usize = 120;
    let mut chars = prompt.chars();
    let preview = chars.by_ref().take(MAX_CHARS).collect::<String>();
    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        preview
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn filter_sort_gallery(
    mut images: Vec<GalleryImage>,
    filter: &GalleryFilter,
) -> std::result::Result<Vec<GalleryImage>, String> {
    let format = filter
        .format
        .as_deref()
        .map(parse_gallery_format)
        .transpose()?;

    images.retain(|image| {
        filter
            .query
            .as_deref()
            .is_none_or(|query| gallery_matches_query(image, query))
            && filter
                .filename
                .as_deref()
                .is_none_or(|name| contains_case_insensitive(&image.filename, name))
            && filter
                .model
                .as_deref()
                .is_none_or(|model| contains_case_insensitive(&image.metadata.model, model))
            && format.is_none_or(|format| gallery_format(image) == Some(format))
    });

    let sort_by = filter.sort_by.as_deref().unwrap_or("timestamp");
    match sort_by {
        "timestamp" | "date" | "time" | "created_at" => {
            images.sort_by_key(|image| image.timestamp);
        }
        "filename" | "name" => {
            images.sort_by(|a, b| a.filename.cmp(&b.filename));
        }
        "model" => {
            images.sort_by(|a, b| a.metadata.model.cmp(&b.metadata.model));
        }
        "prompt" | "description" => {
            images.sort_by(|a, b| a.metadata.prompt.cmp(&b.metadata.prompt));
        }
        "size" | "size_bytes" => {
            images.sort_by_key(|image| image.size_bytes.unwrap_or(0));
        }
        other => {
            return Err(format!(
                "unsupported sort_by '{other}'; use timestamp, filename, model, prompt, or size"
            ));
        }
    }

    let descending = match filter.sort_order.as_deref().unwrap_or("desc") {
        "desc" | "descending" => true,
        "asc" | "ascending" => false,
        other => {
            return Err(format!("unsupported sort_order '{other}'; use asc or desc"));
        }
    };
    if descending {
        images.reverse();
    }

    Ok(images)
}

fn gallery_matches_query(image: &GalleryImage, query: &str) -> bool {
    contains_case_insensitive(&image.filename, query)
        || contains_case_insensitive(&image.metadata.prompt, query)
        || image
            .metadata
            .original_prompt
            .as_deref()
            .is_some_and(|text| contains_case_insensitive(text, query))
        || image
            .metadata
            .negative_prompt
            .as_deref()
            .is_some_and(|text| contains_case_insensitive(text, query))
        || contains_case_insensitive(&image.metadata.model, query)
        || image
            .metadata
            .lora
            .as_deref()
            .is_some_and(|text| contains_case_insensitive(text, query))
        || contains_case_insensitive(&image.metadata.version, query)
}

fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    haystack.to_lowercase().contains(&needle.to_lowercase())
}

fn parse_gallery_format(format: &str) -> std::result::Result<OutputFormat, String> {
    match format
        .trim()
        .trim_start_matches('.')
        .to_lowercase()
        .as_str()
    {
        "png" => Ok(OutputFormat::Png),
        "jpg" | "jpeg" => Ok(OutputFormat::Jpeg),
        "gif" => Ok(OutputFormat::Gif),
        "apng" => Ok(OutputFormat::Apng),
        "webp" => Ok(OutputFormat::Webp),
        "mp4" => Ok(OutputFormat::Mp4),
        other => Err(format!(
            "unsupported format '{other}'; use png, jpeg, gif, apng, webp, or mp4"
        )),
    }
}

fn gallery_format(image: &GalleryImage) -> Option<OutputFormat> {
    image.format.or_else(|| {
        extension_for_filename(&image.filename).and_then(|ext| parse_gallery_format(ext).ok())
    })
}

fn extension_for_filename(filename: &str) -> Option<&str> {
    filename.rsplit_once('.').map(|(_, ext)| ext)
}

fn content_type_for_filename(filename: &str) -> &'static str {
    extension_for_filename(filename)
        .and_then(|ext| parse_gallery_format(ext).ok())
        .map(|format| format.content_type())
        .unwrap_or("application/octet-stream")
}

fn gallery_line(image: &GalleryImage) -> String {
    let format = gallery_format(image)
        .map(|format| format.extension().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let size = image
        .size_bytes
        .map(human_bytes)
        .unwrap_or_else(|| "unknown size".to_string());
    format!(
        "{} [{}; {size}] {} model {}; {}x{}, steps {}, seed {}; prompt: {}",
        image.filename,
        format,
        format_gallery_timestamp(image.timestamp),
        image.metadata.model,
        image.metadata.width,
        image.metadata.height,
        image.metadata.steps,
        image.metadata.seed,
        prompt_preview(&image.metadata.prompt)
    )
}

fn gallery_summary_json(image: &GalleryImage) -> Value {
    json!({
        "filename": image.filename,
        "timestamp": image.timestamp,
        "datetime": format_gallery_timestamp(image.timestamp),
        "format": gallery_format(image).map(|format| format.extension()),
        "size_bytes": image.size_bytes,
        "metadata_synthetic": image.metadata_synthetic,
        "metadata": image.metadata,
    })
}

fn lora_line(lora: &LoraInfo) -> String {
    let author = lora
        .author
        .as_deref()
        .map(|author| format!(" by {author}"))
        .unwrap_or_default();
    let triggers = if lora.trained_words.is_empty() {
        String::new()
    } else {
        format!("; triggers: {}", lora.trained_words.join(", "))
    };
    format!(
        "- {}{} [{}] {}; path: {}{}",
        lora.name, author, lora.family, lora.id, lora.path, triggers
    )
}

fn format_gallery_timestamp(timestamp: u64) -> String {
    let secs = if timestamp > 1_000_000_000_000 {
        timestamp / 1000
    } else {
        timestamp
    };
    time::OffsetDateTime::from_unix_timestamp(secs as i64)
        .ok()
        .and_then(|dt| {
            dt.format(&time::format_description::well_known::Rfc3339)
                .ok()
        })
        .unwrap_or_else(|| secs.to_string())
}

fn normalise_mcp_lora_arg(arg: McpLoraArg) -> std::result::Result<McpLoraRef, String> {
    match arg {
        McpLoraArg::Ref(value) => {
            let value = value.trim().to_string();
            if value.is_empty() {
                return Err("loras entries must not be empty".to_string());
            }
            Ok(McpLoraRef { value, scale: 1.0 })
        }
        McpLoraArg::Object(object) => {
            let id = non_empty_lora_ref(object.id);
            let path = non_empty_lora_ref(object.path);
            let value = match (id, path) {
                (Some(_), Some(_)) => {
                    return Err(
                        "each loras object must provide either id or path, not both".to_string()
                    );
                }
                (Some(id), None) => id,
                (None, Some(path)) => path,
                (None, None) => {
                    return Err("each loras object must provide an id or path".to_string());
                }
            };
            let scale = object.scale.unwrap_or(1.0);
            if !scale.is_finite() {
                return Err("LoRA scale must be a finite number".to_string());
            }
            Ok(McpLoraRef { value, scale })
        }
    }
}

fn non_empty_lora_ref(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let value = value.trim().to_string();
        (!value.is_empty()).then_some(value)
    })
}

fn is_lora_catalog_id(value: &str) -> bool {
    value.starts_with("cv:") || value.starts_with("hf:")
}

fn resolve_mcp_lora_refs(
    refs: &[McpLoraRef],
    catalog: &[LoraInfo],
) -> std::result::Result<Vec<LoraWeight>, String> {
    refs.iter()
        .map(|lora| {
            let path = if is_lora_catalog_id(&lora.value) {
                catalog
                    .iter()
                    .find(|candidate| candidate.id == lora.value)
                    .map(|candidate| candidate.path.clone())
                    .ok_or_else(|| {
                        format!(
                            "installed LoRA id '{}' was not found; use list_loras to see installed ids",
                            lora.value
                        )
                    })?
            } else {
                lora.value.clone()
            };
            Ok(LoraWeight {
                path,
                scale: lora.scale,
                expert: None,
            })
        })
        .collect()
}

fn build_generate_request(
    args: GenerateImageArgs,
    loras: Option<Vec<LoraWeight>>,
) -> std::result::Result<GenerateRequest, String> {
    // The MCP generate tools take no visual conditioning (no source image,
    // keyframes, source video, or extend), so the optional-prompt path in
    // `mold_core::prompt_required_for` can never apply here. If a conditioning
    // input is ever added, relax this check and the `"required": ["prompt"]`
    // arrays in `tool_definitions` together.
    if args.prompt.trim().is_empty() {
        return Err("prompt must not be empty".to_string());
    }

    let output_format = match args.output_format.as_deref().unwrap_or("png") {
        "png" => OutputFormat::Png,
        "jpeg" | "jpg" => OutputFormat::Jpeg,
        other => {
            return Err(format!(
                "unsupported output_format '{other}'; use png or jpeg"
            ))
        }
    };

    let mut config = Config::load_or_default();
    let model = args
        .model
        .unwrap_or_else(|| config.resolved_default_model());
    if crate::catalog_bridge::looks_like_catalog_id(&model) {
        crate::catalog_bridge::install_catalog_model_from_installed_sidecar(&mut config, &model)
            .map_err(|e| format!("failed to resolve installed catalog model '{model}': {e}"))?;
    }
    let model_cfg = config.resolved_model_config(&model);
    let width = args
        .width
        .unwrap_or_else(|| model_cfg.effective_width(&config));
    let height = args
        .height
        .unwrap_or_else(|| model_cfg.effective_height(&config));

    if width == 0 || height == 0 {
        return Err("width and height must be greater than zero".to_string());
    }
    if width & 15 != 0 || height & 15 != 0 {
        return Err("width and height must be multiples of 16".to_string());
    }

    Ok(GenerateRequest {
        collection: None,
        tags: None,
        title: None,
        source_fit: None,
        hdr_exr_dir: None,
        hdr_exr_full_float: false,
        guidance_overrides: None,
        sample_shift: None,
        distill_strength_high: None,
        distill_strength_low: None,
        prompt: args.prompt,
        negative_prompt: args.negative_prompt,
        model,
        width,
        height,
        steps: args
            .steps
            .unwrap_or_else(|| model_cfg.effective_steps(&config)),
        guidance: args
            .guidance
            .unwrap_or_else(|| model_cfg.effective_guidance()),
        seed: args.seed,
        batch_size: 1,
        output_format: Some(output_format),
        embed_metadata: Some(config.effective_embed_metadata(None)),
        scheduler: None,
        cfg_plus: None,
        source_image: None,
        source_image_name: None,
        edit_images: None,
        references: None,
        strength: 0.75,
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: args.expand,
        original_prompt: None,
        prompt_transform: None,
        batch_id: None,
        batch_index: None,
        batch_count: None,
        lora: None,
        frames: None,
        fps: None,
        upscale_model: None,
        gif_preview: false,
        enable_audio: None,
        audio_file: None,
        audio_file_path: None,
        source_video: None,
        source_video_path: None,
        extend_video: None,
        extend_video_path: None,
        extend_overlap_frames: None,
        keyframes: None,
        pipeline: None,
        ic_lora_control: None,
        loras,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: None,
        id_image: None,
        id_image_name: None,
        id_weight: None,
        id_start_step: None,
        id_images: None,
        id_image_names: None,
        true_cfg: None,
        cfg_start_step: None,
    })
}

fn handle_protocol_message(message: Value) -> Option<Value> {
    let id = message.get("id").cloned();
    let method = message.get("method").and_then(Value::as_str)?;

    match (method, id) {
        ("notifications/initialized", _) => None,
        (_, None) => None,
        ("initialize", Some(id)) => Some(response(
            id,
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "mold",
                    "version": mold_core::build_info::version_string()
                }
            }),
        )),
        ("ping", Some(id)) => Some(response(id, json!({}))),
        ("tools/list", Some(id)) => Some(response(id, json!({ "tools": tool_definitions() }))),
        ("prompts/list", Some(id)) => Some(response(id, json!({ "prompts": [] }))),
        ("resources/list", Some(id)) => Some(response(id, json!({ "resources": [] }))),
        (other, Some(id)) => Some(error_response(
            id,
            -32601,
            format!("method not found: {other}"),
        )),
    }
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "generate_image",
            "description": "Generate one image with mold. Requires a running mold serve process, unless MOLD_HOST points at a remote mold server.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The image prompt."
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional mold model name, such as flux2-klein:q8. Defaults to mold's configured default model."
                    },
                    "width": {
                        "type": "integer",
                        "minimum": 16,
                        "multipleOf": 16,
                        "description": "Image width in pixels."
                    },
                    "height": {
                        "type": "integer",
                        "minimum": 16,
                        "multipleOf": 16,
                        "description": "Image height in pixels."
                    },
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Inference step count."
                    },
                    "guidance": {
                        "type": "number",
                        "description": "Guidance scale."
                    },
                    "seed": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional deterministic seed."
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Optional negative prompt for CFG-capable model families."
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["png", "jpeg", "jpg"],
                        "description": "Output image format. Defaults to png."
                    },
                    "expand": {
                        "type": "boolean",
                        "description": "Ask the mold server to expand the prompt before generation."
                    },
                    "loras": {
                        "type": "array",
                        "description": "Optional LoRA stack to apply. Each item can be an installed LoRA id from list_loras, a server-side path, or an object with id/path and optional scale.",
                        "items": {
                            "oneOf": [
                                {
                                    "type": "string",
                                    "description": "Installed LoRA id such as cv:827325, or a server-side .safetensors path. Scale defaults to 1.0."
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "id": {
                                            "type": "string",
                                            "description": "Installed LoRA catalog id, such as cv:827325."
                                        },
                                        "path": {
                                            "type": "string",
                                            "description": "Server-side path to a LoRA .safetensors file."
                                        },
                                        "scale": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 2.0,
                                            "description": "LoRA strength. Defaults to 1.0."
                                        }
                                    },
                                    "additionalProperties": false
                                }
                            ]
                        }
                    }
                },
                "required": ["prompt"],
                "additionalProperties": false
            }
        },
        {
            "name": "generate_image_async",
            "description": "Start an image generation job and return immediately with a job id. Use generation_status to poll progress and fetch the completed image.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The image prompt."
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional mold model name, such as flux2-klein:q8. Defaults to mold's configured default model."
                    },
                    "width": {
                        "type": "integer",
                        "minimum": 16,
                        "multipleOf": 16,
                        "description": "Image width in pixels."
                    },
                    "height": {
                        "type": "integer",
                        "minimum": 16,
                        "multipleOf": 16,
                        "description": "Image height in pixels."
                    },
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Inference step count."
                    },
                    "guidance": {
                        "type": "number",
                        "description": "Guidance scale."
                    },
                    "seed": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional deterministic seed."
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Optional negative prompt for CFG-capable model families."
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["png", "jpeg", "jpg"],
                        "description": "Output image format. Defaults to png."
                    },
                    "expand": {
                        "type": "boolean",
                        "description": "Ask the mold server to expand the prompt before generation."
                    },
                    "loras": {
                        "type": "array",
                        "description": "Optional LoRA stack to apply. Each item can be an installed LoRA id from list_loras, a server-side path, or an object with id/path and optional scale.",
                        "items": {
                            "oneOf": [
                                {
                                    "type": "string",
                                    "description": "Installed LoRA id such as cv:827325, or a server-side .safetensors path. Scale defaults to 1.0."
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "id": {
                                            "type": "string",
                                            "description": "Installed LoRA catalog id, such as cv:827325."
                                        },
                                        "path": {
                                            "type": "string",
                                            "description": "Server-side path to a LoRA .safetensors file."
                                        },
                                        "scale": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 2.0,
                                            "description": "LoRA strength. Defaults to 1.0."
                                        }
                                    },
                                    "additionalProperties": false
                                }
                            ]
                        }
                    }
                },
                "required": ["prompt"],
                "additionalProperties": false
            }
        },
        {
            "name": "generation_status",
            "description": "List async generation jobs or return status for one job. A single job returns its completed image unless include_result is false.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Async generation job id returned by generate_image_async. Omit to list tracked jobs."
                    },
                    "include_result": {
                        "type": "boolean",
                        "description": "Include completed image content when job_id is provided. Defaults to true; pass false to poll state only. A failed or timed-out hydration can be retried by requesting it again."
                    }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "generation_retry",
            "description": "Retry one exact held durable generation after correcting its reported cause. The server instance, batch, client id, and durable child are validated before retry.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Async generation job id returned by generate_image_async. The job must be held and retryable."
                    }
                },
                "required": ["job_id"],
                "additionalProperties": false
            }
        },
        {
            "name": "list_gallery",
            "description": "List, search, filter, and sort saved mold gallery items.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Case-insensitive search across filename, prompt/description, model, LoRA, and version."
                    },
                    "filename": {
                        "type": "string",
                        "description": "Case-insensitive filename substring filter."
                    },
                    "model": {
                        "type": "string",
                        "description": "Case-insensitive model substring filter."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["png", "jpeg", "jpg", "gif", "apng", "webp", "mp4"],
                        "description": "Filter by output format."
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["timestamp", "date", "time", "created_at", "filename", "name", "model", "prompt", "description", "size", "size_bytes"],
                        "description": "Sort field. Defaults to timestamp."
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["desc", "descending", "asc", "ascending"],
                        "description": "Sort direction. Defaults to desc."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Maximum rows to return. Defaults to 25."
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Rows to skip after filtering/sorting. Defaults to 0."
                    }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "get_gallery_image",
            "description": "Fetch a saved gallery image by filename, or fetch the latest item matching search/filter arguments.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Exact gallery filename to fetch. If omitted, the latest matching item is fetched."
                    },
                    "latest": {
                        "type": "boolean",
                        "description": "When filename is omitted, choose the latest matching item. Defaults to true."
                    },
                    "query": {
                        "type": "string",
                        "description": "Case-insensitive search across filename, prompt/description, model, LoRA, and version."
                    },
                    "model": {
                        "type": "string",
                        "description": "Case-insensitive model substring filter."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["png", "jpeg", "jpg", "gif", "apng", "webp", "mp4"],
                        "description": "Filter by output format."
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["timestamp", "date", "time", "created_at", "filename", "name", "model", "prompt", "description", "size", "size_bytes"],
                        "description": "Sort field used when filename is omitted. Defaults to timestamp."
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["desc", "descending", "asc", "ascending"],
                        "description": "Sort direction used when filename is omitted. Defaults to desc."
                    },
                    "thumbnail": {
                        "type": "boolean",
                        "description": "Return a PNG thumbnail instead of the full image. Defaults to true for MP4 gallery items and false otherwise."
                    }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "list_models",
            "description": "List mold models visible to the running mold server.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "downloaded_only": {
                        "type": "boolean",
                        "description": "Only show downloaded models. Defaults to true."
                    },
                    "generation_only": {
                        "type": "boolean",
                        "description": "Hide upscalers, utility models, and auxiliary models. Defaults to true."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Maximum number of models to return. Defaults to 50."
                    }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "list_loras",
            "description": "List installed LoRA adapters. When model is provided, returns LoRAs compatible with that model family.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Optional mold model name, such as flux-dev:q8 or realistic-vision-v5:fp16. Filters LoRAs to the model's compatible family."
                    }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "server_status",
            "description": "Show mold server health, queue, loaded model, and GPU status.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }
    ])
}

fn response(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}

fn error_response(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into()
        }
    })
}

fn text_result(text: impl Into<String>) -> Value {
    json!({
        "content": [{ "type": "text", "text": text.into() }]
    })
}

#[cfg(test)]
fn handle_protocol_message_for_test(message: Value) -> Option<Value> {
    handle_protocol_message(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ENV_LOCK;
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;
    use tokio::io::AsyncReadExt;
    use tokio::sync::Notify;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    struct HeldCanonicalAdmission;

    impl Respond for HeldCanonicalAdmission {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            ResponseTemplate::new(202).set_body_json(json!({
                "id": "batch-1",
                "client_batch_id": body["client_batch_id"],
                "instance_id": "instance-1",
                "durable": true,
                "children": [{
                    "index": 1,
                    "job_id": "durable-job-1",
                    "state": "held",
                    "error": "model dependency is unavailable",
                    "retryable": true
                }]
            }))
        }
    }

    fn transport_request() -> GenerateRequest {
        serde_json::from_value(json!({
            "prompt": "transport proof",
            "model": "flux-dev:q4",
            "width": 64,
            "height": 64,
            "steps": 1,
            "guidance": 1.0
        }))
        .unwrap()
    }

    async fn seed_retryable_canonical_job(jobs: &AsyncJobRegistry) -> String {
        let request = transport_request();
        let id = jobs.create(&request).await.unwrap();
        jobs.finish_canonical(
            &id,
            CanonicalAsyncSettlement {
                authority: Some(GenerationBatchAuthority {
                    instance_id: "instance-1".into(),
                    batch_id: "batch-1".into(),
                    client_batch_id: "client-batch-1".into(),
                }),
                durable_job_id: Some("durable-job-1".into()),
                retryable: Some(true),
                status: AsyncJobStatus::Held,
                error: Some("model dependency is unavailable".into()),
                result: None,
                canonical_child: None,
            },
        )
        .await;
        id
    }

    /// Same as [`seed_retryable_canonical_job`] but with a concrete child
    /// revision, which is the fence an interrupted retry reconciles against.
    async fn seed_retryable_canonical_job_at_revision(
        jobs: &AsyncJobRegistry,
        revision: u64,
    ) -> String {
        let request = transport_request();
        let id = jobs.create(&request).await.unwrap();
        let child: GenerationBatchChild = serde_json::from_value(json!({
            "index": 1,
            "job_id": "durable-job-1",
            "state": "held",
            "error": "model dependency is unavailable",
            "retryable": true,
            "revision": revision
        }))
        .unwrap();
        jobs.finish_canonical(
            &id,
            CanonicalAsyncSettlement {
                authority: Some(GenerationBatchAuthority {
                    instance_id: "instance-1".into(),
                    batch_id: "batch-1".into(),
                    client_batch_id: "client-batch-1".into(),
                }),
                durable_job_id: Some("durable-job-1".into()),
                retryable: Some(true),
                status: AsyncJobStatus::Held,
                error: Some("model dependency is unavailable".into()),
                result: None,
                canonical_child: Some(child),
            },
        )
        .await;
        id
    }

    async fn seed_completed_canonical_job(jobs: &AsyncJobRegistry) -> String {
        let request = transport_request();
        let id = jobs.create(&request).await.unwrap();
        let child = serde_json::from_value(json!({
            "index": 1,
            "job_id": "durable-job-1",
            "state": "complete",
            "result": {
                "filename": "result.png",
                "seed": 4242,
                "generation_time_ms": 7500,
                "gpu": 1
            }
        }))
        .unwrap();
        finish_canonical_child(
            jobs,
            &id,
            GenerationBatchAuthority {
                instance_id: "instance-1".into(),
                batch_id: "batch-1".into(),
                client_batch_id: "client-batch-1".into(),
            },
            child,
            None,
        )
        .await;
        id
    }

    fn spawn_hung_then_hydrated_gallery_server() -> (
        String,
        Arc<Notify>,
        Arc<AtomicUsize>,
        tokio::task::JoinHandle<()>,
    ) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let requests_for_server = requests.clone();
        let first_request_seen = Arc::new(Notify::new());
        let first_request_seen_for_server = first_request_seen.clone();
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(1, 1)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let png = png.into_inner();
        let mut gallery = test_gallery_image("result.png", "flux-dev:q4", "transport proof", 1);
        gallery.metadata.job_id = Some("durable-job-1".into());
        let gallery = serde_json::to_vec(&vec![gallery]).unwrap();

        let server = tokio::spawn(async move {
            let mut hung_sockets = Vec::new();
            for request_index in 0..4 {
                let (mut socket, _) = listener.accept().await.unwrap();
                requests_for_server.fetch_add(1, Ordering::SeqCst);
                let mut request = Vec::new();
                let mut buffer = [0_u8; 2048];
                loop {
                    let read = socket.read(&mut buffer).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                if request_index < 2 {
                    if request_index == 0 {
                        first_request_seen_for_server.notify_waiters();
                    }
                    hung_sockets.push(socket);
                    continue;
                }
                let head = String::from_utf8_lossy(&request);
                let (content_type, body) = if head.starts_with("GET /api/gallery/image/result.png")
                {
                    ("image/png", png.as_slice())
                } else {
                    // Deliberate contract change: hydration now narrows the
                    // listing with `?filename=` instead of transferring the
                    // whole gallery index once per artifact.
                    assert!(
                        head.starts_with("GET /api/gallery?filename=result.png "),
                        "unexpected gallery request: {}",
                        head.lines().next().unwrap_or_default()
                    );
                    ("application/json", gallery.as_slice())
                };
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
                socket.write_all(body).await.unwrap();
            }
            drop(hung_sockets);
        });
        (base_url, first_request_seen, requests, server)
    }

    fn spawn_hung_then_running_retry_server() -> (
        String,
        Arc<Notify>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        tokio::task::JoinHandle<()>,
    ) {
        let first_status_seen = Arc::new(Notify::new());
        let retry_posts = Arc::new(AtomicUsize::new(0));
        let status_reads = Arc::new(AtomicUsize::new(0));
        let first_status_seen_for_server = first_status_seen.clone();
        let retry_posts_for_server = retry_posts.clone();
        let status_reads_for_server = status_reads.clone();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        let server = tokio::spawn(async move {
            let mut hung_sockets = Vec::new();
            for _ in 0..3 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 2048];
                loop {
                    let read = socket.read(&mut buffer).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&request);
                if head.starts_with("POST /api/queue/durable-job-1/retry") {
                    retry_posts_for_server.fetch_add(1, Ordering::SeqCst);
                    socket
                        .write_all(
                            b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        )
                        .await
                        .unwrap();
                    continue;
                }
                assert!(head.starts_with("GET /api/generation-batches/batch-1"));
                let status_read = status_reads_for_server.fetch_add(1, Ordering::SeqCst);
                if status_read == 0 {
                    first_status_seen_for_server.notify_waiters();
                    hung_sockets.push(socket);
                    continue;
                }
                let body = json!({
                    "id": "batch-1",
                    "client_batch_id": "client-batch-1",
                    "instance_id": "instance-1",
                    "durable": true,
                    "children": [{
                        "index": 1,
                        "job_id": "durable-job-1",
                        "state": "running"
                    }]
                })
                .to_string();
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
            drop(hung_sockets);
        });
        (
            base_url,
            first_status_seen,
            retry_posts,
            status_reads,
            server,
        )
    }

    async fn mount_canonical_capabilities(server: &MockServer) {
        let mut capabilities = mold_core::ServerCapabilities::default();
        capabilities.queue.heterogeneous_batch_max_outputs = Some(64);
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(ResponseTemplate::new(200).set_body_json(capabilities))
            .expect(1)
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn async_mcp_uses_canonical_admission_and_preserves_held_authority() {
        let server = MockServer::start().await;
        mount_canonical_capabilities(&server).await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches"))
            .respond_with(HeldCanonicalAdmission)
            .mount(&server)
            .await;

        let jobs = AsyncJobRegistry::default();
        let request = transport_request();
        let local_id = jobs.create(&request).await.unwrap();
        run_async_generation(
            MoldClient::new(&server.uri()),
            jobs.clone(),
            local_id.clone(),
            request,
        )
        .await;

        let job = jobs.get(&local_id).await.unwrap();
        assert_eq!(job.status, AsyncJobStatus::Held);
        assert_eq!(
            job.authority
                .as_ref()
                .map(|authority| authority.instance_id.as_str()),
            Some("instance-1")
        );
        assert_eq!(
            job.authority
                .as_ref()
                .map(|authority| authority.batch_id.as_str()),
            Some("batch-1")
        );
        assert_eq!(job.durable_job_id.as_deref(), Some("durable-job-1"));
        assert_eq!(job.retryable, Some(true));
        assert!(job.error.as_deref().unwrap().contains("dependency"));
        let requests = server.received_requests().await.unwrap();
        assert!(requests
            .iter()
            .any(|request| request.url.path() == "/api/generation-batches"));
        assert!(!requests.iter().any(|request| {
            matches!(request.url.path(), "/api/generate" | "/api/generate/stream")
        }));
    }

    #[tokio::test]
    async fn async_mcp_retries_only_the_exact_held_canonical_child() {
        let server = MockServer::start().await;
        mount_canonical_capabilities(&server).await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches"))
            .respond_with(HeldCanonicalAdmission)
            .mount(&server)
            .await;

        let mcp = McpServer {
            client: MoldClient::new(&server.uri()),
            jobs: AsyncJobRegistry::default(),
        };
        let request = transport_request();
        let local_id = mcp.jobs.create(&request).await.unwrap();
        run_async_generation(
            mcp.client.clone(),
            mcp.jobs.clone(),
            local_id.clone(),
            request,
        )
        .await;
        let client_batch_id = mcp
            .jobs
            .get(&local_id)
            .await
            .unwrap()
            .authority
            .unwrap()
            .client_batch_id;
        let held = json!({
            "id": "batch-1",
            "client_batch_id": client_batch_id,
            "instance_id": "instance-1",
            "durable": true,
            "children": [{
                "index": 1,
                "job_id": "durable-job-1",
                "state": "held",
                "error": "model dependency is unavailable",
                "retryable": true
            }]
        });
        Mock::given(method("GET"))
            .and(path("/api/generation-batches/batch-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(held))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/queue/durable-job-1/retry"))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&server)
            .await;

        let result = mcp
            .tool_generation_retry(json!({ "job_id": local_id }))
            .await
            .unwrap();
        assert!(matches!(
            result["structuredContent"]["status"].as_str(),
            Some("queued" | "held")
        ));

        tokio::task::yield_now().await;
        let requests = server.received_requests().await.unwrap();
        assert!(requests
            .iter()
            .any(|request| request.url.path() == "/api/queue/durable-job-1/retry"));
        let retry_request = requests
            .iter()
            .find(|request| request.url.path() == "/api/queue/durable-job-1/retry")
            .unwrap();
        let retry_body: Value = serde_json::from_slice(&retry_request.body).unwrap();
        assert_eq!(retry_body["instance_id"], "instance-1");
        assert_eq!(retry_body["batch_id"], "batch-1");
        assert_eq!(retry_body["job_id"], "durable-job-1");
        assert!(!requests
            .iter()
            .any(|request| request.url.path() == "/api/generate"));
    }

    struct RetryRecoveryAfterBoundedFailures {
        calls: Arc<AtomicUsize>,
        client_batch_id: String,
    }

    impl Respond for RetryRecoveryAfterBoundedFailures {
        fn respond(&self, _request: &Request) -> ResponseTemplate {
            let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
            if attempt < 5 {
                return ResponseTemplate::new(503);
            }
            if attempt == 5 {
                return ResponseTemplate::new(404);
            }
            ResponseTemplate::new(200).set_body_json(json!({
                "id": "batch-1",
                "client_batch_id": self.client_batch_id,
                "instance_id": "instance-1",
                "durable": true,
                "children": [{
                    "index": 1,
                    "job_id": "durable-job-1",
                    "state": "held",
                    "error": "dependency remains unavailable",
                    "retryable": true
                }]
            }))
        }
    }

    #[tokio::test]
    async fn async_mcp_lazily_reconciles_exact_retry_after_host_recovers() {
        let server = MockServer::start().await;
        mount_canonical_capabilities(&server).await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches"))
            .respond_with(HeldCanonicalAdmission)
            .mount(&server)
            .await;

        let mcp = McpServer {
            client: MoldClient::new(&server.uri()),
            jobs: AsyncJobRegistry::default(),
        };
        let request = transport_request();
        let local_id = mcp.jobs.create(&request).await.unwrap();
        run_async_generation(
            mcp.client.clone(),
            mcp.jobs.clone(),
            local_id.clone(),
            request,
        )
        .await;
        let client_batch_id = mcp
            .jobs
            .get(&local_id)
            .await
            .unwrap()
            .authority
            .unwrap()
            .client_batch_id;
        Mock::given(method("POST"))
            .and(path("/api/queue/durable-job-1/retry"))
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&server)
            .await;
        let status_calls = Arc::new(AtomicUsize::new(0));
        Mock::given(method("GET"))
            .and(path("/api/generation-batches/batch-1"))
            .respond_with(RetryRecoveryAfterBoundedFailures {
                calls: status_calls.clone(),
                client_batch_id,
            })
            .expect(7)
            .mount(&server)
            .await;

        let immediate = mcp
            .tool_generation_retry(json!({ "job_id": local_id }))
            .await
            .unwrap();
        assert_eq!(immediate["structuredContent"]["status"], "queued");

        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let result = mcp
                    .tool_generation_status(json!({
                        "job_id": local_id,
                        "include_result": false
                    }))
                    .await
                    .unwrap();
                if result["structuredContent"]["status"] == "held" {
                    assert_eq!(
                        result["structuredContent"]["error"],
                        "dependency remains unavailable"
                    );
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("exact retry reconciliation did not recover after the host returned");

        assert_eq!(status_calls.load(Ordering::SeqCst), 7);
        let requests = server.received_requests().await.unwrap();
        assert_eq!(
            requests
                .iter()
                .filter(|request| request.url.path() == "/api/queue/durable-job-1/retry")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn ambiguous_retry_ignores_stale_held_until_delayed_active_state() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let retry_posts = Arc::new(AtomicUsize::new(0));
        let retry_posts_for_server = retry_posts.clone();
        let status_reads = Arc::new(AtomicUsize::new(0));
        let status_reads_for_server = status_reads.clone();
        let server = tokio::spawn(async move {
            for _ in 0..4 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 2048];
                loop {
                    let read = socket.read(&mut buffer).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&request);
                if head.starts_with("POST /api/queue/durable-job-1/retry") {
                    retry_posts_for_server.fetch_add(1, Ordering::SeqCst);
                    socket.shutdown().await.unwrap();
                    continue;
                }
                assert!(head.starts_with("GET /api/generation-batches/batch-1"));
                let read = status_reads_for_server.fetch_add(1, Ordering::SeqCst);
                let child = match read {
                    0 => json!({
                        "index": 1,
                        "job_id": "durable-job-1",
                        "state": "held",
                        "error": "old dependency failure",
                        "retryable": true
                    }),
                    1 => json!({
                        "index": 1,
                        "job_id": "durable-job-1",
                        "state": "accepted"
                    }),
                    _ => json!({
                        "index": 1,
                        "job_id": "durable-job-1",
                        "state": "running"
                    }),
                };
                let body = json!({
                    "id": "batch-1",
                    "client_batch_id": "client-batch-1",
                    "instance_id": "instance-1",
                    "durable": true,
                    "children": [child]
                })
                .to_string();
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        });
        let mcp = McpServer {
            client: MoldClient::new(&base_url),
            jobs: AsyncJobRegistry::default(),
        };
        let local_id = seed_retryable_canonical_job(&mcp.jobs).await;

        let retry = mcp
            .tool_generation_retry(json!({ "job_id": local_id }))
            .await
            .unwrap();
        assert_eq!(retry["structuredContent"]["status"], "queued");
        let stale = mcp
            .tool_generation_status(json!({ "job_id": local_id, "include_result": false }))
            .await
            .unwrap();
        assert_eq!(stale["structuredContent"]["status"], "queued");
        assert_eq!(stale["structuredContent"]["retryable"], false);
        assert!(stale["structuredContent"]["error"]
            .as_str()
            .unwrap()
            .contains("pre-commit held state"));

        tokio::time::sleep(reconciliation_initial_backoff()).await;
        let accepted = mcp
            .tool_generation_status(json!({ "job_id": local_id, "include_result": false }))
            .await
            .unwrap();
        assert_eq!(accepted["structuredContent"]["status"], "queued");
        assert_ne!(accepted["structuredContent"]["retryable"], true);

        tokio::time::sleep(reconciliation_initial_backoff()).await;
        let running = mcp
            .tool_generation_status(json!({ "job_id": local_id, "include_result": false }))
            .await
            .unwrap();
        assert_eq!(running["structuredContent"]["status"], "running");
        assert_eq!(retry_posts.load(Ordering::SeqCst), 1);
        assert_eq!(status_reads.load(Ordering::SeqCst), 3);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn an_advanced_revision_settles_a_re_held_retry_without_five_polls() {
        // The retry landed and the job was held again for a NEW reason.
        // State alone cannot tell that from a retry that never arrived, so
        // the old code spent all five confirmation attempts staring at
        // `held` and then settled either case identically.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let status_reads = Arc::new(AtomicUsize::new(0));
        let status_reads_for_server = status_reads.clone();
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 2048];
                loop {
                    let read = socket.read(&mut buffer).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&request);
                if head.starts_with("POST /api/queue/durable-job-1/retry") {
                    // Lost response: the retry is ambiguous to the client.
                    socket.shutdown().await.unwrap();
                    continue;
                }
                assert!(head.starts_with("GET /api/generation-batches/batch-1"));
                status_reads_for_server.fetch_add(1, Ordering::SeqCst);
                let body = json!({
                    "id": "batch-1",
                    "client_batch_id": "client-batch-1",
                    "instance_id": "instance-1",
                    "durable": true,
                    "children": [{
                        "index": 1,
                        "job_id": "durable-job-1",
                        "state": "held",
                        "error": "a different dependency failed",
                        "retryable": true,
                        // Higher than the seeded revision: the retry landed.
                        "revision": 8
                    }]
                })
                .to_string();
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        });
        let mcp = McpServer {
            client: MoldClient::new(&base_url),
            jobs: AsyncJobRegistry::default(),
        };
        let local_id = seed_retryable_canonical_job_at_revision(&mcp.jobs, 7).await;

        mcp.tool_generation_retry(json!({ "job_id": local_id }))
            .await
            .unwrap();
        let settled = mcp
            .tool_generation_status(json!({ "job_id": local_id, "include_result": false }))
            .await
            .unwrap();

        assert_eq!(settled["structuredContent"]["status"], "held");
        assert_eq!(settled["structuredContent"]["retryable"], true);
        assert!(
            settled["structuredContent"]["error"]
                .as_str()
                .unwrap()
                .contains("a different dependency failed"),
            "the NEW hold reason must surface: {settled}"
        );
        assert_eq!(
            status_reads.load(Ordering::SeqCst),
            1,
            "an advanced revision settles on the first read"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn an_unadvanced_revision_names_the_retry_that_never_landed() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            for _ in 0..(1 + AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS) {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 2048];
                loop {
                    let read = socket.read(&mut buffer).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&request);
                if head.starts_with("POST /api/queue/durable-job-1/retry") {
                    socket.shutdown().await.unwrap();
                    continue;
                }
                let body = json!({
                    "id": "batch-1",
                    "client_batch_id": "client-batch-1",
                    "instance_id": "instance-1",
                    "durable": true,
                    "children": [{
                        "index": 1,
                        "job_id": "durable-job-1",
                        "state": "held",
                        "error": "model dependency is unavailable",
                        "retryable": true,
                        // Unchanged: the retry never reached the server.
                        "revision": 7
                    }]
                })
                .to_string();
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        });
        let mcp = McpServer {
            client: MoldClient::new(&base_url),
            jobs: AsyncJobRegistry::default(),
        };
        let local_id = seed_retryable_canonical_job_at_revision(&mcp.jobs, 7).await;

        mcp.tool_generation_retry(json!({ "job_id": local_id }))
            .await
            .unwrap();
        let mut last = serde_json::Value::Null;
        for _ in 0..AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS {
            last = mcp
                .tool_generation_status(json!({ "job_id": local_id, "include_result": false }))
                .await
                .unwrap();
            tokio::time::sleep(reconciliation_initial_backoff()).await;
        }

        assert_eq!(last["structuredContent"]["retryable"], true);
        let error = last["structuredContent"]["error"].as_str().unwrap_or("");
        assert!(
            error.contains("the retry did not reach durable job"),
            "an unlanded retry must be named, not settled silently: {last}"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn list_status_poll_recovers_ambiguous_retry_without_a_job_filter() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/queue/durable-job-1/retry"))
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/generation-batches/batch-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "batch-1",
                "client_batch_id": "client-batch-1",
                "instance_id": "instance-1",
                "durable": true,
                "children": [{
                    "index": 1,
                    "job_id": "durable-job-1",
                    "state": "held",
                    "error": "dependency remains unavailable",
                    "retryable": true
                }]
            })))
            .expect(AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS as u64)
            .mount(&server)
            .await;
        let mcp = McpServer {
            client: MoldClient::new(&server.uri()),
            jobs: AsyncJobRegistry::default(),
        };
        let local_id = seed_retryable_canonical_job(&mcp.jobs).await;
        mcp.tool_generation_retry(json!({ "job_id": local_id }))
            .await
            .unwrap();

        for attempt in 0..AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS {
            if attempt > 0 {
                tokio::time::sleep(reconciliation_initial_backoff()).await;
            }
            let result = mcp.tool_generation_status(json!({})).await.unwrap();
            let job = &result["structuredContent"]["jobs"][0];
            if attempt + 1 < AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS {
                assert_eq!(job["status"], "queued");
                assert_eq!(job["retryable"], false);
            } else {
                assert_eq!(job["status"], "held");
                assert_eq!(job["retryable"], true);
            }
        }
    }

    #[tokio::test]
    async fn reconciliation_read_timeout_releases_key_for_recovery() {
        let (base_url, _first_seen, retry_posts, status_reads, server) =
            spawn_hung_then_running_retry_server();
        let mcp = McpServer {
            client: MoldClient::new(&base_url),
            jobs: AsyncJobRegistry::default(),
        };
        let local_id = seed_retryable_canonical_job(&mcp.jobs).await;
        mcp.tool_generation_retry(json!({ "job_id": local_id }))
            .await
            .unwrap();

        let started = Instant::now();
        let timed_out = mcp
            .tool_generation_status(json!({ "job_id": local_id, "include_result": false }))
            .await
            .unwrap();
        assert!(started.elapsed() < reconciliation_read_timeout() * 2);
        assert_eq!(timed_out["structuredContent"]["status"], "queued");
        assert!(timed_out["structuredContent"]["error"]
            .as_str()
            .unwrap()
            .contains("status read timed out"));

        tokio::time::sleep(reconciliation_initial_backoff()).await;
        let recovered = mcp
            .tool_generation_status(json!({ "job_id": local_id, "include_result": false }))
            .await
            .unwrap();
        assert_eq!(recovered["structuredContent"]["status"], "running");
        assert_eq!(retry_posts.load(Ordering::SeqCst), 1);
        assert_eq!(status_reads.load(Ordering::SeqCst), 2);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn dropped_reconciliation_future_releases_key_for_immediate_recovery() {
        let (base_url, first_seen, retry_posts, status_reads, server) =
            spawn_hung_then_running_retry_server();
        let mcp = McpServer {
            client: MoldClient::new(&base_url),
            jobs: AsyncJobRegistry::default(),
        };
        let local_id = seed_retryable_canonical_job(&mcp.jobs).await;
        mcp.tool_generation_retry(json!({ "job_id": local_id }))
            .await
            .unwrap();

        {
            let pending =
                mcp.tool_generation_status(json!({ "job_id": local_id, "include_result": false }));
            tokio::pin!(pending);
            tokio::select! {
                result = &mut pending => panic!("hung status read completed unexpectedly: {result:?}"),
                () = first_seen.notified() => {}
            }
        }

        let recovered = mcp
            .tool_generation_status(json!({ "job_id": local_id, "include_result": false }))
            .await
            .unwrap();
        assert_eq!(recovered["structuredContent"]["status"], "running");
        assert_eq!(retry_posts.load(Ordering::SeqCst), 1);
        assert_eq!(status_reads.load(Ordering::SeqCst), 2);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn completed_status_skips_gallery_until_bounded_explicit_hydration_and_recovers() {
        let (base_url, first_request_seen, gallery_requests, server) =
            spawn_hung_then_hydrated_gallery_server();
        let mcp = McpServer {
            client: MoldClient::new(&base_url),
            jobs: AsyncJobRegistry::default(),
        };
        let local_id = seed_completed_canonical_job(&mcp.jobs).await;

        let listed = mcp.tool_generation_status(json!({})).await.unwrap();
        assert_eq!(
            listed["structuredContent"]["jobs"][0]["status"],
            "succeeded"
        );
        let status_only = mcp
            .tool_generation_status(json!({
                "job_id": local_id,
                "include_result": false
            }))
            .await
            .unwrap();
        assert_eq!(status_only["structuredContent"]["status"], "succeeded");
        assert_eq!(status_only["structuredContent"]["result"], Value::Null);
        assert_eq!(gallery_requests.load(Ordering::SeqCst), 0);

        {
            let pending = mcp.tool_generation_status(json!({
                "job_id": local_id,
                "include_result": true
            }));
            tokio::pin!(pending);
            tokio::select! {
                result = &mut pending => panic!("hung result hydration completed unexpectedly: {result:?}"),
                () = first_request_seen.notified() => {}
            }
        }

        let started = Instant::now();
        let timed_out = mcp
            .tool_generation_status(json!({
                "job_id": local_id,
                "include_result": true
            }))
            .await
            .unwrap();
        assert!(started.elapsed() < reconciliation_read_timeout() * 2);
        assert_eq!(timed_out["structuredContent"]["status"], "succeeded");
        assert_eq!(timed_out["structuredContent"]["result"], Value::Null);
        assert!(timed_out["structuredContent"]["result_error"]
            .as_str()
            .unwrap()
            .contains("timed out"));

        let hydrated = mcp
            .tool_generation_status(json!({
                "job_id": local_id,
                "include_result": true
            }))
            .await
            .unwrap();
        assert_eq!(hydrated["structuredContent"]["status"], "succeeded");
        assert_eq!(hydrated["structuredContent"]["result"]["image"]["width"], 1);
        // The child settled with these; reporting 0 and null told the caller
        // the render was instantaneous and ran nowhere.
        assert_eq!(
            hydrated["structuredContent"]["result"]["generation_time_ms"],
            7500
        );
        assert_eq!(hydrated["structuredContent"]["result"]["gpu"], 1);
        assert_eq!(hydrated["structuredContent"]["result_error"], Value::Null);
        assert_eq!(gallery_requests.load(Ordering::SeqCst), 4);
        assert_eq!(hydrated["content"][1]["type"], "image");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn concurrent_status_reads_coalesce_exact_retry_reconciliation() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/queue/durable-job-1/retry"))
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/generation-batches/batch-1"))
            .respond_with(ResponseTemplate::new(503).set_delay(Duration::from_millis(75)))
            .expect(1)
            .mount(&server)
            .await;
        let mcp = McpServer {
            client: MoldClient::new(&server.uri()),
            jobs: AsyncJobRegistry::default(),
        };
        let local_id = seed_retryable_canonical_job(&mcp.jobs).await;
        mcp.tool_generation_retry(json!({ "job_id": local_id }))
            .await
            .unwrap();

        let arguments = json!({ "job_id": local_id, "include_result": false });
        let (first, second) = tokio::join!(
            mcp.tool_generation_status(arguments.clone()),
            mcp.tool_generation_status(arguments)
        );
        assert_eq!(first.unwrap()["structuredContent"]["status"], "queued");
        assert_eq!(second.unwrap()["structuredContent"]["status"], "queued");
        assert_eq!(
            server
                .received_requests()
                .await
                .unwrap()
                .iter()
                .filter(|request| request.url.path() == "/api/generation-batches/batch-1")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn retry_shutdown_leaves_no_detached_reconciliation() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/queue/durable-job-1/retry"))
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&server)
            .await;
        let mcp = McpServer {
            client: MoldClient::new(&server.uri()),
            jobs: AsyncJobRegistry::default(),
        };
        let local_id = seed_retryable_canonical_job(&mcp.jobs).await;
        mcp.tool_generation_retry(json!({ "job_id": local_id }))
            .await
            .unwrap();
        drop(mcp);
        tokio::time::sleep(Duration::from_millis(100)).await;

        let requests = server.received_requests().await.unwrap();
        assert_eq!(
            requests
                .iter()
                .filter(|request| request.url.path().starts_with("/api/generation-batches/"))
                .count(),
            0
        );
        assert_eq!(
            requests
                .iter()
                .filter(|request| request.url.path() == "/api/queue/durable-job-1/retry")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn unresolved_job_registry_is_hard_capped_without_pollers() {
        let jobs = AsyncJobRegistry::default();
        let request = transport_request();
        for _ in 0..MAX_ASYNC_JOBS {
            jobs.create(&request).await.unwrap();
        }

        let error = jobs.create(&request).await.unwrap_err();
        assert!(error.contains("registry is full"));
        assert_eq!(jobs.inner.jobs.lock().await.len(), MAX_ASYNC_JOBS);
        assert!(jobs.inner.reconciliations.lock().await.is_empty());
    }

    #[tokio::test]
    async fn list_status_bounds_eligible_reconciliation_work() {
        let jobs = AsyncJobRegistry::default();
        let request = transport_request();
        for index in 0..(MAX_LIST_STATUS_RECONCILIATIONS + 1) {
            let id = jobs.create(&request).await.unwrap();
            jobs.finish_canonical(
                &id,
                CanonicalAsyncSettlement {
                    authority: Some(GenerationBatchAuthority {
                        instance_id: "instance-1".into(),
                        batch_id: format!("batch-{index}"),
                        client_batch_id: format!("client-batch-{index}"),
                    }),
                    durable_job_id: Some(format!("durable-job-{index}")),
                    retryable: Some(true),
                    status: AsyncJobStatus::Held,
                    error: Some("dependency unavailable".into()),
                    result: None,
                    canonical_child: None,
                },
            )
            .await;
            jobs.mark_retry_queued(&id, false, 0).await;
        }

        assert_eq!(
            jobs.eligible_pending_reconciliations(MAX_LIST_STATUS_RECONCILIATIONS)
                .await
                .len(),
            MAX_LIST_STATUS_RECONCILIATIONS
        );
    }

    #[test]
    fn async_status_summarizes_bounded_stage_progress() {
        assert_eq!(
            progress_summary(&SseProgressEvent::StageProgress {
                name: "Evaluating transformer (conditional)".to_string(),
                current: 9,
                total: 48,
            }),
            "Evaluating transformer (conditional) 9/48"
        );
    }

    #[test]
    fn initialize_declares_tools_capability() {
        let response = handle_protocol_message_for_test(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": { "protocolVersion": "2025-06-18" }
        }))
        .expect("initialize should produce a response");

        assert_eq!(response["id"], 1);
        assert_eq!(response["result"]["capabilities"]["tools"], json!({}));
        assert_eq!(response["result"]["serverInfo"]["name"], "mold");
    }

    #[test]
    fn tools_list_exposes_generate_image() {
        let response = handle_protocol_message_for_test(json!({
            "jsonrpc": "2.0",
            "id": "tools",
            "method": "tools/list"
        }))
        .expect("tools/list should produce a response");

        let tools = response["result"]["tools"].as_array().unwrap();
        assert!(tools.iter().any(|tool| tool["name"] == "generate_image"));
    }

    #[test]
    fn tools_list_exposes_async_generation_tools() {
        let response = handle_protocol_message_for_test(json!({
            "jsonrpc": "2.0",
            "id": "tools",
            "method": "tools/list"
        }))
        .expect("tools/list should produce a response");

        let tools = response["result"]["tools"].as_array().unwrap();
        assert!(tools
            .iter()
            .any(|tool| tool["name"] == "generate_image_async"));
        assert!(tools.iter().any(|tool| tool["name"] == "generation_status"));
        assert!(tools.iter().any(|tool| tool["name"] == "generation_retry"));
    }

    #[test]
    fn tools_list_exposes_gallery_tools() {
        let response = handle_protocol_message_for_test(json!({
            "jsonrpc": "2.0",
            "id": "tools",
            "method": "tools/list"
        }))
        .expect("tools/list should produce a response");

        let tools = response["result"]["tools"].as_array().unwrap();
        assert!(tools.iter().any(|tool| tool["name"] == "list_gallery"));
        assert!(tools.iter().any(|tool| tool["name"] == "get_gallery_image"));
    }

    #[tokio::test]
    async fn server_status_queue_projection_sanitizes_legacy_host_utility_identity() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let queue = mold_core::QueueListingWire {
            entries: vec![],
            live_only_entries: vec![],
            plan: Some(mold_core::QueuePlan {
                work_items: vec![
                    mold_core::QueueWorkItem {
                        work_id: "legacy".into(),
                        activity_phase: mold_core::QueueActivityPhase::Queued,
                        planned_device_id: Some("cpu:utility:0".into()),
                        lane_order: Some(0),
                        ..Default::default()
                    },
                    mold_core::QueueWorkItem {
                        work_id: "typed-host".into(),
                        planned_lane_kind: Some(mold_core::QueuePlannedLaneKind::HostUtility),
                        planned_device_id: Some("typed-host-internal".into()),
                        lane_order: Some(1),
                        ..Default::default()
                    },
                    mold_core::QueueWorkItem {
                        work_id: "gpu".into(),
                        planned_lane_kind: Some(mold_core::QueuePlannedLaneKind::Device),
                        planned_device_id: Some("cuda:stable-a".into()),
                        lane_order: Some(2),
                        ..Default::default()
                    },
                    mold_core::QueueWorkItem {
                        work_id: "future".into(),
                        planned_lane_kind: Some(mold_core::QueuePlannedLaneKind::Unknown(
                            "remote_utility".into(),
                        )),
                        planned_device_id: Some("future-public-id".into()),
                        lane_order: Some(3),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            page: None,
        };
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/status"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(mold_core::ServerStatus {
                    version: "test".into(),
                    git_sha: None,
                    build_date: None,
                    models_loaded: vec![],
                    busy: true,
                    current_generation: None,
                    gpu_info: None,
                    uptime_secs: 1,
                    hostname: None,
                    memory_status: None,
                    gpus: None,
                    queue_depth: Some(4),
                    queue_capacity: Some(8),
                    queue_paused: Some(false),
                    instance_id: None,
                    models_disk: None,
                    host_memory: None,
                    durable_media: None,
                }),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/devices"))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/queue"))
            .respond_with(ResponseTemplate::new(200).set_body_json(queue))
            .expect(1)
            .mount(&server)
            .await;

        let result = McpServer::new(Some(server.uri()))
            .tool_server_status()
            .await
            .unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("queue load (server-reported): 4 jobs"));
        assert!(text.contains("queue response: 0 rows"));
        assert!(text.contains("runtime window capacity: 8 jobs"));
        assert!(!text.contains("4/8"));
        assert!(text.contains("work legacy: phase=queued lane=host-utility/0"));
        assert!(text.contains("work typed-host: phase=queued lane=host-utility/1"));
        assert!(text.contains("work gpu: phase=queued lane=cuda:stable-a/2"));
        assert!(text.contains("work future: phase=queued lane=assigned/3"));
        assert!(!text.contains("cpu:utility:0"));
        assert!(!text.contains("typed-host-internal"));
        assert!(!text.contains("future-public-id"));

        let work = result["structuredContent"]["queue"]["plan"]["work_items"]
            .as_array()
            .unwrap();
        assert_eq!(work[0]["planned_lane_kind"], "host_utility");
        assert_eq!(work[0]["planned_device_id"], Value::Null);
        assert_eq!(work[1]["planned_lane_kind"], "host_utility");
        assert_eq!(work[1]["planned_device_id"], Value::Null);
        assert_eq!(work[2]["planned_lane_kind"], "device");
        assert_eq!(work[2]["planned_device_id"], "cuda:stable-a");
        assert_eq!(work[3]["planned_lane_kind"], "remote_utility");
        assert_eq!(work[3]["planned_device_id"], "future-public-id");
    }

    #[test]
    fn paged_queue_summary_distinguishes_backlog_page_and_runtime_window() {
        let status: mold_core::ServerStatus = serde_json::from_value(json!({
            "version": "test",
            "models_loaded": [],
            "busy": true,
            "current_generation": null,
            "gpu_info": null,
            "uptime_secs": 1,
            "queue_depth": 144,
            "queue_capacity": 12
        }))
        .unwrap();
        let row = mold_core::QueueJobEntryWire {
            id: "job".into(),
            model: "flux-dev:q8".into(),
            state: "queued".into(),
            started_at_unix_ms: 0,
            position: 0,
            gpu: None,
            target_gpu: None,
            seed_pinned: None,
            metadata: None,
            durable: Some(true),
            held_reason: None,
            ..Default::default()
        };
        let queue = mold_core::QueueListingWire {
            entries: vec![row; 5],
            page: Some(mold_core::QueuePage {
                limit: 5,
                offset: 0,
                returned: 5,
                next_cursor: Some("opaque".into()),
            }),
            ..Default::default()
        };
        let mut lines = Vec::new();
        append_queue_summary_status_lines(&mut lines, &status, Some(&queue));
        assert_eq!(
            lines,
            vec![
                "queue backlog: 144 total nonterminal jobs",
                "hydrated queue page: 5 rows",
                "runtime window capacity: 12 jobs",
            ]
        );
    }

    #[test]
    fn tools_list_exposes_lora_stack_and_listing() {
        let response = handle_protocol_message_for_test(json!({
            "jsonrpc": "2.0",
            "id": "tools",
            "method": "tools/list"
        }))
        .expect("tools/list should produce a response");

        let tools = response["result"]["tools"].as_array().unwrap();
        let generate = tools
            .iter()
            .find(|tool| tool["name"] == "generate_image")
            .expect("generate_image tool");
        assert!(generate["inputSchema"]["properties"]["loras"].is_object());
        assert!(
            generate["inputSchema"]["properties"]["loras"]["items"]["oneOf"]
                .as_array()
                .is_some_and(|variants| variants.len() == 2)
        );
        assert!(tools.iter().any(|tool| tool["name"] == "list_loras"));
    }

    #[test]
    fn normalise_lora_arg_accepts_ids_paths_and_default_scale() {
        let parsed: GenerateImageArgs = serde_json::from_value(json!({
            "prompt": "a cat",
            "loras": [
                "cv:827325",
                { "path": "/models/a.safetensors" },
                { "id": "cv:123", "scale": 0.65 }
            ]
        }))
        .unwrap();
        assert_eq!(parsed.loras.as_ref().unwrap().len(), 3);
        assert_eq!(
            normalise_mcp_lora_arg(McpLoraArg::Ref("cv:827325".into())).unwrap(),
            McpLoraRef {
                value: "cv:827325".into(),
                scale: 1.0,
            }
        );
        assert_eq!(
            normalise_mcp_lora_arg(McpLoraArg::Ref("/models/a.safetensors".into())).unwrap(),
            McpLoraRef {
                value: "/models/a.safetensors".into(),
                scale: 1.0,
            }
        );
        assert_eq!(
            normalise_mcp_lora_arg(McpLoraArg::Object(McpLoraObject {
                id: Some(" cv:123 ".into()),
                path: None,
                scale: Some(0.65),
            }))
            .unwrap(),
            McpLoraRef {
                value: "cv:123".into(),
                scale: 0.65,
            }
        );
        assert!(normalise_mcp_lora_arg(McpLoraArg::Object(McpLoraObject {
            id: Some("cv:1".into()),
            path: Some("/models/a.safetensors".into()),
            scale: None,
        }))
        .is_err());
    }

    #[test]
    fn resolve_lora_refs_maps_catalog_ids_and_keeps_paths() {
        let catalog = vec![LoraInfo {
            id: "cv:827325".into(),
            name: "Detail".into(),
            family: "flux".into(),
            author: None,
            path: "/models/cv-827325/detail.safetensors".into(),
            trained_words: vec![],
            size_bytes: Some(123),
            thumbnail_url: None,
            added_at: 10,
        }];
        let loras = resolve_mcp_lora_refs(
            &[
                McpLoraRef {
                    value: "cv:827325".into(),
                    scale: 0.7,
                },
                McpLoraRef {
                    value: "/models/manual.safetensors".into(),
                    scale: 1.0,
                },
            ],
            &catalog,
        )
        .unwrap();

        assert_eq!(loras[0].path, "/models/cv-827325/detail.safetensors");
        assert_eq!(loras[0].scale, 0.7);
        assert_eq!(loras[1].path, "/models/manual.safetensors");
        assert!(resolve_mcp_lora_refs(
            &[McpLoraRef {
                value: "cv:missing".into(),
                scale: 1.0,
            }],
            &catalog,
        )
        .is_err());
    }

    #[test]
    fn build_generate_request_preserves_lora_stack() {
        let req = build_generate_request(
            GenerateImageArgs {
                prompt: "a cat".into(),
                model: Some("flux-dev:q8".into()),
                width: Some(1024),
                height: Some(1024),
                steps: Some(4),
                guidance: Some(1.0),
                seed: Some(42),
                negative_prompt: None,
                output_format: Some("png".into()),
                expand: None,
                loras: None,
            },
            Some(vec![
                LoraWeight {
                    path: "/models/a.safetensors".into(),
                    scale: 0.8,

                    expert: None,
                },
                LoraWeight {
                    path: "/models/b.safetensors".into(),
                    scale: 0.4,

                    expert: None,
                },
            ]),
        )
        .expect("valid generate request");

        let loras = req.loras.expect("lora stack");
        assert_eq!(loras.len(), 2);
        assert_eq!(loras[0].path, "/models/a.safetensors");
        assert_eq!(loras[1].scale, 0.4);
    }

    #[test]
    fn build_generate_request_uses_installed_catalog_sidecar_defaults() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior_home = std::env::var_os("MOLD_HOME");
        let prior_models_dir = std::env::var_os("MOLD_MODELS_DIR");
        let dir = tempfile::tempdir().expect("tempdir");
        unsafe {
            std::env::set_var("MOLD_HOME", dir.path());
            std::env::remove_var("MOLD_MODELS_DIR");
        }

        let install_dir = dir.path().join("models/cv-12345");
        let primary_rel = "sdxl/civitai/12345/model.safetensors";
        let primary = install_dir.join(primary_rel);
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        std::fs::write(&primary, b"fake").unwrap();
        let sidecar = mold_catalog::sidecar::CatalogSidecar {
            schema: mold_catalog::sidecar::SIDECAR_SCHEMA,
            id: "cv:12345".to_string(),
            source: "civitai".to_string(),
            source_id: "12345".to_string(),
            name: "SDXL Fine Tune".to_string(),
            author: Some("tester".to_string()),
            family: "sdxl".to_string(),
            family_role: "finetune".to_string(),
            sub_family: None,
            kind: "checkpoint".to_string(),
            modality: "image".to_string(),
            nsfw: None,
            description: None,
            tags: Vec::new(),
            license: None,
            page_url: None,
            thumbnail_url: None,
            size_bytes: Some(4),
            supported: true,
            trained_words: Vec::new(),
            primary_filename_rel: primary_rel.to_string(),
            primary_size_bytes: None,
            low_noise_filename_rel: None,
            low_noise_size_bytes: None,
            written_at: 0,
        };
        mold_catalog::sidecar::write_sidecar(
            &install_dir.join(mold_catalog::sidecar::SIDECAR_FILENAME),
            &sidecar,
        )
        .unwrap();

        let req = build_generate_request(
            GenerateImageArgs {
                prompt: "a catalog default test".into(),
                model: Some("cv:12345".into()),
                width: None,
                height: None,
                steps: None,
                guidance: None,
                seed: None,
                negative_prompt: None,
                output_format: None,
                expand: None,
                loras: None,
            },
            None,
        )
        .expect("catalog sidecar should supply model defaults");

        assert_eq!(req.width, 1024);
        assert_eq!(req.height, 1024);
        assert_eq!(req.steps, 25);
        assert_eq!(req.guidance, 7.5);

        unsafe {
            match prior_home {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
            match prior_models_dir {
                Some(v) => std::env::set_var("MOLD_MODELS_DIR", v),
                None => std::env::remove_var("MOLD_MODELS_DIR"),
            }
        }
    }

    #[tokio::test]
    async fn async_job_registry_tracks_completed_image() {
        let jobs = AsyncJobRegistry::default();
        let req = GenerateRequest {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            prompt: "a cat".into(),
            negative_prompt: None,
            model: "flux2-klein:q8".into(),
            width: 16,
            height: 16,
            steps: 1,
            guidance: 0.0,
            seed: Some(7),
            batch_size: 1,
            output_format: Some(OutputFormat::Png),
            embed_metadata: Some(false),
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        };

        let id = jobs.create(&req).await.unwrap();
        jobs.finish(
            &id,
            Ok(GenerateResponse {
                request_warnings: Vec::new(),
                audio: None,
                images: vec![mold_core::ImageData {
                    data: vec![1, 2, 3],
                    format: OutputFormat::Png,
                    width: 16,
                    height: 16,
                    index: 0,
                }],
                video: None,
                generation_time_ms: 42,
                model: "flux2-klein:q8".into(),
                seed_used: 7,
                gpu: Some(0),
            }),
        )
        .await;

        let job = jobs.get(&id).await.expect("job should exist");
        assert_eq!(job.status, AsyncJobStatus::Succeeded);
        assert_eq!(job.result.unwrap().images[0].data, vec![1, 2, 3]);
    }

    #[test]
    fn gallery_filter_searches_prompt_and_sorts_by_timestamp_desc() {
        let images = vec![
            test_gallery_image("older.png", "flux-dev:q8", "a red chair", 10),
            test_gallery_image("newer.png", "sdxl:fp16", "a blue chair", 20),
            test_gallery_image("other.png", "flux-dev:q8", "a yellow lamp", 30),
        ];
        let filter = GalleryFilter {
            query: Some("chair".into()),
            sort_by: Some("timestamp".into()),
            sort_order: Some("desc".into()),
            ..Default::default()
        };

        let filtered = filter_sort_gallery(images, &filter).unwrap();
        let names = filtered
            .iter()
            .map(|image| image.filename.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["newer.png", "older.png"]);
    }

    fn test_gallery_image(
        filename: &str,
        model: &str,
        prompt: &str,
        timestamp: u64,
    ) -> GalleryImage {
        GalleryImage {
            filename: filename.into(),
            metadata: mold_core::OutputMetadata {
                collection: None,
                tags: None,
                title: None,
                source_fit: None,
                guidance_overrides: None,
                sample_shift: None,
                distill_strength_high: None,
                distill_strength_low: None,
                job_id: None,
                prompt: prompt.into(),
                negative_prompt: None,
                original_prompt: None,
                prompt_transform: None,
                batch_id: None,
                batch_index: None,
                batch_count: None,
                output_mode: None,
                model: model.into(),
                seed: 1,
                steps: 4,
                guidance: 0.0,
                width: 768,
                height: 768,
                generation_width: Some(768),
                generation_height: Some(768),
                strength: None,
                source_image_name: None,
                source_image_sha256: None,
                edit_image_sha256s: None,
                references: None,
                keyframes: None,
                scheduler: None,
                output_format: Some(OutputFormat::Png),
                cfg_plus: None,
                lora: None,
                lora_scale: None,
                loras: None,
                control_model: None,
                control_scale: None,
                upscale_model: None,
                gif_preview: None,
                enable_audio: None,
                audio_file_path: None,
                source_video_path: None,
                extend_video_path: None,
                extend_overlap_frames: None,
                pipeline: None,
                pipeline_requested: None,
                duration_prediction_requested: None,
                pipeline_provenance_sha256: None,
                source_preprocessing: None,
                ic_lora_control: None,
                hdr_exr_dir: None,
                hdr_exr_full_float: false,
                retake_range: None,
                spatial_upscale: None,
                temporal_upscale: None,
                frames: None,
                fps: None,
                chain_job_id: None,
                chain: None,
                version: "test".into(),
                id_image_name: None,
                id_image_sha256: None,
                id_weight: None,
                id_start_step: None,
                id_image_names: None,
                id_image_sha256s: None,
                true_cfg: None,
                cfg_start_step: None,
            },
            timestamp,
            format: Some(OutputFormat::Png),
            size_bytes: Some(123),
            media_version: Some(format!("{timestamp}:123")),
            metadata_synthetic: false,
            title: None,
            tags: Vec::new(),
            favorite: false,
            collections: Vec::new(),
            trashed_at: None,
            purge_at: None,
        }
    }

    #[test]
    fn mcp_generate_tools_keep_the_prompt_required() {
        // Optional prompts are gated on visual conditioning (source image,
        // keyframes, source video, extend). The MCP generate tools accept none
        // of those, so both the JSON-Schema `required` array and the runtime
        // check must stay strict — and must move in lockstep if that changes.
        let tools = tool_definitions();
        for name in ["generate_image", "generate_image_async"] {
            let tool = tools
                .as_array()
                .unwrap()
                .iter()
                .find(|tool| tool["name"] == name)
                .unwrap_or_else(|| panic!("{name} should be advertised"));
            let schema = &tool["inputSchema"];
            let properties = schema["properties"].as_object().unwrap();
            for conditioning in ["source_image", "keyframes", "source_video", "extend_video"] {
                assert!(
                    !properties.contains_key(conditioning),
                    "{name} grew a {conditioning} input; revisit the prompt requirement"
                );
            }
            assert_eq!(
                schema["required"],
                json!(["prompt"]),
                "{name} must keep prompt required"
            );
        }

        let args: GenerateImageArgs = serde_json::from_value(json!({ "prompt": "   " })).unwrap();
        assert!(build_generate_request(args, None)
            .unwrap_err()
            .contains("prompt"));
    }
}
