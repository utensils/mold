//! Live server-owned adaptive batch execution.
//!
//! The public parent is normalized once. Singleton child requests are then
//! admitted through the ordinary authoritative scheduler, but their outputs
//! remain private until the durable reducer and gallery transaction commit the
//! entire ordered parent.

use crate::batch_attempt::DurableBatchAttempt;
use crate::batch_parent::{BatchChildLease, ChildCompletion, CompletionDisposition};
use crate::execution_plan::{DeviceFact, PreparedExecutionInputs, ResolvedExecutionPlan};
use crate::queue::{build_sse_complete_event, SavedOutputNames};
use crate::state::{
    AppState, BatchChildExecution, GenerationJob, GenerationJobResult, SseCompletionPayload,
    SubmitError,
};
use anyhow::{ensure, Context as _};
use mold_core::{
    BatchGenerateOutput, BatchGenerateResponse, GenerateRequest, OutputMetadata,
    SseBatchCompleteEvent,
};
use mold_db::{GenerationRecord, RecordSource};
use mold_scheduler::{
    AdaptiveBatchPlan, BatchDeviceProfile, BatchPartitionPlanner, BatchPartitionRequest,
    BatchSizeEstimate,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const LIVE_BATCH_RECOVERY_VERSION: u32 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LiveBatchRecoveryEnvelope {
    version: u32,
    request: GenerateRequest,
    execution_equivalence_fingerprint: String,
}

#[derive(Debug)]
struct FrozenBatchPlan {
    adaptive: AdaptiveBatchPlan,
    equivalence: String,
    prepared_inputs: PreparedExecutionInputs,
    ordinal_by_device: BTreeMap<String, usize>,
}

pub(crate) struct CompletedServerBatch {
    pub response: BatchGenerateResponse,
    pub event: SseBatchCompleteEvent,
}

struct BatchParentRegistration {
    registry: crate::job_registry::SharedJobRegistry,
    parent_id: String,
    commit_closed: bool,
}

pub(crate) struct RegisteredServerBatch {
    parent_id: String,
    registration: BatchParentRegistration,
    cancel: tokio_util::sync::CancellationToken,
}

impl RegisteredServerBatch {
    pub(crate) fn id(&self) -> &str {
        &self.parent_id
    }
}

pub(crate) fn register_server_batch(state: &AppState) -> RegisteredServerBatch {
    let parent_id = uuid::Uuid::new_v4().to_string();
    let (registration, cancel) = BatchParentRegistration::new(state, &parent_id);
    RegisteredServerBatch {
        parent_id,
        registration,
        cancel,
    }
}

impl BatchParentRegistration {
    fn new(state: &AppState, parent_id: &str) -> (Self, tokio_util::sync::CancellationToken) {
        let cancel = state.job_registry.register_batch_parent(parent_id);
        (
            Self {
                registry: state.job_registry.clone(),
                parent_id: parent_id.to_string(),
                commit_closed: false,
            },
            cancel,
        )
    }

    fn begin_commit(&mut self) -> bool {
        let accepted = self.registry.begin_batch_commit(&self.parent_id);
        self.commit_closed = true;
        accepted
    }
}

impl Drop for BatchParentRegistration {
    fn drop(&mut self) {
        if !self.commit_closed {
            self.registry.remove_batch_parent(&self.parent_id);
        }
    }
}

fn current_device_facts(state: &AppState) -> Vec<DeviceFact> {
    let resources = state.resources.latest();
    state
        .gpu_pool
        .schedulable_workers()
        .into_iter()
        .map(|worker| {
            // Scheduler-owned model allocations are reclaimable for a future
            // queued child. Only memory attributed to other processes reduces
            // the device's future execution capacity here.
            let available = resources
                .as_ref()
                .and_then(|snapshot| {
                    snapshot
                        .gpus
                        .iter()
                        .find(|gpu| gpu.ordinal == worker.gpu.ordinal)
                })
                .map(|gpu| {
                    gpu.vram_total
                        .saturating_sub(gpu.vram_used_by_other.unwrap_or(0))
                })
                .unwrap_or(worker.gpu.total_vram_bytes);
            DeviceFact {
                id: crate::scheduler::worker_device_id(&worker),
                ordinal: worker.gpu.ordinal,
                backend: worker.gpu.backend,
                compute_capability: worker.gpu.compute_capability,
                available_vram_bytes: available,
            }
        })
        .collect()
}

fn device_profile(
    child_count: u32,
    plan: &ResolvedExecutionPlan,
    candidate: &mold_core::GenerationPlacementCandidate,
) -> BatchDeviceProfile {
    let total_ms = candidate
        .predicted_completion_after_ms
        .saturating_sub(candidate.predicted_start_after_ms);
    BatchDeviceProfile {
        device_id: mold_scheduler::DeviceId::new(plan.device_id.clone()),
        available_at_ms: candidate.predicted_start_after_ms,
        initially_warm: candidate.setup_kind == "warm",
        partition_capacity: child_count,
        available_vram_bytes: plan.admitted_available_vram_bytes,
        cold_setup_ms: candidate.setup_ms,
        warm_setup_ms: if candidate.setup_kind == "warm" {
            candidate.setup_ms
        } else {
            0
        },
        setup_host_ram_bytes: plan.predicted_host_increment_bytes,
        size_estimates: vec![BatchSizeEstimate {
            size: 1,
            predicted_run_ms: total_ms.saturating_sub(candidate.setup_ms).max(1),
            predicted_vram_bytes: plan.predicted_vram_peak_bytes,
            predicted_host_ram_bytes: plan.predicted_host_increment_bytes,
        }],
    }
}

fn partition_request(
    child_count: u32,
    devices: Vec<BatchDeviceProfile>,
    host_headroom_bytes: u64,
) -> BatchPartitionRequest {
    BatchPartitionRequest {
        child_count,
        now_ms: 0,
        native_batch_sizes: vec![1],
        host_headroom_bytes,
        devices,
    }
}

fn select_equivalent_plan(
    child_count: u32,
    groups: BTreeMap<String, Vec<BatchDeviceProfile>>,
    host_headroom_bytes: u64,
) -> anyhow::Result<(AdaptiveBatchPlan, String)> {
    let mut candidates = groups
        .into_iter()
        .filter_map(|(equivalence, devices)| {
            BatchPartitionPlanner::plan(&partition_request(
                child_count,
                devices,
                host_headroom_bytes,
            ))
            .ok()
            .map(|adaptive| (adaptive, equivalence))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.0
            .predicted_parent_makespan_ms
            .cmp(&right.0.predicted_parent_makespan_ms)
            .then_with(|| right.0.devices_used.cmp(&left.0.devices_used))
            .then_with(|| left.1.cmp(&right.1))
    });
    candidates
        .into_iter()
        .next()
        .context("no execution-equivalent device set can run this batch")
}

async fn freeze_batch_plan(
    state: &AppState,
    parent_id: &str,
    request: &GenerateRequest,
) -> anyhow::Result<FrozenBatchPlan> {
    let mut execution_request = request.clone();
    execution_request.batch_size = 1;
    let prepared_inputs = crate::variant_dependencies::prepare_execution_inputs(
        state,
        parent_id,
        &execution_request,
        None,
    )
    .await
    .map_err(anyhow::Error::msg)?;
    let device_facts = current_device_facts(state);
    ensure!(!device_facts.is_empty(), "no schedulable batch devices");
    let config = state.config.read().await.clone();
    let offload_requested = matches!(
        mold_inference::runtime_env::value("MOLD_OFFLOAD").as_deref(),
        Some("1") | Some("true") | Some("yes")
    );
    let plans = crate::execution_plan::resolve_execution_plans_with_prepared(
        &config,
        &execution_request,
        &device_facts,
        offload_requested,
        Some(&prepared_inputs),
    )?;
    let host_headroom_bytes = state
        .resources
        .latest()
        .and_then(|snapshot| snapshot.system_ram.available)
        .unwrap_or(u64::MAX / 4);
    let preview = state
        .scheduled_work
        .preview_placement(
            execution_request,
            request.batch_size,
            prepared_inputs.clone(),
        )
        .await
        .map_err(anyhow::Error::msg)?;
    ensure!(
        preview.authoritative && preview.outcome == "planned",
        "authoritative scheduler could not preview batch placement: {}",
        preview.reason.unwrap_or(preview.outcome)
    );
    let generation_stage = preview
        .stage_candidates
        .iter()
        .filter(|stage| stage.copy_index.is_some())
        .map(|stage| stage.stage_index)
        .min()
        .context("batch placement preview has no generation stage")?;
    let timing_by_edge = preview
        .stage_candidates
        .into_iter()
        .filter(|stage| stage.stage_index == generation_stage && stage.copy_index.is_some())
        .map(|stage| {
            (
                (
                    stage.candidate.device_id.clone(),
                    stage
                        .candidate
                        .execution_equivalence_fingerprint
                        .clone()
                        .unwrap_or_default(),
                ),
                stage.candidate,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut groups = BTreeMap::<String, Vec<BatchDeviceProfile>>::new();
    for plan in &plans {
        let equivalence = plan.execution_equivalence_fingerprint.to_string();
        let Some(candidate) = timing_by_edge.get(&(plan.device_id.clone(), equivalence.clone()))
        else {
            continue;
        };
        groups.entry(equivalence).or_default().push(device_profile(
            request.batch_size,
            plan,
            candidate,
        ));
    }
    let (adaptive, equivalence) =
        select_equivalent_plan(request.batch_size, groups, host_headroom_bytes)?;
    let ordinal_by_device = device_facts
        .into_iter()
        .map(|device| (device.id, device.ordinal))
        .collect();
    Ok(FrozenBatchPlan {
        adaptive,
        equivalence,
        prepared_inputs,
        ordinal_by_device,
    })
}

fn frozen_seed(parent_id: &str, requested: Option<u64>) -> u64 {
    requested.unwrap_or_else(|| {
        let id = uuid::Uuid::parse_str(parent_id).unwrap_or_else(|_| uuid::Uuid::new_v4());
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(&id.as_bytes()[..8]);
        u64::from_le_bytes(bytes)
    })
}

/// Conservative encoded-output bound used by the atomic staging preflight.
///
/// Lossless images are bounded from the final RGBA raster. Video is bounded
/// from every uncompressed RGBA frame (encoded containers are smaller in
/// normal operation), plus generated PCM audio and explicit container/journal
/// headroom. Post-generation spatial/temporal upscalers are applied before the
/// bound so admission covers the artifact that is actually staged.
fn conservative_batch_output_bytes(request: &GenerateRequest) -> anyhow::Result<u64> {
    const BYTES_PER_RGBA_PIXEL: u64 = 4;
    const AUDIO_SAMPLE_RATE: u64 = 48_000;
    const AUDIO_CHANNELS: u64 = 2;
    const AUDIO_BYTES_PER_SAMPLE: u64 = 4;
    const PER_CHILD_OVERHEAD: u64 = 1024 * 1024;

    let mut area_multiplier = 1_u64;
    if request.upscale_model.is_some() {
        // Every currently supported post-upscaler is at most 4x linear.
        area_multiplier = area_multiplier
            .checked_mul(16)
            .context("batch output-size upscale area overflow")?;
    }
    if request.spatial_upscale.is_some() {
        area_multiplier = area_multiplier
            .checked_mul(4)
            .context("batch output-size spatial upscale overflow")?;
    }
    let mut frame_count = u64::from(request.frames.unwrap_or(1));
    if request.temporal_upscale.is_some() {
        frame_count = frame_count
            .checked_mul(2)
            .context("batch output-size temporal upscale overflow")?;
    }
    let pixels = u64::from(request.width)
        .checked_mul(u64::from(request.height))
        .and_then(|value| value.checked_mul(area_multiplier))
        .context("batch output-size pixel count overflow")?;
    let raw_media_bytes = pixels
        .checked_mul(frame_count)
        .and_then(|value| value.checked_mul(BYTES_PER_RGBA_PIXEL))
        .context("batch output-size frame bytes overflow")?;

    let audio_bytes = if request.frames.is_some() && request.enable_audio != Some(false) {
        let fps = u64::from(request.fps.unwrap_or(24).max(1));
        let duration_seconds = frame_count.div_ceil(fps).max(1);
        duration_seconds
            .checked_mul(AUDIO_SAMPLE_RATE)
            .and_then(|value| value.checked_mul(AUDIO_CHANNELS))
            .and_then(|value| value.checked_mul(AUDIO_BYTES_PER_SAMPLE))
            .context("batch output-size audio bytes overflow")?
    } else {
        0
    };
    let preview_bytes = if request.gif_preview {
        // A requested animated preview may eventually be staged with the
        // primary artifact. Reserving another raw-frame bound is deliberately
        // conservative for palette-encoded GIF.
        raw_media_bytes
    } else {
        0
    };
    let payload = raw_media_bytes
        .checked_add(audio_bytes)
        .and_then(|value| value.checked_add(preview_bytes))
        .context("batch output-size payload overflow")?;
    let per_child = payload
        .checked_add(payload / 4)
        .and_then(|value| value.checked_add(PER_CHILD_OVERHEAD))
        .context("batch output-size overhead overflow")?;
    per_child
        .checked_mul(u64::from(request.batch_size))
        .context("batch output-size child total overflow")
}

fn normalize_children(
    parent_id: &str,
    request: &GenerateRequest,
    base_seed: u64,
) -> Vec<GenerateRequest> {
    (0..request.batch_size)
        .map(|index| {
            let mut child = request.clone();
            child.batch_size = 1;
            child.seed = Some(base_seed.wrapping_add(u64::from(index)));
            child.batch_id = Some(parent_id.to_string());
            child.batch_index = Some(index + 1);
            child.batch_count = Some(request.batch_size);
            child
        })
        .collect()
}

fn recovery_envelope(
    request: &GenerateRequest,
    plan: &FrozenBatchPlan,
) -> LiveBatchRecoveryEnvelope {
    LiveBatchRecoveryEnvelope {
        version: LIVE_BATCH_RECOVERY_VERSION,
        request: request.clone(),
        execution_equivalence_fingerprint: plan.equivalence.clone(),
    }
}

fn decode_recovery_envelope(
    value: &serde_json::Value,
) -> anyhow::Result<LiveBatchRecoveryEnvelope> {
    let envelope: LiveBatchRecoveryEnvelope = serde_json::from_value(value.clone())
        .context("durable live-batch recovery envelope is missing or invalid")?;
    ensure!(
        envelope.version == LIVE_BATCH_RECOVERY_VERSION,
        "unsupported live-batch recovery envelope version {}",
        envelope.version
    );
    ensure!(
        envelope.request.batch_size > 1,
        "durable live-batch recovery request is not a parent batch"
    );
    ensure!(
        envelope.request.seed.is_some(),
        "durable live-batch recovery request has no frozen seed"
    );
    ensure!(
        !envelope.execution_equivalence_fingerprint.is_empty(),
        "durable live-batch recovery equivalence is empty"
    );
    Ok(envelope)
}

fn batch_records(
    output_dir: &Path,
    parent: &GenerateRequest,
    children: &[GenerateRequest],
) -> Vec<GenerationRecord> {
    let timestamp = mold_core::time::now_epoch_ms_u64();
    let format = parent.resolved_output_format();
    let extension = format.extension();
    children
        .iter()
        .enumerate()
        .map(|(index, child)| {
            let filename = mold_core::default_output_filename(
                &parent.model,
                timestamp,
                extension,
                parent.batch_size,
                index as u32,
            );
            let metadata = OutputMetadata::from_generate_request(
                child,
                child.seed.expect("server batch child seed is frozen"),
                child.scheduler,
                mold_core::build_info::version_string(),
            );
            GenerationRecord::from_save(
                output_dir,
                filename,
                format,
                metadata,
                RecordSource::Server,
                timestamp.try_into().unwrap_or(i64::MAX),
            )
        })
        .collect()
}

fn media_bytes(result: &GenerationJobResult) -> &[u8] {
    result
        .response
        .video
        .as_ref()
        .map_or(result.image.data.as_slice(), |video| video.data.as_slice())
}

fn completed_record(
    output_dir: &Path,
    filename: &str,
    request: &GenerateRequest,
    result: &GenerationJobResult,
) -> GenerationRecord {
    let mut metadata = OutputMetadata::from_generate_request(
        request,
        result.response.seed_used,
        request.scheduler,
        mold_core::build_info::version_string(),
    );
    let format = if let Some(video) = &result.response.video {
        metadata.width = video.width;
        metadata.height = video.height;
        metadata.frames = Some(video.frames);
        metadata.fps = Some(video.fps);
        video.format
    } else {
        crate::queue::apply_output_dimensions_to_metadata(&mut metadata, &result.image);
        result.image.format
    };
    let mut record = GenerationRecord::from_save(
        output_dir,
        filename,
        format,
        metadata,
        RecordSource::Server,
        mold_core::time::now_epoch_ms_u64()
            .try_into()
            .unwrap_or(i64::MAX),
    );
    record.generation_time_ms = Some(
        result
            .response
            .generation_time_ms
            .try_into()
            .unwrap_or(i64::MAX),
    );
    record.backend = Some(mold_inference::compiled_backend_label().to_string());
    record
}

async fn submit_child(
    state: &AppState,
    parent_id: &str,
    request: GenerateRequest,
    lease: BatchChildLease,
    plan: &FrozenBatchPlan,
    ordinal: usize,
    retry: u8,
) -> anyhow::Result<(
    tokio::sync::oneshot::Receiver<Result<GenerationJobResult, String>>,
    std::sync::Arc<tokio::sync::Notify>,
)> {
    let id = if lease.child_index == 0 && retry == 0 {
        parent_id.to_string()
    } else {
        format!("{parent_id}:child:{}:try:{retry}", lease.child_index + 1)
    };
    let metadata = Box::new(OutputMetadata::from_generate_request(
        &request,
        request.seed.unwrap_or(0),
        request.scheduler,
        mold_core::build_info::version_string(),
    ));
    let cancel = state.job_registry.register_job(
        &id,
        &request.model,
        Some(ordinal),
        Some(true),
        Some(metadata),
    );
    if !state.job_registry.register_batch_child(parent_id, &id) {
        state.job_registry.remove(&id);
        anyhow::bail!("batch parent cancellation authority is no longer open");
    }
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let job = GenerationJob {
        id: id.clone(),
        request,
        completion_payload: SseCompletionPayload::Full,
        progress_tx: None,
        result_tx,
        output_dir: None,
        batch_child: Some(BatchChildExecution {
            lease,
            execution_equivalence_fingerprint: plan.equivalence.clone(),
            prepared_inputs: plan.prepared_inputs.clone(),
        }),
    };
    state
        .queue
        .submit(job, state.queue_capacity)
        .await
        .inspect_err(|_| state.job_registry.remove(&id))
        .map_err(|error| match error {
            SubmitError::Full { pending, capacity } => {
                anyhow::anyhow!("generation queue is full ({pending}/{capacity}); retry later")
            }
            SubmitError::Shutdown => anyhow::anyhow!("generation queue is shutting down"),
        })?;
    Ok((result_rx, cancel))
}

/// Execute a normalized server-owned parent. No child writes to the gallery.
pub(crate) async fn execute_server_batch(
    state: &AppState,
    authority: RegisteredServerBatch,
    mut request: GenerateRequest,
    output_dir: PathBuf,
    completion_payload: SseCompletionPayload,
) -> anyhow::Result<CompletedServerBatch> {
    ensure!(
        request.batch_size > 1,
        "server batch execution requires batch_size > 1"
    );
    ensure!(
        request.batch_id.is_none()
            && request.batch_index.is_none()
            && request.batch_count.is_none(),
        "raw server batches cannot also carry prepared-sibling batch authority"
    );
    let RegisteredServerBatch {
        parent_id,
        registration: mut parent_registration,
        cancel: parent_cancel,
    } = authority;
    ensure!(
        !parent_cancel.is_cancelled(),
        "batch parent cancelled before planning"
    );
    let plan = freeze_batch_plan(state, &parent_id, &request).await?;
    ensure!(
        !parent_cancel.is_cancelled(),
        "batch parent cancelled during planning"
    );
    let base_seed = frozen_seed(&parent_id, request.seed);
    request.seed = Some(base_seed);
    let children = normalize_children(&parent_id, &request, base_seed);
    let records = batch_records(&output_dir, &request, &children);
    let estimated_bytes = conservative_batch_output_bytes(&request)?;
    crate::batch_transaction::preflight_disk_space(&output_dir, estimated_bytes)?;
    let normalized = serde_json::to_value(recovery_envelope(&request, &plan))?;
    let mut attempt = DurableBatchAttempt::begin(&output_dir, &parent_id, normalized, records)?;
    attempt.start()?;

    let mut receivers = Vec::with_capacity(children.len());
    let mut initial_error = None;
    for (index, child) in children.iter().cloned().enumerate() {
        let partition = plan.adaptive.partition_at(index as u32)?;
        ensure!(
            partition.child_start == index as u32 && partition.size == 1,
            "singleton production batch planner returned a non-singleton partition"
        );
        let ordinal = *plan
            .ordinal_by_device
            .get(partition.device_id.as_str())
            .context("adaptive batch plan references an unknown device")?;
        let (lease, _) = attempt.grant(index)?;
        match submit_child(state, &parent_id, child, lease.clone(), &plan, ordinal, 0).await {
            Ok(submitted) => receivers.push((index, lease, ordinal, submitted.0, submitted.1)),
            Err(error) => {
                let _ = attempt.complete_without_artifact(&lease, ChildCompletion::Cancelled)?;
                initial_error = Some(error);
                break;
            }
        }
    }

    let mut results = Vec::with_capacity(children.len());
    let mut terminal_error = initial_error;
    if terminal_error.is_some() {
        // Submission failed after earlier siblings were admitted. Close the
        // one parent authority now so queued siblings are removed immediately
        // and running siblings only drain; none may publish a late receipt.
        let _ = state.job_registry.cancel_queued(&parent_id);
    }
    let mut parent_cancel_reduced = false;
    for (index, mut lease, ordinal, mut receiver, mut cancel) in receivers {
        let mut retry = 0_u8;
        loop {
            let result = tokio::select! {
                result = &mut receiver => result.unwrap_or_else(|_| {
                    Err("batch child worker dropped its result".to_string())
                }),
                _ = parent_cancel.cancelled() => Err("batch parent cancelled".to_string()),
                _ = cancel.notified() => Err("cancelled".to_string()),
            };
            if parent_cancel.is_cancelled() && !parent_cancel_reduced {
                if matches!(
                    attempt.parent().state(),
                    crate::batch_parent::BatchParentState::Queued
                        | crate::batch_parent::BatchParentState::Running
                        | crate::batch_parent::BatchParentState::Prepared
                ) {
                    let _ = attempt.request_cancel()?;
                }
                parent_cancel_reduced = true;
                terminal_error.get_or_insert_with(|| anyhow::anyhow!("batch parent cancelled"));
            }
            match result {
                Ok(result) => {
                    if terminal_error.is_some() {
                        let _ = attempt
                            .complete_without_artifact(&lease, ChildCompletion::Cancelled)?;
                        break;
                    }
                    let filename = attempt.transaction().manifest().children[index]
                        .final_name
                        .clone();
                    let record =
                        completed_record(&output_dir, &filename, &children[index], &result);
                    let disposition =
                        attempt.stage_record_and_accept(&lease, record, media_bytes(&result))?;
                    ensure!(
                        matches!(
                            disposition,
                            CompletionDisposition::Accepted
                                | CompletionDisposition::AttemptPrepared
                        ),
                        "live batch child completion lost parent authority: {disposition:?}"
                    );
                    results.push(result);
                    break;
                }
                Err(error) => {
                    let completion = if error.contains("cancelled") {
                        ChildCompletion::Cancelled
                    } else {
                        ChildCompletion::Failed
                    };
                    let disposition = attempt.complete_without_artifact(&lease, completion)?;
                    if disposition == CompletionDisposition::RetryChild {
                        retry = retry.saturating_add(1);
                        let granted = attempt.grant(index)?;
                        lease = granted.0;
                        let submitted = submit_child(
                            state,
                            &parent_id,
                            children[index].clone(),
                            lease.clone(),
                            &plan,
                            ordinal,
                            retry,
                        )
                        .await;
                        match submitted {
                            Ok(submitted) => {
                                receiver = submitted.0;
                                cancel = submitted.1;
                                continue;
                            }
                            Err(submit_error) => {
                                let _ = attempt.complete_without_artifact(
                                    &lease,
                                    ChildCompletion::Cancelled,
                                )?;
                                if terminal_error.is_none() {
                                    terminal_error = Some(submit_error);
                                    let _ = state.job_registry.cancel_queued(&parent_id);
                                }
                                break;
                            }
                        }
                    }
                    if terminal_error.is_none() {
                        terminal_error =
                            Some(anyhow::anyhow!("batch child {} failed: {error}", index + 1));
                        let _ = state.job_registry.cancel_queued(&parent_id);
                    }
                    break;
                }
            }
        }
    }
    if let Some(error) = terminal_error {
        ensure!(
            attempt.parent().state() == crate::batch_parent::BatchParentState::Fenced,
            "failed batch did not drain every active child"
        );
        attempt.rollback_fenced()?;
        return Err(error);
    }
    if parent_cancel.is_cancelled() || !parent_registration.begin_commit() {
        let _ = attempt.request_cancel()?;
        ensure!(
            attempt.parent().state() == crate::batch_parent::BatchParentState::Fenced,
            "cancelled prepared batch did not fence"
        );
        attempt.rollback_fenced()?;
        anyhow::bail!("batch parent cancelled");
    }
    ensure!(
        results.len() == children.len(),
        "batch completed without every ordered child"
    );
    attempt
        .converge_commit(&state.gallery_publication_gate, state.metadata_db.clone())
        .await?;

    let filenames = attempt
        .transaction()
        .manifest()
        .children
        .iter()
        .map(|child| child.final_name.clone())
        .collect::<Vec<_>>();
    let mut outputs = Vec::with_capacity(results.len());
    let mut events = Vec::with_capacity(results.len());
    for (index, (result, filename)) in results.into_iter().zip(filenames).enumerate() {
        let metadata = OutputMetadata::from_generate_request(
            &children[index],
            result.response.seed_used,
            children[index].scheduler,
            mold_core::build_info::version_string(),
        );
        let saved = SavedOutputNames {
            output: Some(filename.clone()),
            original: None,
        };
        events.push(build_sse_complete_event(
            &result.response,
            &result.image,
            None,
            Some(&metadata),
            &saved,
            completion_payload,
        ));
        state.events.publish(mold_core::ServerEvent::GalleryAdded {
            filename: filename.clone(),
            image: None,
        });
        outputs.push(BatchGenerateOutput {
            batch_index: index as u32 + 1,
            filename,
            response: result.response,
        });
    }
    Ok(CompletedServerBatch {
        response: BatchGenerateResponse {
            batch_id: parent_id.clone(),
            outputs,
        },
        event: SseBatchCompleteEvent {
            batch_id: parent_id,
            outputs: events,
        },
    })
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LiveBatchResumeReport {
    pub resumed: usize,
    pub committed: usize,
    pub rolled_back: usize,
}

fn pending_child_indices(attempt: &DurableBatchAttempt) -> anyhow::Result<Vec<usize>> {
    let mut cursor = 0;
    let mut indices = Vec::new();
    loop {
        let window = attempt.parent().pending_window(cursor, 1024)?;
        indices.extend(window.indices);
        let Some(next) = window.next_cursor else {
            break;
        };
        cursor = next;
    }
    Ok(indices)
}

fn terminally_cancel_recovered_attempt(attempt: &mut DurableBatchAttempt) -> anyhow::Result<()> {
    if matches!(
        attempt.parent().state(),
        crate::batch_parent::BatchParentState::Queued
            | crate::batch_parent::BatchParentState::Running
            | crate::batch_parent::BatchParentState::Retrying
    ) {
        let _ = attempt.request_cancel()?;
    }
    ensure!(
        attempt.parent().state() == crate::batch_parent::BatchParentState::Fenced,
        "unresumable recovered batch did not reach its durable fence"
    );
    attempt.rollback_fenced()?;
    attempt.finalize_fence()?;
    ensure!(
        attempt.parent().state() == crate::batch_parent::BatchParentState::Cancelled,
        "unresumable recovered batch did not persist its terminal state"
    );
    Ok(())
}

async fn resume_recovered_batch(
    state: &AppState,
    output_dir: &Path,
    parent_id: &str,
) -> anyhow::Result<bool> {
    let (mut attempt, _) = DurableBatchAttempt::recover(output_dir, parent_id)?;
    let envelope =
        match decode_recovery_envelope(&attempt.transaction().manifest().normalized_request) {
            Ok(envelope) => envelope,
            Err(error) => {
                tracing::error!(%parent_id, error = %error, "rolling back unresumable live batch");
                terminally_cancel_recovered_attempt(&mut attempt)?;
                return Ok(false);
            }
        };
    if envelope.request.batch_size as usize != attempt.parent().total_children() {
        tracing::error!(
            %parent_id,
            request_children = envelope.request.batch_size,
            durable_children = attempt.parent().total_children(),
            "rolling back live batch with inconsistent durable child count"
        );
        terminally_cancel_recovered_attempt(&mut attempt)?;
        return Ok(false);
    }

    let (mut parent_registration, parent_cancel) = BatchParentRegistration::new(state, parent_id);
    let plan = match freeze_batch_plan(state, parent_id, &envelope.request).await {
        Ok(plan)
            if plan.equivalence == envelope.execution_equivalence_fingerprint
                && !parent_cancel.is_cancelled() =>
        {
            plan
        }
        Ok(plan) => {
            tracing::error!(
                %parent_id,
                expected = %envelope.execution_equivalence_fingerprint,
                actual = %plan.equivalence,
                cancelled = parent_cancel.is_cancelled(),
                "rolling back live batch because exact recovery authority drifted"
            );
            terminally_cancel_recovered_attempt(&mut attempt)?;
            return Ok(false);
        }
        Err(error) => {
            tracing::error!(%parent_id, error = %error, "rolling back live batch that cannot be replanned exactly");
            terminally_cancel_recovered_attempt(&mut attempt)?;
            return Ok(false);
        }
    };

    let request = envelope.request;
    if matches!(
        attempt.parent().state(),
        crate::batch_parent::BatchParentState::Queued
            | crate::batch_parent::BatchParentState::Retrying
    ) {
        attempt.start()?;
    }
    let base_seed = request
        .seed
        .expect("validated recovery envelope has a frozen seed");
    let children = normalize_children(parent_id, &request, base_seed);
    let pending = pending_child_indices(&attempt)?;
    let mut receivers = Vec::with_capacity(pending.len());
    let mut terminal_error = None;
    for index in pending {
        let partition = plan.adaptive.partition_at(index as u32)?;
        ensure!(
            partition.child_start == index as u32 && partition.size == 1,
            "recovered singleton batch planner returned a non-singleton partition"
        );
        let ordinal = *plan
            .ordinal_by_device
            .get(partition.device_id.as_str())
            .context("recovered batch plan references an unknown device")?;
        let (lease, _) = attempt.grant(index)?;
        match submit_child(
            state,
            parent_id,
            children[index].clone(),
            lease.clone(),
            &plan,
            ordinal,
            0,
        )
        .await
        {
            Ok(submitted) => {
                receivers.push((index, lease, ordinal, submitted.0, submitted.1));
            }
            Err(error) => {
                let _ = attempt.complete_without_artifact(&lease, ChildCompletion::Cancelled)?;
                terminal_error = Some(error);
                let _ = state.job_registry.cancel_queued(parent_id);
                break;
            }
        }
    }

    let mut parent_cancel_reduced = false;
    for (index, mut lease, ordinal, mut receiver, mut cancel) in receivers {
        let mut retry = 0_u8;
        loop {
            let result = tokio::select! {
                result = &mut receiver => result.unwrap_or_else(|_| {
                    Err("recovered batch child worker dropped its result".to_string())
                }),
                _ = parent_cancel.cancelled() => Err("batch parent cancelled".to_string()),
                _ = cancel.notified() => Err("cancelled".to_string()),
            };
            if parent_cancel.is_cancelled() && !parent_cancel_reduced {
                if matches!(
                    attempt.parent().state(),
                    crate::batch_parent::BatchParentState::Queued
                        | crate::batch_parent::BatchParentState::Running
                        | crate::batch_parent::BatchParentState::Prepared
                ) {
                    let _ = attempt.request_cancel()?;
                }
                parent_cancel_reduced = true;
                terminal_error
                    .get_or_insert_with(|| anyhow::anyhow!("recovered batch parent cancelled"));
            }
            match result {
                Ok(result) if terminal_error.is_none() => {
                    let filename = attempt.transaction().manifest().children[index]
                        .final_name
                        .clone();
                    let record = completed_record(output_dir, &filename, &children[index], &result);
                    let disposition =
                        attempt.stage_record_and_accept(&lease, record, media_bytes(&result))?;
                    ensure!(
                        matches!(
                            disposition,
                            CompletionDisposition::Accepted
                                | CompletionDisposition::AttemptPrepared
                        ),
                        "recovered batch child completion lost parent authority: {disposition:?}"
                    );
                    break;
                }
                Ok(_) => {
                    let _ =
                        attempt.complete_without_artifact(&lease, ChildCompletion::Cancelled)?;
                    break;
                }
                Err(error) => {
                    let completion = if error.contains("cancelled") {
                        ChildCompletion::Cancelled
                    } else {
                        ChildCompletion::Failed
                    };
                    let disposition = attempt.complete_without_artifact(&lease, completion)?;
                    if disposition == CompletionDisposition::RetryChild {
                        retry = retry.saturating_add(1);
                        lease = attempt.grant(index)?.0;
                        match submit_child(
                            state,
                            parent_id,
                            children[index].clone(),
                            lease.clone(),
                            &plan,
                            ordinal,
                            retry,
                        )
                        .await
                        {
                            Ok(submitted) => {
                                receiver = submitted.0;
                                cancel = submitted.1;
                                continue;
                            }
                            Err(submit_error) => {
                                let _ = attempt.complete_without_artifact(
                                    &lease,
                                    ChildCompletion::Cancelled,
                                )?;
                                if terminal_error.is_none() {
                                    terminal_error = Some(submit_error);
                                    let _ = state.job_registry.cancel_queued(parent_id);
                                }
                                break;
                            }
                        }
                    }
                    if terminal_error.is_none() {
                        terminal_error = Some(anyhow::anyhow!(
                            "recovered batch child {} failed: {error}",
                            index + 1
                        ));
                        let _ = state.job_registry.cancel_queued(parent_id);
                    }
                    break;
                }
            }
        }
    }

    if terminal_error.is_some() {
        ensure!(
            attempt.parent().state() == crate::batch_parent::BatchParentState::Fenced,
            "failed recovered batch did not drain every active child"
        );
        attempt.rollback_fenced()?;
        attempt.finalize_fence()?;
        return Ok(false);
    }
    if parent_cancel.is_cancelled() || !parent_registration.begin_commit() {
        let _ = attempt.request_cancel()?;
        ensure!(
            attempt.parent().state() == crate::batch_parent::BatchParentState::Fenced,
            "cancelled recovered batch did not fence"
        );
        attempt.rollback_fenced()?;
        attempt.finalize_fence()?;
        return Ok(false);
    }
    attempt
        .converge_commit(&state.gallery_publication_gate, state.metadata_db.clone())
        .await?;
    for child in &attempt.transaction().manifest().children {
        state.events.publish(mold_core::ServerEvent::GalleryAdded {
            filename: child.final_name.clone(),
            image: None,
        });
    }
    Ok(true)
}

pub(crate) async fn resume_recovered_batches(
    state: &AppState,
    output_dir: &Path,
    outcomes: &[crate::batch_attempt::RecoveredParentOutcome],
) -> anyhow::Result<LiveBatchResumeReport> {
    let mut report = LiveBatchResumeReport::default();
    for outcome in outcomes {
        let crate::batch_attempt::RecoveredParentOutcome::Resumable { parent_id, .. } = outcome
        else {
            continue;
        };
        report.resumed += 1;
        if resume_recovered_batch(state, output_dir, parent_id).await? {
            report.committed += 1;
        } else {
            report.rolled_back += 1;
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(device: &str) -> BatchDeviceProfile {
        BatchDeviceProfile {
            device_id: mold_scheduler::DeviceId::new(device),
            available_at_ms: 0,
            initially_warm: false,
            partition_capacity: 64,
            available_vram_bytes: 10,
            cold_setup_ms: 1,
            warm_setup_ms: 0,
            setup_host_ram_bytes: 1,
            size_estimates: vec![BatchSizeEstimate {
                size: 1,
                predicted_run_ms: 10,
                predicted_vram_bytes: 1,
                predicted_host_ram_bytes: 1,
            }],
        }
    }

    #[test]
    fn equivalent_group_uses_every_compatible_device_for_large_parent() {
        let groups = BTreeMap::from([
            (
                "same".to_string(),
                (0..8).map(|index| profile(&format!("d{index}"))).collect(),
            ),
            ("different".to_string(), vec![profile("d8")]),
        ]);
        let (adaptive, equivalence) = select_equivalent_plan(64, groups, u64::MAX / 4).unwrap();
        assert_eq!(equivalence, "same");
        assert_eq!(adaptive.devices_used, 8);
        assert_eq!(adaptive.partition_count(), 64);
    }

    #[test]
    fn normalization_freezes_order_seed_and_parent_provenance() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "batch",
            "model": "flux",
            "width": 64,
            "height": 64,
            "steps": 1,
            "batch_size": 3,
            "seed": 9
        }))
        .unwrap();
        let children = normalize_children("parent", &request, 9);
        assert_eq!(
            children.iter().map(|child| child.seed).collect::<Vec<_>>(),
            vec![Some(9), Some(10), Some(11)]
        );
        assert!(children.iter().all(|child| child.batch_size == 1));
        assert_eq!(children[2].batch_id.as_deref(), Some("parent"));
        assert_eq!(children[2].batch_index, Some(3));
        assert_eq!(children[2].batch_count, Some(3));
    }

    fn durable_attempt(directory: &Path) -> (DurableBatchAttempt, Vec<GenerateRequest>) {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "batch",
            "model": "flux",
            "width": 64,
            "height": 64,
            "steps": 1,
            "batch_size": 2,
            "seed": 9,
            "output_format": "png"
        }))
        .unwrap();
        let children = normalize_children("parent", &request, 9);
        let records = batch_records(directory, &request, &children);
        let mut attempt = DurableBatchAttempt::begin(
            directory,
            "parent",
            serde_json::to_value(&request).unwrap(),
            records,
        )
        .unwrap();
        attempt.start().unwrap();
        (attempt, children)
    }

    fn recoverable_attempt(
        directory: &Path,
        child_count: u32,
        started: bool,
    ) -> (DurableBatchAttempt, Vec<GenerateRequest>, Vec<String>) {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "restart batch",
            "model": "flux-dev:q8",
            "width": 64,
            "height": 64,
            "steps": 1,
            "batch_size": child_count,
            "seed": 9,
            "output_format": "png"
        }))
        .unwrap();
        let children = normalize_children("restart-parent", &request, 9);
        let records = batch_records(directory, &request, &children);
        let filenames = records
            .iter()
            .map(|record| record.filename.clone())
            .collect();
        let envelope = LiveBatchRecoveryEnvelope {
            version: LIVE_BATCH_RECOVERY_VERSION,
            request,
            execution_equivalence_fingerprint: "frozen-but-unavailable".to_string(),
        };
        let mut attempt = DurableBatchAttempt::begin(
            directory,
            "restart-parent",
            serde_json::to_value(envelope).unwrap(),
            records,
        )
        .unwrap();
        if started {
            attempt.start().unwrap();
        }
        (attempt, children, filenames)
    }

    #[test]
    fn cancellation_vs_last_success_discards_bytes_without_receipt() {
        let directory = tempfile::tempdir().unwrap();
        let (mut attempt, _) = durable_attempt(directory.path());
        let first = attempt.grant(0).unwrap().0;
        let last = attempt.grant(1).unwrap().0;
        assert_eq!(
            attempt.request_cancel().unwrap(),
            CompletionDisposition::Accepted
        );
        attempt
            .complete_without_artifact(&first, ChildCompletion::Cancelled)
            .unwrap();
        assert_eq!(
            attempt.stage_and_accept(&last, b"late success").unwrap(),
            CompletionDisposition::AttemptFencedDeletePrivateArtifact
        );
        assert!(attempt.transaction().staged_receipts().is_empty());
        assert_eq!(
            attempt.parent().state(),
            crate::batch_parent::BatchParentState::Fenced
        );
        attempt.rollback_fenced().unwrap();
    }

    #[test]
    fn terminal_failure_vs_last_success_discards_bytes_without_receipt() {
        let directory = tempfile::tempdir().unwrap();
        let (mut attempt, _) = durable_attempt(directory.path());
        let first = attempt.grant(0).unwrap().0;
        let last = attempt.grant(1).unwrap().0;
        assert_eq!(
            attempt
                .complete_without_artifact(&first, ChildCompletion::Failed)
                .unwrap(),
            CompletionDisposition::RetryChild
        );
        let retry = attempt.grant(0).unwrap().0;
        assert_eq!(
            attempt
                .complete_without_artifact(&retry, ChildCompletion::Failed)
                .unwrap(),
            CompletionDisposition::Accepted
        );
        assert_eq!(
            attempt.stage_and_accept(&last, b"late success").unwrap(),
            CompletionDisposition::AttemptFencedDeletePrivateArtifact
        );
        assert!(attempt.transaction().staged_receipts().is_empty());
        attempt.rollback_fenced().unwrap();
    }

    #[test]
    fn video_disk_bound_accounts_for_every_frame_and_audio() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "video batch",
            "model": "ltx-2-19b-distilled:fp8",
            "width": 1280,
            "height": 720,
            "steps": 8,
            "batch_size": 2,
            "frames": 97,
            "fps": 24,
            "enable_audio": true,
            "output_format": "mp4"
        }))
        .unwrap();
        let raster_only = u64::from(request.width)
            * u64::from(request.height)
            * 4
            * u64::from(request.batch_size);
        assert!(
            conservative_batch_output_bytes(&request).unwrap() > raster_only * 97,
            "video admission must include all frames, audio, and container overhead"
        );
    }

    #[test]
    fn post_upscale_disk_bound_accounts_for_larger_final_artifact() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "upscale batch",
            "model": "flux-dev:q8",
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "batch_size": 2,
            "output_format": "png"
        }))
        .unwrap();
        let base = conservative_batch_output_bytes(&request).unwrap();
        request.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        assert!(
            conservative_batch_output_bytes(&request).unwrap() >= base * 8,
            "a 4x linear post-upscale must reserve for the 16x-area final artifact"
        );
    }

    #[test]
    fn disk_bound_rejects_arithmetic_overflow() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "overflow",
            "model": "ltx-video:q8",
            "width": 64,
            "height": 64,
            "steps": 1,
            "batch_size": 2,
            "frames": 9,
            "output_format": "mp4"
        }))
        .unwrap();
        request.width = u32::MAX;
        request.height = u32::MAX;
        request.frames = Some(u32::MAX);
        request.batch_size = u32::MAX;
        assert!(conservative_batch_output_bytes(&request).is_err());
    }

    #[test]
    fn registered_parent_id_is_immediately_publicly_cancellable() {
        let state = AppState::for_tests();
        let authority = register_server_batch(&state);
        let parent_id = authority.id().to_string();
        assert!(uuid::Uuid::parse_str(&parent_id).is_ok());
        state.job_registry.cancel_queued(&parent_id).unwrap();
        assert!(authority.cancel.is_cancelled());
        assert!(
            !state.job_registry.begin_batch_commit(&parent_id),
            "cancellation must win before any child is registered"
        );
    }

    #[tokio::test]
    async fn restart_before_submission_fails_closed_without_publication_or_orphan() {
        let directory = tempfile::tempdir().unwrap();
        let (attempt, _, filenames) = recoverable_attempt(directory.path(), 2, false);
        drop(attempt);
        let gate = crate::batch_transaction::GalleryPublicationGate::default();
        let db = std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let recovered = crate::batch_attempt::recover_batches(directory.path(), &gate, db.clone())
            .await
            .unwrap();
        assert!(matches!(
            recovered.outcomes.as_slice(),
            [crate::batch_attempt::RecoveredParentOutcome::Resumable { .. }]
        ));

        let state = AppState::for_tests();
        let mut events = state.events.subscribe();
        let report = resume_recovered_batches(&state, directory.path(), &recovered.outcomes)
            .await
            .unwrap();
        assert_eq!(report.rolled_back, 1);
        assert_eq!(db.as_ref().as_ref().unwrap().count().unwrap(), 0);
        assert!(filenames
            .iter()
            .all(|filename| !directory.path().join(filename).exists()));
        assert!(matches!(
            events.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));

        let settled = crate::batch_attempt::recover_batches(directory.path(), &gate, db)
            .await
            .unwrap();
        assert!(matches!(
            settled.outcomes.as_slice(),
            [crate::batch_attempt::RecoveredParentOutcome::Terminal {
                state: crate::batch_parent::BatchParentState::Cancelled,
                ..
            }]
        ));
    }

    #[tokio::test]
    async fn restart_mixed_completed_and_running_rolls_back_every_private_artifact() {
        let directory = tempfile::tempdir().unwrap();
        let (mut attempt, _, filenames) = recoverable_attempt(directory.path(), 2, true);
        let completed = attempt.grant(0).unwrap().0;
        attempt
            .stage_and_accept(&completed, b"private-one")
            .unwrap();
        let _lost_running = attempt.grant(1).unwrap().0;
        drop(attempt);
        let gate = crate::batch_transaction::GalleryPublicationGate::default();
        let db = std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let recovered = crate::batch_attempt::recover_batches(directory.path(), &gate, db.clone())
            .await
            .unwrap();
        assert_eq!(recovered.leases_requeued, 1);

        let state = AppState::for_tests();
        let report = resume_recovered_batches(&state, directory.path(), &recovered.outcomes)
            .await
            .unwrap();
        assert_eq!(report.rolled_back, 1);
        assert_eq!(db.as_ref().as_ref().unwrap().count().unwrap(), 0);
        assert!(filenames
            .iter()
            .all(|filename| !directory.path().join(filename).exists()));
        assert!(!directory
            .path()
            .join(crate::batch_transaction::TRANSACTION_DIR)
            .join("restart-parent")
            .join("attempts")
            .join("0")
            .exists());
    }

    #[tokio::test]
    async fn restart_after_all_staged_converges_atomically_before_resume() {
        let directory = tempfile::tempdir().unwrap();
        let (mut attempt, _, filenames) = recoverable_attempt(directory.path(), 2, true);
        for index in 0..2 {
            let lease = attempt.grant(index).unwrap().0;
            attempt
                .stage_and_accept(&lease, format!("private-{index}").as_bytes())
                .unwrap();
        }
        assert_eq!(
            attempt.parent().state(),
            crate::batch_parent::BatchParentState::Prepared
        );
        drop(attempt);
        let gate = crate::batch_transaction::GalleryPublicationGate::default();
        let db = std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let recovered = crate::batch_attempt::recover_batches(directory.path(), &gate, db.clone())
            .await
            .unwrap();
        assert!(matches!(
            recovered.outcomes.as_slice(),
            [crate::batch_attempt::RecoveredParentOutcome::Committed { .. }]
        ));
        assert_eq!(db.as_ref().as_ref().unwrap().count().unwrap(), 2);
        assert!(filenames
            .iter()
            .all(|filename| directory.path().join(filename).is_file()));
    }

    #[tokio::test]
    async fn restart_cancellation_is_terminal_and_receipt_free() {
        let directory = tempfile::tempdir().unwrap();
        let (mut attempt, _, filenames) = recoverable_attempt(directory.path(), 2, true);
        let _active = attempt.grant(0).unwrap().0;
        assert_eq!(
            attempt.request_cancel().unwrap(),
            CompletionDisposition::Accepted
        );
        drop(attempt);
        let gate = crate::batch_transaction::GalleryPublicationGate::default();
        let db = std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let recovered = crate::batch_attempt::recover_batches(directory.path(), &gate, db.clone())
            .await
            .unwrap();
        assert!(matches!(
            recovered.outcomes.as_slice(),
            [crate::batch_attempt::RecoveredParentOutcome::Terminal {
                state: crate::batch_parent::BatchParentState::Cancelled,
                ..
            }]
        ));
        assert_eq!(db.as_ref().as_ref().unwrap().count().unwrap(), 0);
        assert!(filenames
            .iter()
            .all(|filename| !directory.path().join(filename).exists()));
    }
}
