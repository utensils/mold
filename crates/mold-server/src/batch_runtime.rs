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
use futures::stream::{FuturesUnordered, StreamExt as _};
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

/// Maximum cardinality of one live, atomic server-owned batch.
///
/// This is deliberately separate from both the planner's arbitrary-`u32`
/// cardinality support and the reducer's sparse 1,024-child window. A live
/// HTTP parent still owns one durable manifest record, result slot, final
/// filename, and ordered completion entry per output. Capping that O(N)
/// delivery/materialization surface at 64 keeps admission bounded while
/// remaining well above the product's ordinary interactive batch sizes.
pub(crate) const MAX_LIVE_SERVER_BATCH_OUTPUTS: u32 = 64;
pub(crate) const BATCH_OUTPUT_LIMIT_EXCEEDED_CODE: &str = "BATCH_OUTPUT_LIMIT_EXCEEDED";

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("batch_size ({requested}) exceeds the live server batch output limit ({limit})")]
pub(crate) struct LiveBatchAdmissionError {
    pub requested: u32,
    pub limit: u32,
}

pub(crate) fn validate_live_server_batch_size(
    request: &GenerateRequest,
) -> Result<(), LiveBatchAdmissionError> {
    if request.batch_size > MAX_LIVE_SERVER_BATCH_OUTPUTS {
        return Err(LiveBatchAdmissionError {
            requested: request.batch_size,
            limit: MAX_LIVE_SERVER_BATCH_OUTPUTS,
        });
    }
    Ok(())
}

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

pub(crate) enum CompletedServerBatch {
    Json(BatchGenerateResponse),
    Sse(SseBatchCompleteEvent),
}

#[derive(Clone, Copy)]
pub(crate) enum ServerBatchDelivery {
    Json,
    Sse(SseCompletionPayload),
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

fn spawn_owned_supervisor<T, F>(future: F) -> tokio::sync::oneshot::Receiver<T>
where
    T: Send + 'static,
    F: std::future::Future<Output = T> + Send + 'static,
{
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let result = future.await;
        let _ = result_tx.send(result);
    });
    result_rx
}

/// Own the durable parent independently from the HTTP response future.
/// Disconnecting either a JSON or SSE client must never drop live leases or
/// strand an attempt until restart.
pub(crate) fn spawn_server_batch(
    state: AppState,
    authority: RegisteredServerBatch,
    request: GenerateRequest,
    output_dir: PathBuf,
    delivery: ServerBatchDelivery,
) -> tokio::sync::oneshot::Receiver<anyhow::Result<CompletedServerBatch>> {
    spawn_owned_supervisor(async move {
        execute_server_batch(&state, authority, request, output_dir, delivery).await
    })
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
        crate::variant_dependencies::DependencyPreparationContext::default(),
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
    let timing_by_edge = state
        .scheduled_work
        .batch_device_profiles(
            execution_request,
            request.batch_size,
            prepared_inputs.clone(),
        )
        .await
        .map_err(anyhow::Error::msg)?
        .into_iter()
        .map(|candidate| {
            (
                (
                    candidate.device_id.clone(),
                    candidate
                        .execution_equivalence_fingerprint
                        .clone()
                        .unwrap_or_default(),
                ),
                candidate,
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

fn normalized_child(
    parent_id: &str,
    request: &GenerateRequest,
    base_seed: u64,
    index: u32,
) -> GenerateRequest {
    let mut child = request.clone();
    child.batch_size = 1;
    child.seed = Some(base_seed.wrapping_add(u64::from(index)));
    child.batch_id = Some(parent_id.to_string());
    child.batch_index = Some(index + 1);
    child.batch_count = Some(request.batch_size);
    child
}

#[cfg(test)]
fn normalize_children(
    parent_id: &str,
    request: &GenerateRequest,
    base_seed: u64,
) -> Vec<GenerateRequest> {
    (0..request.batch_size)
        .map(|index| normalized_child(parent_id, request, base_seed, index))
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
    base_seed: u64,
    metadata_template: &OutputMetadata,
) -> Vec<GenerationRecord> {
    let timestamp = mold_core::time::now_epoch_ms_u64();
    let format = parent.resolved_output_format();
    let extension = format.extension();
    (0..parent.batch_size)
        .map(|index| {
            let filename = mold_core::default_output_filename(
                &parent.model,
                timestamp,
                extension,
                parent.batch_size,
                index,
            );
            let metadata = child_metadata(
                metadata_template,
                index,
                base_seed.wrapping_add(index.into()),
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

fn batch_metadata_template(
    parent_id: &str,
    parent: &GenerateRequest,
    base_seed: u64,
) -> OutputMetadata {
    let mut metadata = OutputMetadata::from_generate_request(
        parent,
        base_seed,
        parent.scheduler,
        mold_core::build_info::version_string(),
    );
    metadata.batch_id = Some(parent_id.to_string());
    metadata.batch_count = Some(parent.batch_size);
    metadata
}

fn child_metadata(template: &OutputMetadata, child_index: u32, seed: u64) -> OutputMetadata {
    let mut metadata = template.clone();
    metadata.batch_index = Some(child_index.saturating_add(1));
    metadata.seed = seed;
    metadata
}

fn media_bytes(result: &GenerationJobResult) -> &[u8] {
    // Audio is probed first because an audio print's `result.image` is the
    // waveform tile the queue synthesizes so the SSE and gallery pipelines
    // have a raster to lay out. `batch_records` already named this child
    // `.wav` from the request's resolved format, so a video-or-image probe
    // commits PNG bytes under a WAV filename.
    if let Some(audio) = result.response.audio.as_ref() {
        return audio.data.as_slice();
    }
    result
        .response
        .video
        .as_ref()
        .map_or(result.image.data.as_slice(), |video| video.data.as_slice())
}

#[cfg(test)]
fn committed_video_preview<'a>(
    result: &'a GenerationJobResult,
    filename: &'a str,
) -> Option<(&'a str, &'a [u8])> {
    result
        .response
        .video
        .as_ref()
        .filter(|video| !video.gif_preview.is_empty())
        .map(|video| (filename, video.gif_preview.as_slice()))
}

fn completed_record(
    output_dir: &Path,
    filename: &str,
    metadata_template: &OutputMetadata,
    child_index: u32,
    result: &GenerationJobResult,
) -> GenerationRecord {
    let mut metadata = child_metadata(metadata_template, child_index, result.response.seed_used);
    let format = if let Some(audio) = &result.response.audio {
        // Audio has no dimensions of its own; record the waveform tile's, so
        // the gallery grid lays the row out with a real aspect ratio instead
        // of the request's (meaningless) video shape. Frames/fps stay unset —
        // a WAV has neither, and `isVideoItem` keys off `video_frames`.
        metadata.apply_output_dimensions(audio.thumbnail_width, audio.thumbnail_height);
        audio.format
    } else if let Some(video) = &result.response.video {
        metadata.apply_video_output(video);
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

struct GrantedBatchChild {
    lease: BatchChildLease,
    cancellation: mold_inference::InferenceCancellationToken,
    admission_cancellation: tokio_util::sync::CancellationToken,
}

struct ChildSubmission {
    request: GenerateRequest,
    metadata: OutputMetadata,
    granted: GrantedBatchChild,
    ordinal: usize,
    retry: u8,
}

struct AwaitingBatchChild {
    job_id: String,
    index: usize,
    lease: BatchChildLease,
    retry: u8,
    receiver: tokio::sync::oneshot::Receiver<Result<GenerationJobResult, String>>,
    queued_cancel: std::sync::Arc<tokio::sync::Notify>,
}

struct AwaitedBatchChild {
    job_id: String,
    index: usize,
    lease: BatchChildLease,
    retry: u8,
    result: Result<GenerationJobResult, String>,
}

async fn await_batch_child(mut child: AwaitingBatchChild) -> AwaitedBatchChild {
    let result = tokio::select! {
        result = &mut child.receiver => result.unwrap_or_else(|_| {
            Err("batch child worker dropped its result".to_string())
        }),
        _ = child.queued_cancel.notified() => {
            Err("batch child cancelled before worker execution".to_string())
        }
    };
    AwaitedBatchChild {
        job_id: child.job_id,
        index: child.index,
        lease: child.lease,
        retry: child.retry,
        result,
    }
}

#[derive(Debug)]
struct CompactBatchResult {
    response: mold_core::GenerateResponse,
    image: mold_core::ImageData,
}

fn stage_batch_result_auxiliaries(
    attempt: &mut DurableBatchAttempt,
    lease: &BatchChildLease,
    result: &GenerationJobResult,
) -> anyhow::Result<()> {
    if let Some(audio) = result.response.audio.as_ref() {
        // The waveform tile is the only raster an audio print has, and it
        // commits to the same `cache/thumbnails/<final_name>.png` the single
        // -job path writes. There is no animated preview for a WAV.
        attempt.stage_video_auxiliaries(lease, &audio.thumbnail, &[])?;
        return Ok(());
    }
    if let Some(video) = result.response.video.as_ref() {
        attempt.stage_video_auxiliaries(lease, &video.thumbnail, &video.gif_preview)?;
    }
    Ok(())
}

fn compact_batch_result(mut result: GenerationJobResult) -> CompactBatchResult {
    if let Some(audio) = result.response.audio.as_mut() {
        // Compaction exists so a wide batch does not hold every child's media
        // in memory at once. A WAV is the largest thing an audio child owns,
        // so leaving it resident would defeat the whole mechanism.
        audio.data.clear();
        audio.thumbnail.clear();
    }
    if let Some(video) = result.response.video.as_mut() {
        video.data.clear();
        video.thumbnail.clear();
        video.gif_preview.clear();
    } else {
        for image in &mut result.response.images {
            image.data.clear();
        }
    }
    result.image.data.clear();
    CompactBatchResult {
        response: result.response,
        image: result.image,
    }
}

fn video_thumbnail_path(filename: &str) -> PathBuf {
    mold_core::Config::mold_dir()
        .unwrap_or_else(|| PathBuf::from(".mold"))
        .join("cache")
        .join("thumbnails")
        .join(format!("{filename}.png"))
}

fn video_preview_path(filename: &str) -> PathBuf {
    mold_core::Config::mold_dir()
        .unwrap_or_else(|| PathBuf::from(".mold"))
        .join("cache")
        .join("previews")
        .join(mold_core::media_paths::preview_gif_filename(filename))
}

fn hydrate_batch_result(
    output_dir: &Path,
    filename: &str,
    mut compact: CompactBatchResult,
    include_media: bool,
) -> anyhow::Result<GenerationJobResult> {
    if !include_media {
        return Ok(GenerationJobResult {
            response: compact.response,
            image: compact.image,
        });
    }
    let media = std::fs::read(output_dir.join(filename))
        .with_context(|| format!("reading committed batch output {filename}"))?;
    if let Some(audio) = compact.response.audio.as_mut() {
        audio.data = media;
        audio.thumbnail = std::fs::read(video_thumbnail_path(filename)).unwrap_or_default();
        // The raster slot keeps carrying the waveform tile, exactly as the
        // single-job path leaves it: audio has no image of its own, and every
        // consumer downstream reads `image` as "something to lay out".
        compact.image.data = audio.thumbnail.clone();
        return Ok(GenerationJobResult {
            response: compact.response,
            image: compact.image,
        });
    }
    if let Some(video) = compact.response.video.as_mut() {
        video.data = media;
        video.thumbnail = std::fs::read(video_thumbnail_path(filename)).unwrap_or_default();
        video.gif_preview = std::fs::read(video_preview_path(filename)).unwrap_or_default();
        compact.image.data = video.thumbnail.clone();
    } else {
        ensure!(
            compact.response.images.len() <= 1,
            "singleton batch child returned {} images",
            compact.response.images.len()
        );
        compact.image.data = media.clone();
        if compact.response.images.is_empty() {
            compact.response.images.push(compact.image.clone());
        } else if let Some(image) = compact.response.images.first_mut() {
            image.data = media;
        }
    }
    Ok(GenerationJobResult {
        response: compact.response,
        image: compact.image,
    })
}

impl GrantedBatchChild {
    fn new(
        (lease, cancellation): (BatchChildLease, mold_inference::InferenceCancellationToken),
        admission_cancellation: tokio_util::sync::CancellationToken,
    ) -> Self {
        Self {
            lease,
            cancellation,
            admission_cancellation,
        }
    }
}

async fn submit_child(
    state: &AppState,
    parent_id: &str,
    plan: &FrozenBatchPlan,
    submission: ChildSubmission,
) -> anyhow::Result<(
    String,
    tokio::sync::oneshot::Receiver<Result<GenerationJobResult, String>>,
    std::sync::Arc<tokio::sync::Notify>,
)> {
    let ChildSubmission {
        request,
        metadata,
        granted,
        ordinal,
        retry,
    } = submission;
    let id = if granted.lease.child_index == 0 && retry == 0 {
        parent_id.to_string()
    } else {
        format!(
            "{parent_id}:child:{}:try:{retry}",
            granted.lease.child_index + 1
        )
    };
    let queued_cancel = state.job_registry.register_job(
        &id,
        &request.model,
        Some(ordinal),
        Some(true),
        Some(Box::new(metadata)),
    );
    if !state.job_registry.register_batch_child(parent_id, &id) {
        state.job_registry.remove(&id);
        anyhow::bail!("batch parent cancellation authority is no longer open");
    }
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let admission_cancellation = granted.admission_cancellation.clone();
    let job = GenerationJob {
        id: id.clone(),
        request,
        resolved_references: None,
        completion_payload: SseCompletionPayload::Full,
        progress_tx: None,
        result_tx,
        output_dir: None,
        batch_child: Some(BatchChildExecution {
            lease: granted.lease,
            cancellation: granted.cancellation,
            execution_equivalence_fingerprint: plan.equivalence.clone(),
            prepared_inputs: plan.prepared_inputs.clone(),
        }),
        // Batch children are recovered by the batch transaction's own durable
        // manifest, never by the singleton journal.
        journal: None,
        #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
        h3_private_ingress_grant: None,
    };
    state
        .queue
        .submit_when_available(
            &mut Some(job),
            state.queue_capacity,
            &admission_cancellation,
        )
        .await
        .inspect_err(|_| state.job_registry.remove(&id))
        .map_err(|error| match error {
            SubmitError::Full { pending, capacity } => {
                anyhow::anyhow!("generation queue is full ({pending}/{capacity}); retry later")
            }
            SubmitError::Cancelled => anyhow::anyhow!("batch parent cancelled before admission"),
            SubmitError::Shutdown => anyhow::anyhow!("generation queue is shutting down"),
        })?;
    Ok((id, result_rx, queued_cancel))
}

struct BatchSubmissionContext<'a> {
    state: &'a AppState,
    parent_id: &'a str,
    parent_request: &'a GenerateRequest,
    base_seed: u64,
    metadata_template: &'a OutputMetadata,
    plan: &'a FrozenBatchPlan,
    parent_cancel: tokio_util::sync::CancellationToken,
}

async fn grant_and_submit_child(
    context: &BatchSubmissionContext<'_>,
    attempt: &mut DurableBatchAttempt,
    index: usize,
    retry: u8,
) -> anyhow::Result<AwaitingBatchChild> {
    let child_index = u32::try_from(index).context("batch child index exceeds u32")?;
    let partition = context.plan.adaptive.partition_at(child_index)?;
    ensure!(
        partition.child_start == child_index && partition.size == 1,
        "singleton production batch planner returned a non-singleton partition"
    );
    let ordinal = *context
        .plan
        .ordinal_by_device
        .get(partition.device_id.as_str())
        .context("adaptive batch plan references an unknown device")?;
    ensure!(
        !context.parent_cancel.is_cancelled(),
        "batch parent cancelled before child grant"
    );
    let granted = GrantedBatchChild::new(attempt.grant(index)?, context.parent_cancel.clone());
    let lease = granted.lease.clone();
    let child = normalized_child(
        context.parent_id,
        context.parent_request,
        context.base_seed,
        child_index,
    );
    let metadata = child_metadata(
        context.metadata_template,
        child_index,
        context.base_seed.wrapping_add(child_index.into()),
    );
    let (job_id, receiver, queued_cancel) = match submit_child(
        context.state,
        context.parent_id,
        context.plan,
        ChildSubmission {
            request: child,
            metadata,
            granted,
            ordinal,
            retry,
        },
    )
    .await
    {
        Ok(submitted) => submitted,
        Err(error) => {
            let _ = attempt.complete_without_artifact(&lease, ChildCompletion::Cancelled)?;
            return Err(error);
        }
    };
    Ok(AwaitingBatchChild {
        job_id,
        index,
        lease,
        retry,
        receiver,
        queued_cancel,
    })
}

fn rolling_batch_window_for(child_count: usize, schedulable_lanes: usize) -> anyhow::Result<usize> {
    ensure!(schedulable_lanes > 0, "no schedulable batch lanes");
    Ok(child_count
        .min(schedulable_lanes.saturating_mul(2))
        .clamp(1, crate::batch_parent::MAX_MATERIALIZED_CHILDREN))
}

/// Execute a normalized server-owned parent. No child writes to the gallery.
pub(crate) async fn execute_server_batch(
    state: &AppState,
    authority: RegisteredServerBatch,
    mut request: GenerateRequest,
    output_dir: PathBuf,
    delivery: ServerBatchDelivery,
) -> anyhow::Result<CompletedServerBatch> {
    validate_live_server_batch_size(&request)?;
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
    let child_count =
        usize::try_from(request.batch_size).context("batch size exceeds platform usize")?;
    let estimated_bytes = conservative_batch_output_bytes(&request)?;
    crate::batch_transaction::preflight_disk_space(&output_dir, estimated_bytes)?;
    let metadata_template = batch_metadata_template(&parent_id, &request, base_seed);
    let records = batch_records(&output_dir, &request, base_seed, &metadata_template);
    let normalized = serde_json::to_value(recovery_envelope(&request, &plan))?;
    let mut attempt = DurableBatchAttempt::begin(&output_dir, &parent_id, normalized, records)?;
    attempt.start()?;

    let window = rolling_batch_window_for(child_count, plan.adaptive.devices_used as usize)?;
    let submission = BatchSubmissionContext {
        state,
        parent_id: &parent_id,
        parent_request: &request,
        base_seed,
        metadata_template: &metadata_template,
        plan: &plan,
        parent_cancel: parent_cancel.clone(),
    };
    let mut active = FuturesUnordered::new();
    let mut next_index = 0_usize;
    let mut terminal_error = None;
    while next_index < child_count && active.len() < window {
        match grant_and_submit_child(&submission, &mut attempt, next_index, 0).await {
            Ok(child) => {
                active.push(await_batch_child(child));
                next_index += 1;
            }
            Err(error) => {
                terminal_error = Some(error);
                let _ = state.job_registry.cancel_queued(&parent_id);
                break;
            }
        }
    }
    let mut results = (0..child_count)
        .map(|_| None)
        .collect::<Vec<Option<CompactBatchResult>>>();
    let mut parent_cancel_reduced = false;
    while !active.is_empty() {
        let completed = tokio::select! {
            _ = parent_cancel.cancelled(), if !parent_cancel_reduced => {
                let _ = attempt.request_cancel()?;
                let _ = state.job_registry.cancel_queued(&parent_id);
                parent_cancel_reduced = true;
                terminal_error.get_or_insert_with(|| anyhow::anyhow!("batch parent cancelled"));
                continue;
            }
            completed = active.next() => completed.expect("non-empty batch future set ended"),
        };
        let AwaitedBatchChild {
            job_id,
            index,
            lease,
            retry,
            result,
        } = completed;
        state
            .job_registry
            .unregister_batch_child(&parent_id, &job_id);
        match result {
            Ok(result) if terminal_error.is_none() => {
                let filename = attempt.transaction().manifest().children[index]
                    .final_name
                    .clone();
                let record = completed_record(
                    &output_dir,
                    &filename,
                    &metadata_template,
                    index as u32,
                    &result,
                );
                stage_batch_result_auxiliaries(&mut attempt, &lease, &result)?;
                let disposition =
                    attempt.stage_record_and_accept(&lease, record, media_bytes(&result))?;
                ensure!(
                    matches!(
                        disposition,
                        CompletionDisposition::Accepted | CompletionDisposition::AttemptPrepared
                    ),
                    "live batch child completion lost parent authority: {disposition:?}"
                );
                results[index] = Some(compact_batch_result(result));
                if next_index < child_count {
                    match grant_and_submit_child(&submission, &mut attempt, next_index, 0).await {
                        Ok(child) => {
                            active.push(await_batch_child(child));
                            next_index += 1;
                        }
                        Err(error) => {
                            terminal_error = Some(error);
                            let _ = state.job_registry.cancel_queued(&parent_id);
                        }
                    }
                }
            }
            Ok(_) => {
                let _ = attempt.complete_without_artifact(&lease, ChildCompletion::Cancelled)?;
            }
            Err(error) => {
                let completion = if error.contains("cancelled") {
                    ChildCompletion::Cancelled
                } else {
                    ChildCompletion::Failed
                };
                let disposition = attempt.complete_without_artifact(&lease, completion)?;
                if disposition == CompletionDisposition::RetryChild && terminal_error.is_none() {
                    match grant_and_submit_child(
                        &submission,
                        &mut attempt,
                        index,
                        retry.saturating_add(1),
                    )
                    .await
                    {
                        Ok(child) => {
                            active.push(await_batch_child(child));
                            continue;
                        }
                        Err(submit_error) => terminal_error = Some(submit_error),
                    }
                }
                if terminal_error.is_none() {
                    terminal_error =
                        Some(anyhow::anyhow!("batch child {} failed: {error}", index + 1));
                }
                let _ = state.job_registry.cancel_queued(&parent_id);
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
        results.iter().all(Option::is_some),
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
    let mut outputs =
        matches!(delivery, ServerBatchDelivery::Json).then(|| Vec::with_capacity(results.len()));
    let mut events =
        matches!(delivery, ServerBatchDelivery::Sse(_)).then(|| Vec::with_capacity(results.len()));
    for (index, (compact, filename)) in results.into_iter().zip(filenames).enumerate() {
        let completion_payload = match delivery {
            ServerBatchDelivery::Json | ServerBatchDelivery::Sse(SseCompletionPayload::Full) => {
                SseCompletionPayload::Full
            }
            ServerBatchDelivery::Sse(SseCompletionPayload::MetadataOnly) => {
                SseCompletionPayload::MetadataOnly
            }
        };
        let result = hydrate_batch_result(
            &output_dir,
            &filename,
            compact.context("committed batch child has no compact result")?,
            completion_payload == SseCompletionPayload::Full,
        )?;
        if let Some(events) = events.as_mut() {
            let mut metadata =
                child_metadata(&metadata_template, index as u32, result.response.seed_used);
            if let Some(video) = result.response.video.as_ref() {
                metadata.apply_video_output(video);
            }
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
        }
        state.events.publish(mold_core::ServerEvent::GalleryAdded {
            filename: filename.clone(),
            image: None,
        });
        if let Some(outputs) = outputs.as_mut() {
            outputs.push(BatchGenerateOutput {
                batch_index: index as u32 + 1,
                filename,
                response: result.response,
            });
        }
    }
    Ok(match (outputs, events) {
        (Some(outputs), None) => CompletedServerBatch::Json(BatchGenerateResponse {
            batch_id: parent_id.clone(),
            outputs,
        }),
        (None, Some(outputs)) => CompletedServerBatch::Sse(SseBatchCompleteEvent {
            batch_id: parent_id,
            outputs,
        }),
        _ => unreachable!("server batch delivery has exactly one wire response"),
    })
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LiveBatchResumeReport {
    pub resumed: usize,
    pub committed: usize,
    pub rolled_back: usize,
}

fn next_pending_child(attempt: &DurableBatchAttempt) -> anyhow::Result<Option<usize>> {
    Ok(attempt
        .parent()
        .pending_window(0, 1)?
        .indices
        .into_iter()
        .next())
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
    if let Err(error) = validate_live_server_batch_size(&envelope.request) {
        tracing::error!(
            %parent_id,
            error = %error,
            "rolling back recovered live batch above the current delivery limit"
        );
        terminally_cancel_recovered_attempt(&mut attempt)?;
        return Ok(false);
    }
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
    let metadata_template = batch_metadata_template(parent_id, &request, base_seed);
    let child_count = attempt.parent().total_children();
    let window = rolling_batch_window_for(child_count, plan.adaptive.devices_used as usize)?;
    let submission = BatchSubmissionContext {
        state,
        parent_id,
        parent_request: &request,
        base_seed,
        metadata_template: &metadata_template,
        plan: &plan,
        parent_cancel: parent_cancel.clone(),
    };
    let mut active = FuturesUnordered::new();
    let mut terminal_error = None;
    while active.len() < window {
        let Some(index) = next_pending_child(&attempt)? else {
            break;
        };
        match grant_and_submit_child(&submission, &mut attempt, index, 0).await {
            Ok(child) => {
                active.push(await_batch_child(child));
            }
            Err(error) => {
                terminal_error = Some(error);
                let _ = state.job_registry.cancel_queued(parent_id);
                break;
            }
        }
    }

    let mut parent_cancel_reduced = false;
    while !active.is_empty() {
        let completed = tokio::select! {
            _ = parent_cancel.cancelled(), if !parent_cancel_reduced => {
                let _ = attempt.request_cancel()?;
                let _ = state.job_registry.cancel_queued(parent_id);
                parent_cancel_reduced = true;
                terminal_error.get_or_insert_with(|| {
                    anyhow::anyhow!("recovered batch parent cancelled")
                });
                continue;
            }
            completed = active.next() => completed.expect("non-empty recovered batch future set ended"),
        };
        let AwaitedBatchChild {
            job_id,
            index,
            lease,
            retry,
            result,
        } = completed;
        state
            .job_registry
            .unregister_batch_child(parent_id, &job_id);
        match result {
            Ok(result) if terminal_error.is_none() => {
                let filename = attempt.transaction().manifest().children[index]
                    .final_name
                    .clone();
                let record = completed_record(
                    output_dir,
                    &filename,
                    &metadata_template,
                    index as u32,
                    &result,
                );
                stage_batch_result_auxiliaries(&mut attempt, &lease, &result)?;
                let disposition =
                    attempt.stage_record_and_accept(&lease, record, media_bytes(&result))?;
                ensure!(
                    matches!(
                        disposition,
                        CompletionDisposition::Accepted | CompletionDisposition::AttemptPrepared
                    ),
                    "recovered batch child completion lost parent authority: {disposition:?}"
                );
                drop(compact_batch_result(result));
                if let Some(next_index) = next_pending_child(&attempt)? {
                    match grant_and_submit_child(&submission, &mut attempt, next_index, 0).await {
                        Ok(child) => {
                            active.push(await_batch_child(child));
                        }
                        Err(error) => {
                            terminal_error = Some(error);
                            let _ = state.job_registry.cancel_queued(parent_id);
                        }
                    }
                }
            }
            Ok(_) => {
                let _ = attempt.complete_without_artifact(&lease, ChildCompletion::Cancelled)?;
            }
            Err(error) => {
                let completion = if error.contains("cancelled") {
                    ChildCompletion::Cancelled
                } else {
                    ChildCompletion::Failed
                };
                let disposition = attempt.complete_without_artifact(&lease, completion)?;
                if disposition == CompletionDisposition::RetryChild && terminal_error.is_none() {
                    match grant_and_submit_child(
                        &submission,
                        &mut attempt,
                        index,
                        retry.saturating_add(1),
                    )
                    .await
                    {
                        Ok(child) => {
                            active.push(await_batch_child(child));
                            continue;
                        }
                        Err(submit_error) => terminal_error = Some(submit_error),
                    }
                }
                if terminal_error.is_none() {
                    terminal_error = Some(anyhow::anyhow!(
                        "recovered batch child {} failed: {error}",
                        index + 1
                    ));
                }
                let _ = state.job_registry.cancel_queued(parent_id);
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

    #[test]
    fn metadata_template_matches_independent_child_metadata_without_rehashing_source() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "batch",
            "model": "flux",
            "width": 64,
            "height": 64,
            "steps": 1,
            "batch_size": 3,
            "seed": 9
        }))
        .unwrap();
        request.source_image = Some(vec![1, 2, 3, 4]);
        request.source_image_name = Some("source.png".to_string());
        let template = batch_metadata_template("parent", &request, 9);
        let child = normalized_child("parent", &request, 9, 2);
        let expected = OutputMetadata::from_generate_request(
            &child,
            11,
            child.scheduler,
            mold_core::build_info::version_string(),
        );
        assert_eq!(child_metadata(&template, 2, 11), expected);
    }

    #[test]
    fn huge_parent_materializes_only_a_bounded_eligible_lane_window() {
        assert_eq!(rolling_batch_window_for(1_000_000, 8).unwrap(), 16);
        assert_eq!(rolling_batch_window_for(1, 8).unwrap(), 1);
        // Shared queue capacity is intentionally not part of this bound:
        // waitable admission drains as lanes dequeue work, so capacity 1 or 4
        // must not strand the other eligible GPUs.
        for _queue_capacity in [1, 4] {
            assert_eq!(rolling_batch_window_for(1_000_000, 8).unwrap(), 16);
        }
    }

    #[tokio::test]
    async fn parent_cancellation_waits_for_running_worker_result_ack() {
        let directory = tempfile::tempdir().unwrap();
        let (mut attempt, _) = durable_attempt(directory.path());
        let (lease, _) = attempt.grant(0).unwrap();
        attempt.request_cancel().unwrap();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let waiter = tokio::spawn(await_batch_child(AwaitingBatchChild {
            job_id: "running-child".to_string(),
            index: 0,
            lease,
            retry: 0,
            receiver: result_rx,
            queued_cancel: std::sync::Arc::new(tokio::sync::Notify::new()),
        }));
        tokio::task::yield_now().await;
        assert!(
            !waiter.is_finished(),
            "parent cancellation is a signal, not a synthetic worker acknowledgement"
        );
        assert!(result_tx
            .send(Err("worker observed cancellation".to_string()))
            .is_ok());
        let completed = waiter.await.unwrap();
        match completed.result {
            Err(error) => assert_eq!(error, "worker observed cancellation"),
            Ok(_) => panic!("cancelled worker unexpectedly succeeded"),
        }
    }

    #[tokio::test]
    async fn dropping_http_waiter_does_not_drop_owned_batch_supervisor() {
        let started = std::sync::Arc::new(tokio::sync::Notify::new());
        let release = std::sync::Arc::new(tokio::sync::Notify::new());
        let finished = std::sync::Arc::new(tokio::sync::Notify::new());
        let result_rx = spawn_owned_supervisor({
            let started = started.clone();
            let release = release.clone();
            let finished = finished.clone();
            async move {
                started.notify_one();
                release.notified().await;
                finished.notify_one();
            }
        });
        started.notified().await;
        drop(result_rx);
        release.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(1), finished.notified())
            .await
            .expect("owned batch supervisor must survive response waiter disconnect");
    }

    #[test]
    fn committed_video_batch_retains_preview_under_final_gallery_name() {
        let result = GenerationJobResult {
            response: mold_core::GenerateResponse {
                audio: None,
                images: Vec::new(),
                video: Some(mold_core::VideoData {
                    data: b"mp4".to_vec(),
                    format: mold_core::OutputFormat::Mp4,
                    width: 64,
                    height: 64,
                    frames: 9,
                    fps: 24,
                    pipeline: None,
                    pipeline_provenance_sha256: None,
                    source_preprocessing: None,
                    thumbnail: b"png".to_vec(),
                    gif_preview: b"GIF89a".to_vec(),
                    has_audio: false,
                    duration_ms: None,
                    audio_sample_rate: None,
                    audio_channels: None,
                }),
                generation_time_ms: 1,
                model: "ltx-video:q8".to_string(),
                seed_used: 7,
                gpu: Some(1),
            },
            image: mold_core::ImageData {
                data: b"png".to_vec(),
                format: mold_core::OutputFormat::Png,
                width: 64,
                height: 64,
                index: 0,
            },
        };

        let (filename, bytes) =
            committed_video_preview(&result, "ordered-child-2.mp4").expect("video preview");
        assert_eq!(filename, "ordered-child-2.mp4");
        assert_eq!(bytes, b"GIF89a");
    }

    fn audio_job_result(wav: &[u8], waveform: &[u8]) -> GenerationJobResult {
        GenerationJobResult {
            response: mold_core::GenerateResponse {
                audio: Some(mold_core::AudioData {
                    data: wav.to_vec(),
                    format: mold_core::OutputFormat::Wav,
                    sample_rate: 24_000,
                    channels: 2,
                    duration_ms: 5_010,
                    thumbnail: waveform.to_vec(),
                    thumbnail_width: 640,
                    thumbnail_height: 360,
                }),
                images: Vec::new(),
                video: None,
                generation_time_ms: 1,
                model: "ltx-2-19b-dev:fp8".to_string(),
                seed_used: 7,
                gpu: Some(0),
            },
            // What the queue synthesizes so the SSE and gallery pipelines have
            // a raster: the waveform tile, never the artifact.
            image: mold_core::ImageData {
                data: waveform.to_vec(),
                format: mold_core::OutputFormat::Png,
                width: 640,
                height: 360,
                index: 0,
            },
        }
    }

    /// `batch_records` names an audio child `.wav` from the request's resolved
    /// format. Staging `result.image` therefore committed the waveform PNG
    /// under that name — a corrupt gallery artifact, recorded as an image.
    #[test]
    fn server_owned_audio_batch_commits_the_wav_not_its_waveform() {
        let result = audio_job_result(b"RIFF....WAVEfmt ", b"\x89PNG");
        assert_eq!(
            media_bytes(&result),
            b"RIFF....WAVEfmt ",
            "the committed bytes must be the artifact the child rendered",
        );

        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "heavy rain on a tin roof",
            "model": "ltx-2-19b-dev:fp8",
            "width": 960,
            "height": 576,
            "steps": 40,
            "batch_size": 2,
            "seed": 7,
            "frames": 121,
            "fps": 24,
            "pipeline": "t2a",
            "output_format": "wav"
        }))
        .unwrap();
        let template = batch_metadata_template("parent", &request, 7);
        let record = completed_record(
            Path::new("/gallery"),
            "ltx-2-19b-dev-1700000000-0.wav",
            &template,
            0,
            &result,
        );
        assert_eq!(
            record.format,
            mold_core::OutputFormat::Wav,
            "the gallery row must agree with the filename and the bytes",
        );
        assert_eq!(
            (record.metadata.width, record.metadata.height),
            (640, 360),
            "audio has no dimensions; the waveform tile's are what the grid lays out",
        );
        // `frames` stays: for t2a it is the duration the user asked for, not a
        // render shape, and the single-job path records it too. Nothing reads
        // it as "this is a video" — that is `video_frames`, a separate field.
        assert_eq!(record.metadata.frames, Some(121));
    }

    /// Compaction exists so a wide batch does not hold every child's media
    /// resident. A WAV is the largest thing an audio child owns.
    #[test]
    fn compacting_an_audio_batch_child_releases_its_wav() {
        let compact = compact_batch_result(audio_job_result(b"RIFF-large", b"\x89PNG"));
        let audio = compact
            .response
            .audio
            .expect("the slot itself must survive");
        assert!(audio.data.is_empty(), "the WAV must not stay resident");
        assert!(audio.thumbnail.is_empty(), "nor its waveform tile");
        assert_eq!(
            audio.sample_rate, 24_000,
            "the metadata is what compaction keeps"
        );
    }

    #[test]
    fn hydrating_an_audio_batch_child_reads_the_committed_wav_back() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("child.wav"), b"RIFF-committed").unwrap();
        let compact = compact_batch_result(audio_job_result(b"RIFF-committed", b"\x89PNG"));

        let hydrated = hydrate_batch_result(directory.path(), "child.wav", compact, true).unwrap();
        let audio = hydrated
            .response
            .audio
            .expect("audio must survive hydration");
        assert_eq!(audio.data, b"RIFF-committed".to_vec());
        assert!(
            hydrated.response.images.is_empty(),
            "an audio child must never be hydrated into the image list",
        );
    }

    #[test]
    fn json_batch_hydrates_primary_still_when_worker_response_images_are_empty() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("child.png"), b"committed-image").unwrap();
        let compact = CompactBatchResult {
            response: mold_core::GenerateResponse {
                audio: None,
                images: Vec::new(),
                video: None,
                generation_time_ms: 1,
                model: "flux-dev:q8".to_string(),
                seed_used: 7,
                gpu: Some(0),
            },
            image: mold_core::ImageData {
                data: Vec::new(),
                format: mold_core::OutputFormat::Png,
                width: 64,
                height: 64,
                index: 0,
            },
        };

        let hydrated = hydrate_batch_result(directory.path(), "child.png", compact, true).unwrap();
        assert_eq!(hydrated.image.data, b"committed-image");
        assert_eq!(hydrated.response.images.len(), 1);
        assert_eq!(hydrated.response.images[0].data, b"committed-image");
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
        let template = batch_metadata_template("parent", &request, 9);
        let records = batch_records(directory, &request, 9, &template);
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
        let template = batch_metadata_template("restart-parent", &request, 9);
        let records = batch_records(directory, &request, 9, &template);
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
        let (first, cancellation) = attempt.grant(0).unwrap();
        let child_execution = BatchChildExecution {
            lease: first.clone(),
            cancellation,
            execution_equivalence_fingerprint: "same".to_string(),
            prepared_inputs: PreparedExecutionInputs {
                authority_fingerprint: "prepared".to_string(),
                by_device: BTreeMap::new(),
                retryable_device_failures: BTreeMap::new(),
                model_config_overlay: None,
                #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                h3_private_ingress_grant: None,
                #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                h3_private_admission_by_device: BTreeMap::new(),
            },
        };
        let last = attempt.grant(1).unwrap().0;
        assert_eq!(
            attempt.request_cancel().unwrap(),
            CompletionDisposition::Accepted
        );
        assert!(
            child_execution.cancellation.is_cancelled(),
            "the exact durable child token carried into GPU work must observe parent cancellation"
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
    fn live_batch_output_admission_has_an_inclusive_boundary() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "bounded parent",
            "model": "flux-dev:q8",
            "width": 64,
            "height": 64,
            "steps": 1,
            "batch_size": 2,
            "output_format": "png"
        }))
        .unwrap();

        request.batch_size = MAX_LIVE_SERVER_BATCH_OUTPUTS;
        assert_eq!(validate_live_server_batch_size(&request), Ok(()));

        request.batch_size = MAX_LIVE_SERVER_BATCH_OUTPUTS + 1;
        assert_eq!(
            validate_live_server_batch_size(&request),
            Err(LiveBatchAdmissionError {
                requested: MAX_LIVE_SERVER_BATCH_OUTPUTS + 1,
                limit: MAX_LIVE_SERVER_BATCH_OUTPUTS,
            })
        );

        request.batch_size = 1;
        assert_eq!(validate_live_server_batch_size(&request), Ok(()));
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
