//! Payload-free MiniMax H3 packed-attention benchmark.
//!
//! This binary is available only with `dev-bins,h3-attention-rc`. It reuses
//! the exact SM89 release-candidate authority qualified by
//! `h3_attention_qualification`, allocates only synthetic device tensors, and
//! never constructs an inference engine, resolves a model, or opens an
//! artifact. Execution is deliberately explicit:
//!
//! ```text
//! h3_attention_benchmark --scenario release-124-target --iterations 3
//! h3_attention_benchmark --scenario release-345-target --iterations 1 --allow-long
//! h3_attention_benchmark --plan-only
//! ```

use std::collections::HashSet;
use std::env;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use mold_candle::minimax_h3::{
    execute_h3_attention_with_telemetry, H3AttentionBackend, H3AttentionDType, H3AttentionDevice,
    H3AttentionKernel, H3AttentionModelContract, H3AttentionReleaseCandidateBuild,
    H3AttentionReleaseRequirement, H3AttentionTelemetry, H3AttentionWorkspace,
    H3FrozenAttentionPlan, VisualTemporalGeometry, LATENT_ROWS_PER_SECOND,
};
use mold_core::minimax_h3::{
    self as contract, GenerationReferencePreparedShape, DEFAULT_HEIGHT, DEFAULT_WIDTH, FIXED_FPS,
    MAX_FRAMES, REVIEWED_COMPACT_FRAMES,
};
use mold_core::{GenerationReference, GenerationReferenceAuthority, GenerationReferenceProvenance};
use serde::Serialize;

const TOKEN_REFINER_ROWS: usize = 256;
const TEXT_ROWS: u64 = 256;
const VIDEO_ROW_CELL_PIXELS: u32 = 32;
const DEFAULT_WARMUP_ITERATIONS: usize = 1;
const MAX_WARMUP_ITERATIONS: usize = 3;
const MAX_MEASURED_ITERATIONS: usize = 10;
const MAX_LONG_WARMUP_ITERATIONS: usize = 1;
const MAX_LONG_MEASURED_ITERATIONS: usize = 3;

const USAGE: &str = "MiniMax H3 packed-attention benchmark\n\
\n\
Usage:\n\
  h3_attention_benchmark --list\n\
  h3_attention_benchmark --plan-only [--scenario ID ...]\n\
  h3_attention_benchmark --scenario ID [--scenario ID ...] --iterations N [--warmup N]\n\
                         [--allow-long]\n\
\n\
Execution requires at least one explicit --scenario and --iterations. The\n\
345-frame scenarios additionally require --allow-long and are capped at one\n\
warmup plus three measured iterations. Reports are JSON on stdout; redirect\n\
stdout when a durable report is wanted. No model or media path is accepted.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum ReferenceMix {
    None,
    Image,
    VideoImageAudio,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScenarioSpec {
    id: &'static str,
    target_frames: u32,
    reference_mix: ReferenceMix,
}

const SCENARIOS: &[ScenarioSpec] = &[
    ScenarioSpec {
        id: "release-124-target",
        target_frames: REVIEWED_COMPACT_FRAMES,
        reference_mix: ReferenceMix::None,
    },
    ScenarioSpec {
        id: "release-124-image",
        target_frames: REVIEWED_COMPACT_FRAMES,
        reference_mix: ReferenceMix::Image,
    },
    ScenarioSpec {
        id: "release-124-mixed",
        target_frames: REVIEWED_COMPACT_FRAMES,
        reference_mix: ReferenceMix::VideoImageAudio,
    },
    ScenarioSpec {
        id: "release-345-target",
        target_frames: MAX_FRAMES,
        reference_mix: ReferenceMix::None,
    },
    ScenarioSpec {
        id: "release-345-mixed",
        target_frames: MAX_FRAMES,
        reference_mix: ReferenceMix::VideoImageAudio,
    },
];

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Help,
    List,
    Plan {
        scenario_ids: Vec<String>,
    },
    Execute {
        scenario_ids: Vec<String>,
        warmup_iterations: usize,
        measured_iterations: usize,
        allow_long: bool,
    },
}

#[derive(Debug, Default)]
struct RawArguments {
    list: bool,
    plan_only: bool,
    help: bool,
    scenario_ids: Vec<String>,
    warmup_iterations: Option<usize>,
    measured_iterations: Option<usize>,
    allow_long: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ReferenceSourceFacts {
    Image {
        width: u32,
        height: u32,
    },
    Video {
        width: u32,
        height: u32,
        frame_count: u32,
        duration_ms: u64,
        fps: f64,
        has_audio: bool,
        audio_samples_per_channel: u64,
        audio_sample_rate: u32,
        audio_channels: u16,
    },
    Audio {
        duration_ms: u64,
        sample_rate: u32,
        channels: u16,
        samples_per_channel: u64,
    },
}

#[derive(Clone, Debug, Serialize)]
struct ReferenceFact {
    request_index: usize,
    source: ReferenceSourceFacts,
    prepared: GenerationReferencePreparedShape,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum PackedSegmentKind {
    Text,
    ReferenceVideoAudio,
    ReferenceVideoVisual,
    ReferenceImageVisual,
    ReferenceAudio,
    GeneratedAudio,
    GeneratedVideo,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct PackedSegment {
    kind: PackedSegmentKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_index: Option<usize>,
    rows: u64,
    start_row: u64,
    end_row_exclusive: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct PackedRowComposition {
    text_rows: u64,
    condition_visual_rows: u64,
    condition_audio_rows: u64,
    generated_audio_rows: u64,
    generated_video_rows: u64,
    total_packed_rows: u64,
}

#[derive(Clone, Debug, Serialize)]
struct ScenarioFacts {
    id: &'static str,
    target_frames: u32,
    target_width: u32,
    target_height: u32,
    target_fps: u32,
    target_latent_frames: usize,
    reference_mix: ReferenceMix,
    references: Vec<ReferenceFact>,
    rows: PackedRowComposition,
    packed_segments_in_order: Vec<PackedSegment>,
}

#[derive(Clone, Debug, Serialize)]
struct AttentionTensorBytes {
    query_bytes: u64,
    key_bytes: u64,
    value_bytes: u64,
    output_bytes: u64,
    total_qkv_output_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
struct PlanFacts {
    plan_identity_sha256: String,
    backend: H3AttentionBackend,
    kernel: H3AttentionKernel,
    dtype: H3AttentionDType,
    sequence_rows: usize,
    heads: usize,
    head_dim: usize,
    tensor_bytes: AttentionTensorBytes,
    workspace: H3AttentionWorkspace,
}

impl PlanFacts {
    fn from_plan(plan: &H3FrozenAttentionPlan) -> Result<Self> {
        if plan.dtype() != H3AttentionDType::Bf16 {
            bail!("H3 release benchmark tensor accounting requires BF16");
        }
        let rows = u64::try_from(plan.rows())?;
        let heads = u64::try_from(plan.heads())?;
        let head_dim = u64::try_from(plan.head_dim())?;
        let per_tensor_bytes = rows
            .checked_mul(heads)
            .and_then(|elements| elements.checked_mul(head_dim))
            .and_then(|elements| elements.checked_mul(2))
            .context("H3 attention tensor byte accounting overflowed")?;
        let total_qkv_output_bytes = per_tensor_bytes
            .checked_mul(4)
            .context("H3 attention Q/K/V/output byte accounting overflowed")?;
        if total_qkv_output_bytes != plan.workspace().input_output_tensor_bytes {
            bail!("H3 attention tensor bytes differ from the frozen workspace");
        }
        Ok(Self {
            plan_identity_sha256: plan.identity_sha256().to_owned(),
            backend: plan.backend(),
            kernel: plan.kernel(),
            dtype: plan.dtype(),
            sequence_rows: plan.rows(),
            heads: plan.heads(),
            head_dim: plan.head_dim(),
            tensor_bytes: AttentionTensorBytes {
                query_bytes: per_tensor_bytes,
                key_bytes: per_tensor_bytes,
                value_bytes: per_tensor_bytes,
                output_bytes: per_tensor_bytes,
                total_qkv_output_bytes,
            },
            workspace: plan.workspace().clone(),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
struct PlannedScenario {
    scenario: ScenarioFacts,
    attention: PlanFacts,
}

#[derive(Debug, Serialize)]
struct ScenarioCatalogReport {
    schema: &'static str,
    scenarios: Vec<ScenarioFacts>,
    model_artifacts_accessed: bool,
    runtime_activated: bool,
}

#[derive(Debug, Serialize)]
struct Selection {
    plan_only: bool,
    selected_scenarios: Vec<String>,
    warmup_iterations: usize,
    measured_iterations: usize,
    allow_long: bool,
    max_warmup_iterations: usize,
    max_measured_iterations: usize,
    max_long_warmup_iterations: usize,
    max_long_measured_iterations: usize,
}

#[derive(Debug, Serialize)]
struct PerStepTiming {
    definition: &'static str,
    warmup_iterations: usize,
    measured_iterations: usize,
    samples_micros: Vec<u64>,
    minimum_micros: u64,
    median_micros: u64,
    maximum_micros: u64,
}

#[derive(Debug, Serialize)]
struct ScenarioExecution {
    scenario_id: String,
    reference_mix: ReferenceMix,
    exact_packed_rows: usize,
    telemetry: H3AttentionTelemetry,
    per_step: PerStepTiming,
    input_fixture: &'static str,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    schema: &'static str,
    decision: &'static str,
    build: H3AttentionReleaseCandidateBuild,
    observed_device: H3AttentionDevice,
    authority_identity_sha256: String,
    selection: Selection,
    planned_scenarios: Vec<PlannedScenario>,
    executions: Vec<ScenarioExecution>,
    release_activation: &'static str,
    release_requirements: Vec<H3AttentionReleaseRequirement>,
    model_artifacts_accessed: bool,
    runtime_activated: bool,
}

fn parse_positive_usize(flag: &str, value: Option<String>) -> Result<usize> {
    let value = value.ok_or_else(|| anyhow!("{flag} requires a value"))?;
    let parsed = value
        .parse::<usize>()
        .with_context(|| format!("{flag} must be a positive integer"))?;
    if parsed == 0 {
        bail!("{flag} must be positive");
    }
    Ok(parsed)
}

fn parse_nonnegative_usize(flag: &str, value: Option<String>) -> Result<usize> {
    let value = value.ok_or_else(|| anyhow!("{flag} requires a value"))?;
    value
        .parse::<usize>()
        .with_context(|| format!("{flag} must be a non-negative integer"))
}

fn parse_args(arguments: impl IntoIterator<Item = String>) -> Result<Command> {
    let mut raw = RawArguments::default();
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--help" | "-h" => raw.help = true,
            "--list" => raw.list = true,
            "--plan-only" => raw.plan_only = true,
            "--allow-long" => raw.allow_long = true,
            "--scenario" => raw.scenario_ids.push(
                arguments
                    .next()
                    .ok_or_else(|| anyhow!("--scenario requires an ID"))?,
            ),
            "--iterations" => {
                raw.measured_iterations =
                    Some(parse_positive_usize("--iterations", arguments.next())?)
            }
            "--warmup" => {
                raw.warmup_iterations = Some(parse_nonnegative_usize("--warmup", arguments.next())?)
            }
            _ => bail!("unknown argument {argument:?}\n\n{USAGE}"),
        }
    }

    if raw.help {
        if raw.list
            || raw.plan_only
            || raw.allow_long
            || !raw.scenario_ids.is_empty()
            || raw.warmup_iterations.is_some()
            || raw.measured_iterations.is_some()
        {
            bail!("--help cannot be combined with benchmark options");
        }
        return Ok(Command::Help);
    }
    validate_scenario_ids(&raw.scenario_ids)?;
    if raw.list {
        if raw.plan_only
            || raw.allow_long
            || !raw.scenario_ids.is_empty()
            || raw.warmup_iterations.is_some()
            || raw.measured_iterations.is_some()
        {
            bail!("--list cannot be combined with benchmark options");
        }
        return Ok(Command::List);
    }
    if raw.plan_only {
        if raw.allow_long || raw.warmup_iterations.is_some() || raw.measured_iterations.is_some() {
            bail!("--plan-only does not accept execution controls");
        }
        return Ok(Command::Plan {
            scenario_ids: raw.scenario_ids,
        });
    }
    if raw.scenario_ids.is_empty() {
        bail!("execution requires at least one explicit --scenario\n\n{USAGE}");
    }
    let measured_iterations = raw
        .measured_iterations
        .ok_or_else(|| anyhow!("execution requires an explicit --iterations value"))?;
    let warmup_iterations = raw.warmup_iterations.unwrap_or(DEFAULT_WARMUP_ITERATIONS);
    if measured_iterations > MAX_MEASURED_ITERATIONS {
        bail!("--iterations cannot exceed {MAX_MEASURED_ITERATIONS}");
    }
    if warmup_iterations > MAX_WARMUP_ITERATIONS {
        bail!("--warmup cannot exceed {MAX_WARMUP_ITERATIONS}");
    }
    let includes_long = selected_specs(&raw.scenario_ids)
        .iter()
        .any(|scenario| scenario.target_frames == MAX_FRAMES);
    if includes_long && !raw.allow_long {
        bail!("345-frame scenarios require the explicit --allow-long opt-in");
    }
    if includes_long
        && (measured_iterations > MAX_LONG_MEASURED_ITERATIONS
            || warmup_iterations > MAX_LONG_WARMUP_ITERATIONS)
    {
        bail!(
            "345-frame scenarios allow at most {MAX_LONG_WARMUP_ITERATIONS} warmup and {MAX_LONG_MEASURED_ITERATIONS} measured iterations"
        );
    }
    Ok(Command::Execute {
        scenario_ids: raw.scenario_ids,
        warmup_iterations,
        measured_iterations,
        allow_long: raw.allow_long,
    })
}

fn validate_scenario_ids(ids: &[String]) -> Result<()> {
    let mut seen = HashSet::new();
    for id in ids {
        if scenario_spec(id).is_none() {
            bail!("unknown scenario {id:?}; use --list to inspect the fixed scenario catalog");
        }
        if !seen.insert(id) {
            bail!("duplicate scenario {id:?}");
        }
    }
    Ok(())
}

fn scenario_spec(id: &str) -> Option<ScenarioSpec> {
    SCENARIOS.iter().copied().find(|scenario| scenario.id == id)
}

fn selected_specs(ids: &[String]) -> Vec<ScenarioSpec> {
    if ids.is_empty() {
        SCENARIOS.to_vec()
    } else {
        ids.iter()
            .map(|id| scenario_spec(id).expect("validated scenario ID"))
            .collect()
    }
}

fn descriptor_provenance() -> GenerationReferenceProvenance {
    GenerationReferenceProvenance::default()
}

fn image_reference() -> GenerationReference {
    GenerationReference::Image {
        media: GenerationReferenceAuthority::Descriptor,
        provenance: descriptor_provenance(),
        mime_type: "image/png".into(),
        width: 48,
        height: 48,
    }
}

fn video_reference() -> GenerationReference {
    GenerationReference::Video {
        media: GenerationReferenceAuthority::Descriptor,
        provenance: descriptor_provenance(),
        mime_type: "video/mp4".into(),
        width: 64,
        height: 64,
        frame_count: Some(48),
        duration_ms: 2_000,
        fps: 24.0,
        has_audio: true,
        audio_duration_ms: Some(2_000),
        audio_sample_count: Some(96_000),
        audio_sample_rate: Some(48_000),
        audio_channels: Some(2),
    }
}

fn audio_reference() -> GenerationReference {
    GenerationReference::Audio {
        media: GenerationReferenceAuthority::Descriptor,
        provenance: descriptor_provenance(),
        mime_type: "audio/wav".into(),
        duration_ms: 2_000,
        sample_rate: 48_000,
        channels: 1,
        sample_count: Some(96_000),
    }
}

fn reference_descriptors(reference_mix: ReferenceMix) -> Vec<GenerationReference> {
    match reference_mix {
        ReferenceMix::None => Vec::new(),
        ReferenceMix::Image => vec![image_reference()],
        ReferenceMix::VideoImageAudio => {
            vec![video_reference(), image_reference(), audio_reference()]
        }
    }
}

fn source_facts(reference: &GenerationReference) -> Result<ReferenceSourceFacts> {
    match reference {
        GenerationReference::Image { width, height, .. } => Ok(ReferenceSourceFacts::Image {
            width: *width,
            height: *height,
        }),
        GenerationReference::Video {
            width,
            height,
            frame_count,
            duration_ms,
            fps,
            has_audio,
            audio_sample_count,
            audio_sample_rate,
            audio_channels,
            ..
        } => Ok(ReferenceSourceFacts::Video {
            width: *width,
            height: *height,
            frame_count: frame_count.context("benchmark video fixture lacks frame_count")?,
            duration_ms: *duration_ms,
            fps: *fps,
            has_audio: *has_audio,
            audio_samples_per_channel: audio_sample_count
                .context("benchmark video fixture lacks audio_sample_count")?,
            audio_sample_rate: audio_sample_rate
                .context("benchmark video fixture lacks audio_sample_rate")?,
            audio_channels: audio_channels
                .context("benchmark video fixture lacks audio_channels")?,
        }),
        GenerationReference::Audio {
            duration_ms,
            sample_rate,
            channels,
            sample_count,
            ..
        } => Ok(ReferenceSourceFacts::Audio {
            duration_ms: *duration_ms,
            sample_rate: *sample_rate,
            channels: *channels,
            samples_per_channel: sample_count
                .context("benchmark audio fixture lacks sample_count")?,
        }),
    }
}

fn checked_add(label: &str, left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| anyhow!("{label} overflowed"))
}

fn push_segment(
    segments: &mut Vec<PackedSegment>,
    cursor: &mut u64,
    kind: PackedSegmentKind,
    reference_index: Option<usize>,
    rows: u64,
) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    let end = checked_add("packed segment row range", *cursor, rows)?;
    segments.push(PackedSegment {
        kind,
        reference_index,
        rows,
        start_row: *cursor,
        end_row_exclusive: end,
    });
    *cursor = end;
    Ok(())
}

fn scenario_facts(spec: ScenarioSpec) -> Result<ScenarioFacts> {
    let references = reference_descriptors(spec.reference_mix);
    let prepared = contract::reference_prepared_shapes_for_target(&references, spec.target_frames)
        .map_err(|error| anyhow!(error))?;
    let reference_facts = references
        .iter()
        .zip(&prepared)
        .enumerate()
        .map(|(index, (reference, prepared))| {
            Ok(ReferenceFact {
                request_index: index + 1,
                source: source_facts(reference)?,
                prepared: prepared.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let target_latent_frames = VisualTemporalGeometry::default()
        .encoded_frames(usize::try_from(spec.target_frames)?)
        .map_err(|error| anyhow!(error))?;
    let rows_per_video_frame = u64::from(DEFAULT_WIDTH / VIDEO_ROW_CELL_PIXELS)
        .checked_mul(u64::from(DEFAULT_HEIGHT / VIDEO_ROW_CELL_PIXELS))
        .context("target video rows per latent frame overflowed")?;
    let generated_video_rows = u64::try_from(target_latent_frames)?
        .checked_mul(rows_per_video_frame)
        .context("target generated video rows overflowed")?;
    let rounded_audio_rows_per_channel = u64::from(spec.target_frames)
        .checked_mul(u64::try_from(LATENT_ROWS_PER_SECOND)?)
        .and_then(|rows| rows.checked_add(u64::from(FIXED_FPS / 2)))
        .context("target generated audio rows overflowed")?
        / u64::from(FIXED_FPS);
    let generated_audio_rows = rounded_audio_rows_per_channel
        .checked_mul(u64::from(contract::AUDIO_CHANNELS))
        .context("target generated audio channel rows overflowed")?;
    let condition_visual_rows = prepared.iter().try_fold(0u64, |total, shape| {
        checked_add("condition visual rows", total, shape.visual_rows)
    })?;
    let condition_audio_rows = prepared.iter().try_fold(0u64, |total, shape| {
        checked_add("condition audio rows", total, shape.audio_rows)
    })?;
    let total_packed_rows = [
        TEXT_ROWS,
        condition_visual_rows,
        condition_audio_rows,
        generated_audio_rows,
        generated_video_rows,
    ]
    .into_iter()
    .try_fold(0u64, |total, rows| {
        checked_add("total packed rows", total, rows)
    })?;

    let mut segments = Vec::new();
    let mut cursor = 0;
    push_segment(
        &mut segments,
        &mut cursor,
        PackedSegmentKind::Text,
        None,
        TEXT_ROWS,
    )?;
    for (index, (reference, shape)) in references.iter().zip(&prepared).enumerate() {
        let request_index = index + 1;
        match reference {
            GenerationReference::Image { .. } => push_segment(
                &mut segments,
                &mut cursor,
                PackedSegmentKind::ReferenceImageVisual,
                Some(request_index),
                shape.visual_rows,
            )?,
            GenerationReference::Video { .. } => {
                push_segment(
                    &mut segments,
                    &mut cursor,
                    PackedSegmentKind::ReferenceVideoAudio,
                    Some(request_index),
                    shape.audio_rows,
                )?;
                push_segment(
                    &mut segments,
                    &mut cursor,
                    PackedSegmentKind::ReferenceVideoVisual,
                    Some(request_index),
                    shape.visual_rows,
                )?;
            }
            GenerationReference::Audio { .. } => push_segment(
                &mut segments,
                &mut cursor,
                PackedSegmentKind::ReferenceAudio,
                Some(request_index),
                shape.audio_rows,
            )?,
        }
    }
    push_segment(
        &mut segments,
        &mut cursor,
        PackedSegmentKind::GeneratedAudio,
        None,
        generated_audio_rows,
    )?;
    push_segment(
        &mut segments,
        &mut cursor,
        PackedSegmentKind::GeneratedVideo,
        None,
        generated_video_rows,
    )?;
    if cursor != total_packed_rows {
        bail!("scenario segment rows do not match the packed-row total");
    }

    Ok(ScenarioFacts {
        id: spec.id,
        target_frames: spec.target_frames,
        target_width: DEFAULT_WIDTH,
        target_height: DEFAULT_HEIGHT,
        target_fps: FIXED_FPS,
        target_latent_frames,
        reference_mix: spec.reference_mix,
        references: reference_facts,
        rows: PackedRowComposition {
            text_rows: TEXT_ROWS,
            condition_visual_rows,
            condition_audio_rows,
            generated_audio_rows,
            generated_video_rows,
            total_packed_rows,
        },
        packed_segments_in_order: segments,
    })
}

fn plan_scenario(
    authority: &mold_candle::minimax_h3::H3AttentionRuntimeAuthority,
    spec: ScenarioSpec,
) -> Result<PlannedScenario> {
    let scenario = scenario_facts(spec)?;
    let packed_rows = usize::try_from(scenario.rows.total_packed_rows)?;
    let execution = authority
        .freeze_execution(1, TOKEN_REFINER_ROWS, packed_rows)
        .map_err(|error| anyhow!(error))?;
    Ok(PlannedScenario {
        attention: PlanFacts::from_plan(&execution.packed_transformer)?,
        scenario,
    })
}

fn benchmark_scenario(
    device: &Device,
    authority: &mold_candle::minimax_h3::H3AttentionRuntimeAuthority,
    plan: &PlannedScenario,
    warmup_iterations: usize,
    measured_iterations: usize,
) -> Result<ScenarioExecution> {
    let rows = plan.attention.sequence_rows;
    let shape = (1, rows, plan.attention.heads, plan.attention.head_dim);
    // Direct BF16 device allocation avoids the multi-gigabyte host F32 copies
    // that a vector-backed fixture would create at the 108k-row boundary. The
    // FlashAttention kernel's work is value-independent; the tiny qualifier
    // remains the numerical-parity authority.
    let query = Tensor::zeros(shape, DType::BF16, device)?;
    let key = Tensor::zeros(shape, DType::BF16, device)?;
    let value = Tensor::zeros(shape, DType::BF16, device)?;
    let execution = authority
        .freeze_execution(1, TOKEN_REFINER_ROWS, rows)
        .map_err(|error| anyhow!(error))?;
    let attention_plan = &execution.packed_transformer;
    if attention_plan.identity_sha256() != plan.attention.plan_identity_sha256 {
        bail!("benchmark attention plan changed after report planning");
    }

    for _ in 0..warmup_iterations {
        let (output, _) = execute_h3_attention_with_telemetry(attention_plan, &query, &key, &value)
            .map_err(|error| anyhow!(error))?;
        drop(output);
    }
    let mut samples = Vec::with_capacity(measured_iterations);
    let mut representative = None;
    for _ in 0..measured_iterations {
        let (output, telemetry) =
            execute_h3_attention_with_telemetry(attention_plan, &query, &key, &value)
                .map_err(|error| anyhow!(error))?;
        drop(output);
        samples.push(telemetry.elapsed_micros);
        representative.get_or_insert(telemetry);
    }
    let mut sorted = samples.clone();
    sorted.sort_unstable();
    let minimum_micros = *sorted
        .first()
        .context("benchmark emitted no timing samples")?;
    let median_micros = sorted[sorted.len() / 2];
    let maximum_micros = *sorted
        .last()
        .context("benchmark emitted no timing samples")?;
    let mut telemetry = representative.context("benchmark emitted no telemetry")?;
    telemetry.elapsed_micros = median_micros;
    if telemetry.sequence_rows != rows
        || telemetry.backend != plan.attention.backend
        || telemetry.kernel != plan.attention.kernel
        || telemetry.dtype != plan.attention.dtype
        || telemetry.heads != plan.attention.heads
        || telemetry.head_dim != plan.attention.head_dim
        || telemetry.workspace != plan.attention.workspace
    {
        bail!("benchmark telemetry differs from the frozen scenario plan");
    }

    Ok(ScenarioExecution {
        scenario_id: plan.scenario.id.to_owned(),
        reference_mix: plan.scenario.reference_mix,
        exact_packed_rows: rows,
        telemetry,
        per_step: PerStepTiming {
            definition: "one synchronized full-noncausal packed self-attention dispatch for one transformer block",
            warmup_iterations,
            measured_iterations,
            samples_micros: samples,
            minimum_micros,
            median_micros,
            maximum_micros,
        },
        input_fixture: "three separate device-allocated BF16 zero tensors; no host F32 fixture",
    })
}

fn print_catalog(specs: &[ScenarioSpec]) -> Result<()> {
    let report = ScenarioCatalogReport {
        schema: "mold.minimax-h3.attention-benchmark-scenarios.v1",
        scenarios: specs
            .iter()
            .copied()
            .map(scenario_facts)
            .collect::<Result<Vec<_>>>()?,
        model_artifacts_accessed: false,
        runtime_activated: false,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_cuda(
    selected: &[ScenarioSpec],
    plan_only: bool,
    warmup_iterations: usize,
    measured_iterations: usize,
    allow_long: bool,
) -> Result<()> {
    let cuda = Device::new_cuda(0).context(
        "MiniMax H3 attention benchmark requires one visible CUDA device; no CPU fallback is allowed",
    )?;
    let observed_device = H3AttentionDevice::from_candle(&cuda);
    let build = H3AttentionReleaseCandidateBuild::current();
    let authority = build
        .qualify(observed_device)
        .map_err(|error| anyhow!(error))?;
    authority
        .verify_model(H3AttentionModelContract::released_bf16(), &cuda)
        .map_err(|error| anyhow!(error))?;

    // Every invocation plans the complete fixed catalog so both released
    // 124-frame and 345-frame boundaries remain visible even when execution is
    // deliberately narrowed to one scenario.
    let planned_scenarios = SCENARIOS
        .iter()
        .copied()
        .map(|spec| plan_scenario(&authority, spec))
        .collect::<Result<Vec<_>>>()?;
    let mut executions = Vec::new();
    if !plan_only {
        for spec in selected {
            let plan = planned_scenarios
                .iter()
                .find(|plan| plan.scenario.id == spec.id)
                .expect("fixed scenario was planned");
            executions.push(benchmark_scenario(
                &cuda,
                &authority,
                plan,
                warmup_iterations,
                measured_iterations,
            )?);
        }
    }
    let release_requirements = authority
        .require_release_activation()
        .expect_err("the H3 release candidate must never activate a shipping runtime")
        .requirements;
    let report = BenchmarkReport {
        schema: "mold.minimax-h3.attention-packed-row-benchmark.v1",
        decision: "benchmark-release-candidate-only",
        build,
        observed_device,
        authority_identity_sha256: authority.identity_sha256().to_owned(),
        selection: Selection {
            plan_only,
            selected_scenarios: selected
                .iter()
                .map(|scenario| scenario.id.to_owned())
                .collect(),
            warmup_iterations,
            measured_iterations,
            allow_long,
            max_warmup_iterations: MAX_WARMUP_ITERATIONS,
            max_measured_iterations: MAX_MEASURED_ITERATIONS,
            max_long_warmup_iterations: MAX_LONG_WARMUP_ITERATIONS,
            max_long_measured_iterations: MAX_LONG_MEASURED_ITERATIONS,
        },
        planned_scenarios,
        executions,
        release_activation: "rejected",
        release_requirements,
        model_artifacts_accessed: false,
        runtime_activated: false,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn main() -> Result<()> {
    match parse_args(env::args().skip(1))? {
        Command::Help => {
            println!("{USAGE}");
            Ok(())
        }
        Command::List => print_catalog(SCENARIOS),
        Command::Plan { scenario_ids } => {
            let selected = selected_specs(&scenario_ids);
            run_cuda(&selected, true, 0, 0, false)
        }
        Command::Execute {
            scenario_ids,
            warmup_iterations,
            measured_iterations,
            allow_long,
        } => {
            let selected = selected_specs(&scenario_ids);
            run_cuda(
                &selected,
                false,
                warmup_iterations,
                measured_iterations,
                allow_long,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn release_scenarios_bind_exact_target_and_reference_rows() {
        let target = scenario_facts(scenario_spec("release-124-target").unwrap()).unwrap();
        assert_eq!(target.target_latent_frames, 37);
        assert_eq!(target.rows.generated_video_rows, 37_296);
        assert_eq!(target.rows.generated_audio_rows, 414);
        assert_eq!(target.rows.total_packed_rows, 37_966);
        assert_eq!(
            target
                .packed_segments_in_order
                .iter()
                .map(|segment| segment.rows)
                .collect::<Vec<_>>(),
            [256, 414, 37_296]
        );

        let image = scenario_facts(scenario_spec("release-124-image").unwrap()).unwrap();
        assert_eq!(image.rows.condition_visual_rows, 4_096);
        assert_eq!(image.rows.condition_audio_rows, 0);
        assert_eq!(image.rows.total_packed_rows, 42_062);

        let mixed = scenario_facts(scenario_spec("release-124-mixed").unwrap()).unwrap();
        assert_eq!(mixed.references[0].prepared.visual_rows, 6_912);
        assert_eq!(mixed.references[0].prepared.audio_rows, 160);
        assert_eq!(mixed.references[1].prepared.visual_rows, 4_096);
        assert_eq!(mixed.references[2].prepared.audio_rows, 160);
        assert_eq!(mixed.rows.condition_visual_rows, 11_008);
        assert_eq!(mixed.rows.condition_audio_rows, 320);
        assert_eq!(mixed.rows.total_packed_rows, 49_294);

        let long = scenario_facts(scenario_spec("release-345-target").unwrap()).unwrap();
        assert_eq!(long.target_latent_frames, 107);
        assert_eq!(long.rows.generated_video_rows, 107_856);
        assert_eq!(long.rows.generated_audio_rows, 1_206);
        assert_eq!(long.rows.total_packed_rows, 109_318);
        assert_eq!(
            long.packed_segments_in_order
                .iter()
                .map(|segment| segment.rows)
                .collect::<Vec<_>>(),
            [256, 1_206, 107_856]
        );
        let long_mixed = scenario_facts(scenario_spec("release-345-mixed").unwrap()).unwrap();
        assert_eq!(long_mixed.rows.total_packed_rows, 120_646);
    }

    #[test]
    fn packed_segments_preserve_ref2va_request_and_modality_order() {
        let mixed = scenario_facts(scenario_spec("release-124-mixed").unwrap()).unwrap();
        assert_eq!(
            mixed
                .packed_segments_in_order
                .iter()
                .map(|segment| (&segment.kind, segment.reference_index, segment.rows))
                .collect::<Vec<_>>(),
            vec![
                (&PackedSegmentKind::Text, None, 256),
                (&PackedSegmentKind::ReferenceVideoAudio, Some(1), 160),
                (&PackedSegmentKind::ReferenceVideoVisual, Some(1), 6_912),
                (&PackedSegmentKind::ReferenceImageVisual, Some(2), 4_096),
                (&PackedSegmentKind::ReferenceAudio, Some(3), 160),
                (&PackedSegmentKind::GeneratedAudio, None, 414),
                (&PackedSegmentKind::GeneratedVideo, None, 37_296),
            ]
        );
        assert!(mixed
            .packed_segments_in_order
            .windows(2)
            .all(|pair| pair[0].end_row_exclusive == pair[1].start_row));
        assert_eq!(
            mixed
                .packed_segments_in_order
                .last()
                .unwrap()
                .end_row_exclusive,
            mixed.rows.total_packed_rows
        );
    }

    #[test]
    fn execution_requires_explicit_scenario_iterations_and_long_opt_in() {
        assert!(parse_args(Vec::<String>::new()).is_err());
        assert!(parse_args(strings(&["--scenario", "release-124-target"])).is_err());
        assert!(parse_args(strings(&[
            "--scenario",
            "release-345-target",
            "--iterations",
            "1"
        ]))
        .is_err());
        assert_eq!(
            parse_args(strings(&[
                "--scenario",
                "release-345-target",
                "--iterations",
                "3",
                "--warmup",
                "1",
                "--allow-long"
            ]))
            .unwrap(),
            Command::Execute {
                scenario_ids: vec!["release-345-target".into()],
                warmup_iterations: 1,
                measured_iterations: 3,
                allow_long: true,
            }
        );
        assert!(parse_args(strings(&[
            "--scenario",
            "release-345-target",
            "--iterations",
            "4",
            "--allow-long"
        ]))
        .is_err());
    }

    #[test]
    fn selection_rejects_unknown_duplicates_and_unsafe_counts() {
        assert!(parse_args(strings(&["--scenario", "unknown", "--iterations", "1"])).is_err());
        assert!(parse_args(strings(&[
            "--scenario",
            "release-124-target",
            "--scenario",
            "release-124-target",
            "--iterations",
            "1"
        ]))
        .is_err());
        assert!(parse_args(strings(&[
            "--scenario",
            "release-124-target",
            "--iterations",
            "11"
        ]))
        .is_err());
        assert!(parse_args(strings(&[
            "--scenario",
            "release-124-target",
            "--iterations",
            "1",
            "--warmup",
            "4"
        ]))
        .is_err());
    }
}
