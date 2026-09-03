//! Wire types for server-side chained video generation.
//!
//! A *chain* is a sequence of per-clip render stages stitched into a single
//! output video. The v1 CLI UX is single-prompt + arbitrary length, but the
//! wire format is stages-based from day one so the eventual movie-maker
//! (multi-prompt, keyframes, selective regen) can author stages by hand
//! without a breaking change.
//!
//! The server only ever sees the canonical [`ChainRequest`] shape — a
//! `Vec<ChainStage>`. Callers can either build that directly or use the
//! auto-expand form (`prompt` + `total_frames` + `clip_frames`), which
//! [`ChainRequest::normalise`] collapses into stages.
//!
//! See `tasks/render-chain-v1-plan.md` for the full design rationale.

use serde::{Deserialize, Serialize};

use crate::error::{MoldError, Result};
use crate::types::{DevicePlacement, GenerateRequest, OutputFormat, OutputMetadata, VideoData};

/// How the boundary between the previous stage and this stage is rendered.
///
/// - `Smooth`: the engine honors the motion-tail latent carryover from the
///   prior clip (v1 default behaviour). Produces a visual morph when the
///   prompt changes.
/// - `Cut`: fresh latent, no carryover. If the stage has a `source_image`
///   the engine uses it as the i2v seed; otherwise pure t2v.
/// - `Fade`: same engine path as `Cut`, plus a post-stitch alpha blend of
///   the last `fade_frames` of the prior clip with the first `fade_frames`
///   of this clip.
///
/// Stage 0's transition is meaningless (nothing to transition from) and is
/// coerced to `Smooth` during `ChainRequest::normalise`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum TransitionMode {
    #[default]
    Smooth,
    Cut,
    Fade,
}

/// Per-clip provenance recorded into gallery metadata for chain outputs —
/// the durable record of what each clip asked for, so a sequence can be
/// traced (and later re-edited) from the Library. Seeds are the effective
/// per-stage seeds, encoded as decimal strings (full-range u64).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainStageMetadata {
    #[serde(deserialize_with = "crate::prompt_text::deserialize_prompt")]
    pub prompt: String,
    pub frames: u32,
    pub transition: TransitionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_frames: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<String>,
    /// Ordered LoRA stack that shaped this clip. Kept per-stage because
    /// sequence clips may intentionally use different adapters.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<LoraSpec>,
}

/// Structured multi-clip provenance block on [`crate::OutputMetadata`]
/// (additive; absent for single generations and legacy rows).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainOutputMetadata {
    pub stage_count: u32,
    pub motion_tail_frames: u32,
    pub stages: Vec<ChainStageMetadata>,
}

/// Optional provenance supplied by the caller of
/// [`ChainRequest::stitched_output_metadata`]: the durable job id (absent
/// on the ephemeral shim and CLI local renders) and the effective per-stage
/// seeds once rendering has assigned them.
#[derive(Debug, Clone, Copy)]
pub struct ChainProvenance<'a> {
    pub chain_job_id: Option<&'a str>,
    pub stage_seeds: Option<&'a [u64]>,
}

/// Per-stage LoRA adapter spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct LoraSpec {
    pub path: String,
    pub scale: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Per-stage named reference character/style. **Reserved for sub-project
/// B** — populating this causes `ChainRequest::normalise` to return 422.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct NamedRef {
    pub name: String,
    #[serde(with = "crate::types::base64_bytes")]
    pub image: Vec<u8>,
}

/// A single rendered clip in a chain. Concatenated in order with motion-tail
/// trimming on continuations (stages with `idx >= 1` drop the leading
/// `motion_tail_frames` pixel frames of their output because those duplicate
/// the tail of the previous stage that the engine carried across as
/// latent-space conditioning).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainStage {
    /// Prompt used for this stage. In v1 all stages receive the same prompt
    /// (auto-expand form replicates it); the movie-maker UI in v2 will let
    /// users author per-stage prompts.
    #[schema(example = "a cat walking through autumn leaves")]
    pub prompt: String,

    /// Frame count for this stage. Must be `8k+1` (LTX-2 pipeline constraint:
    /// 9, 17, 25, …, 97).
    #[schema(example = 97)]
    pub frames: u32,

    /// Optional starting image (raw PNG/JPEG bytes, base64 in JSON). In v1
    /// this is only meaningful on `stages[0]`; later stages draw their
    /// conditioning from the prior stage's motion-tail latents instead.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::types::base64_opt"
    )]
    pub source_image: Option<Vec<u8>>,

    /// Optional negative prompt for CFG-based stages. v1 LTX-2 ignores this
    /// (the distilled family doesn't use CFG); the field is reserved so the
    /// movie-maker can round-trip it without re-migrating the wire format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,

    /// Optional per-stage seed offset. `None` in v1 — the orchestrator
    /// derives each stage's seed from the chain's base seed. Reserved as the
    /// v2 movie-maker override hook for "regenerate just this stage with a
    /// different seed".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_offset: Option<u64>,

    // NEW in multi-prompt v2 ───────────────────────────────────────────
    /// Boundary style between the previous stage and this stage.
    /// Stage 0's value is coerced to `Smooth` in `normalise`.
    #[serde(default)]
    pub transition: TransitionMode,

    /// Length in pixel frames of the crossfade when `transition == Fade`.
    /// `None` means use the server-announced default (8 frames). Capped
    /// at `fade_frames_max` from `/api/capabilities/chain-limits`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_frames: Option<u32>,

    // RESERVED for C — populated values are rejected by normalise ─────
    /// **Reserved for sub-project C.** Populating this in a request
    /// produces 422 in this release.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Ordered LoRA stack applied while rendering this clip.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<LoraSpec>,

    /// **Reserved for sub-project B.** Non-empty values produce 422.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<NamedRef>,
}

/// Chained generation request. Server accepts either the canonical form
/// (`stages` non-empty) or the auto-expand form (`prompt` + `total_frames` +
/// `clip_frames`); [`ChainRequest::normalise`] collapses the latter into the
/// former so downstream code only deals with `stages`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainRequest {
    #[schema(example = "ltx-2-19b-distilled:fp8")]
    pub model: String,

    /// Canonical stages list. Empty triggers auto-expand from
    /// `prompt`/`total_frames`/`clip_frames`.
    #[serde(default)]
    pub stages: Vec<ChainStage>,

    /// Pixel frames of motion-tail overlap between consecutive stages.
    /// `0` = no overlap (simple concat). `>0` = the final K pixel frames of
    /// stage N's latents are threaded into stage N+1's conditioning, and
    /// stage N+1's leading K output frames are dropped at stitch time.
    ///
    /// Defaults to `17` (matches the CLI `--motion-tail` and SPA defaults):
    /// `1 + 16` lands on the LTX-2 VAE's `1 + 8k` causal-grid for a clean
    /// re-encode of the carryover RGB frames. Values that do not satisfy
    /// `1 + 8k` will fail the receiving stage's tail re-encode at the VAE.
    /// Must be strictly less than each stage's `frames`.
    #[serde(default = "default_motion_tail_frames")]
    #[schema(example = 17)]
    pub motion_tail_frames: u32,

    #[schema(example = 1216)]
    pub width: u32,
    #[schema(example = 704)]
    pub height: u32,
    #[serde(default = "default_fps")]
    #[schema(example = 24)]
    pub fps: u32,

    /// Chain base seed. Per-stage seeds are derived as
    /// `base_seed ^ ((stage_idx as u64) << 32)` by the orchestrator so the
    /// whole chain is reproducible from a single seed value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 42)]
    pub seed: Option<u64>,

    #[schema(example = 8)]
    pub steps: u32,

    #[schema(example = 3.0)]
    pub guidance: f64,

    /// Denoising strength for `stages[0].source_image`. Ignored when the
    /// first stage has no source image. Continuation stages are always
    /// full-strength conditioned via motion-tail latents.
    #[serde(default = "default_strength")]
    #[schema(example = 1.0)]
    pub strength: f64,

    #[serde(default = "default_output_format")]
    pub output_format: OutputFormat,

    /// This sequence is an implementation detail of ONE print, not a sequence
    /// the user authored.
    ///
    /// `mold run --frames 200` splits a long video into clips because the
    /// model cannot render it in one pass; the user asked for a video, not for
    /// a chain. An ephemeral job renders exactly like an authored one and
    /// publishes the same stitched print, but it is absent from
    /// `GET /api/chain-jobs`, its working directory is swept after
    /// finalization, it refuses resume, and its print carries NO
    /// `chain_job_id` — so "Reuse settings" restores a one-shot rather than
    /// opening the clip rail (`studio/lib/sequenceReuse.ts`).
    ///
    /// Additive: absent means `false`, which is an authored sequence.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ephemeral: bool,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<DevicePlacement>,

    /// User-authored title for the **stitched** print. Validated by
    /// [`crate::validate_print_title`] at submission, embedded into the
    /// stitched output's `OutputMetadata.title`, seeded into its gallery row,
    /// and folded into its filename as a lossy `~slug` exactly like a
    /// one-shot. Additive; absent means untitled.
    ///
    /// A sequence has one print, so this titles that print — never an
    /// intermediate clip, which is a working artifact inside the job dir and
    /// never reaches the gallery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "Smurf village at dusk")]
    pub title: Option<String>,

    /// Tags to file the **stitched** print under. Same contract, cap, and
    /// normalization as `GenerateRequest.tags`. Additive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,

    /// Collection to file the **stitched** print into. Same contract as
    /// `GenerateRequest.collection`. Additive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<crate::CollectionRef>,

    /// Original source prompt shared by a client-prepared sibling batch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_transform: Option<crate::PromptTransformProvenance>,
    /// Durable prepared-batch identity and one-based sibling position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_index: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_count: Option<u32>,

    /// User-facing authoring mode, independent of normalized execution
    /// shape. Older clients omit it and retain authored-Sequence behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_mode: Option<crate::GenerationOutputMode>,

    // ── Auto-expand form ────────────────────────────────────────────────
    // These are only read when `stages` is empty; `normalise` clears them
    // after expansion so the canonical form only ever carries `stages`.
    /// Auto-expand: single prompt replicated across all stages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    /// Auto-expand: total pixel frames the stitched output should cover.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_frames: Option<u32>,

    /// Auto-expand: per-clip frame count. Defaults to `97` (LTX-2 19B/22B
    /// distilled cap). Must be `8k+1`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_frames: Option<u32>,

    /// Auto-expand: starting image for `stages[0]`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::types::base64_opt"
    )]
    pub source_image: Option<Vec<u8>>,

    /// Generate per-stage audio and mux it into the final stitched output.
    /// Only meaningful for AV-capable families (LTX-2 / LTX-2.3); the server
    /// rejects `Some(true)` for non-AV models. `None` means "no preference"
    /// and resolves to off — chains opt in to audio explicitly so existing
    /// callers don't suddenly start producing audio they didn't ask for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_audio: Option<bool>,
}

/// Canonical TOML-shaped projection of a normalised [`ChainRequest`].
///
/// Echoed back in [`ChainResponse::script`] so clients can save the exact
/// form that was rendered without re-serialising the request body (which
/// carries auto-expand sugar and other transport-only fields).
#[derive(Debug, Clone, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainScript {
    pub schema: String, // always "mold.chain.v1"
    pub chain: ChainScriptChain,
    #[serde(rename = "stage")]
    pub stages: Vec<ChainStage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainScriptChain {
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub steps: u32,
    pub guidance: f64,
    pub strength: f64,
    pub motion_tail_frames: u32,
    pub output_format: OutputFormat,
    /// Echo of [`ChainRequest::enable_audio`]. Omitted from TOML when unset
    /// so v1 scripts (no audio) deserialise unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_audio: Option<bool>,
}

impl From<&ChainRequest> for ChainScript {
    fn from(req: &ChainRequest) -> Self {
        ChainScript {
            schema: "mold.chain.v1".into(),
            chain: ChainScriptChain {
                model: req.model.clone(),
                width: req.width,
                height: req.height,
                fps: req.fps,
                seed: req.seed,
                steps: req.steps,
                guidance: req.guidance,
                strength: req.strength,
                motion_tail_frames: req.motion_tail_frames,
                output_format: req.output_format,
                enable_audio: req.enable_audio,
            },
            stages: req.stages.clone(),
        }
    }
}

/// VRAM feasibility estimate — populated by sub-project D. `None` in this
/// release.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VramEstimate {
    /// Peak VRAM the heaviest single stage is predicted to reach — the max
    /// over stages, never their sum. Sequence stages execute strictly one at
    /// a time, so no two working sets are ever co-resident; summing would
    /// report roughly N times the truth and make a long sequence look
    /// infeasible on any card, which is the exact opposite of the signal a
    /// user needs. A long sequence costs time, not memory.
    pub worst_case_bytes: u64,
    /// Whether every stage fit the roomiest sampled device at validation
    /// time. **Advisory only.** Admission re-derives placement from live
    /// device facts, so this must never gate submission — VRAM freed between
    /// validate and submit would strand a job that would have run.
    pub fits: bool,
}

/// One normalized stage in a chain validation response. Media and negative
/// prompt contents are deliberately not echoed; callers only need to know
/// which conditioning inputs survived normalization.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainValidationStage {
    pub prompt: String,
    pub frames: u32,
    pub output_frames: u32,
    pub transition: TransitionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_frames: Option<u32>,
    pub has_source_image: bool,
    pub has_negative_prompt: bool,
}

/// Read-only normalized plan returned by
/// `POST /api/generate/chain/validate`. This endpoint never creates a durable
/// job or starts downloads.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainValidationResponse {
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub motion_tail_frames: u32,
    pub stage_count: u32,
    pub estimated_total_frames: u32,
    pub estimated_duration_ms: u64,
    pub stages: Vec<ChainValidationStage>,
    pub warnings: Vec<String>,
    /// Reserved until the server's chain estimator is populated. Kept in the
    /// stable response now so clients can render it additively when available.
    pub vram_estimate: Option<VramEstimate>,
}

impl ChainValidationResponse {
    pub fn from_normalized(req: &ChainRequest, warnings: Vec<String>) -> Self {
        let estimated_total_frames = req.estimated_total_frames();
        let fps = req.fps.max(1);
        Self {
            model: req.model.clone(),
            width: req.width,
            height: req.height,
            fps,
            motion_tail_frames: req.motion_tail_frames,
            stage_count: req.stages.len() as u32,
            estimated_total_frames,
            estimated_duration_ms: u64::from(estimated_total_frames) * 1_000 / u64::from(fps),
            stages: req
                .stages
                .iter()
                .enumerate()
                .map(|(idx, stage)| {
                    let next = req.stages.get(idx + 1);
                    ChainValidationStage {
                        prompt: stage.prompt.clone(),
                        frames: stage.frames,
                        output_frames: stage_contributed_frames(
                            idx,
                            stage.frames,
                            stage.transition,
                            next.map(|candidate| candidate.transition),
                            next.and_then(|candidate| candidate.fade_frames),
                            req.motion_tail_frames,
                        ),
                        transition: stage.transition,
                        fade_frames: stage.fade_frames,
                        has_source_image: stage.source_image.is_some(),
                        has_negative_prompt: stage
                            .negative_prompt
                            .as_deref()
                            .is_some_and(|value| !value.trim().is_empty()),
                    }
                })
                .collect(),
            warnings,
            vram_estimate: None,
        }
    }

    /// Attach an advisory VRAM estimate. Separate from `from_normalized`
    /// because the estimate needs live device facts the core crate cannot see.
    #[must_use]
    pub fn with_vram_estimate(mut self, estimate: Option<VramEstimate>) -> Self {
        self.vram_estimate = estimate;
        self
    }
}

/// Response from a chained generation request. The `video` is the stitched
/// output; individual per-stage clips are not returned.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainResponse {
    pub video: VideoData,
    /// Number of stages that actually ran (matches `request.stages.len()`
    /// after normalisation).
    #[schema(example = 5)]
    pub stage_count: u32,
    /// GPU ordinal that handled the chain (multi-GPU servers only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,

    // NEW ──────────────────────────────────────────────────────────────
    /// Canonical TOML-shaped echo of the rendered script. Clients can save
    /// this directly as a `.toml` file.
    pub script: ChainScript,

    /// Reserved for sub-project D; `None` in this release.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_estimate: Option<VramEstimate>,

    /// Advisories the server attached to this response — see
    /// [`crate::GenerateResponse::request_warnings`]. For a chain the one that
    /// matters is a stitched-print filing the host could not apply.
    /// Populated by [`crate::MoldClient`] from `x-mold-request-warning`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub request_warnings: Vec<String>,
}

/// SSE completion event for a successful chain run. Streamed as the final
/// `data:` frame under the `event: complete` SSE type. The payload is
/// base64-encoded to stay JSON-safe; clients decode it into `VideoData`.
///
/// This is a sibling to [`crate::types::SseCompleteEvent`] rather than an
/// extension so image/video vs. chain completion shapes stay independent
/// and can evolve separately.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SseChainCompleteEvent {
    /// Base64-encoded stitched video bytes (format per `format` field), or an
    /// empty string when `X-Mold-SSE-Payload: metadata-only` was requested.
    pub video: String,
    pub format: OutputFormat,
    #[schema(example = 1216)]
    pub width: u32,
    #[schema(example = 704)]
    pub height: u32,
    #[schema(example = 400)]
    pub frames: u32,
    #[schema(example = 24)]
    pub fps: u32,
    /// Base64-encoded first-frame PNG thumbnail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<String>,
    /// Base64-encoded animated GIF preview (always emitted for gallery UI).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gif_preview: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub has_audio: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_sample_rate: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_channels: Option<u32>,
    /// Number of stages that ran end-to-end.
    #[schema(example = 5)]
    pub stage_count: u32,
    /// GPU ordinal that handled the chain (multi-GPU only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,
    /// Wall-clock elapsed time across all stages + stitching.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_time_ms: Option<u64>,
    /// Canonical echo of the normalised chain request, so streaming clients
    /// can save/reload the rendered script without re-serialising the
    /// transport-only fields in the submitted request body.
    #[serde(default)]
    pub script: ChainScript,
    /// Reserved for sub-project D; `None` in this release.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_estimate: Option<VramEstimate>,
    /// Filename this stitched output was saved under in the server gallery.
    /// Present for servers that persist chain output and absent on older
    /// servers or when gallery output is disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// The exact metadata recorded for the saved stitched output. Streaming
    /// clients can use this with `filename` instead of reconstructing chain
    /// provenance from the request or encoded media.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Object)]
    pub metadata: Option<Box<OutputMetadata>>,
}

/// Chain-specific SSE progress event. Streamed as `data:` JSON frames from
/// `POST /api/generate/chain/stream` under the `event: progress` SSE type.
///
/// Per-stage denoise steps are wrapped with `stage_idx` so consumers can
/// render stacked progress bars (overall chain + per-stage) without a
/// separate subscription. Non-denoise engine events (weight load, cache
/// hits, etc.) are intentionally not forwarded through this enum in v1 —
/// they're scoped to individual stages and the UX goal for v1 is per-stage
/// progress, not per-component telemetry.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChainProgressEvent {
    /// Emitted once at the start of the chain, after normalisation. Gives
    /// consumers the final stage count and the target stitched frame total
    /// so they can size progress bars up front.
    ChainStart {
        stage_count: u32,
        estimated_total_frames: u32,
    },
    /// Stage `stage_idx` (0-indexed) has started its denoise loop.
    StageStart { stage_idx: u32 },
    /// Per-step denoise progress for the active stage.
    DenoiseStep {
        stage_idx: u32,
        step: u32,
        total: u32,
    },
    /// Stage finished generating; `frames_emitted` is the raw clip frame
    /// count before motion-tail trim at stitch time.
    StageDone { stage_idx: u32, frames_emitted: u32 },
    /// All stages complete; stitching/encoding the final MP4.
    Stitching { total_frames: u32 },
}

/// Structured error payload returned in the 502 response body when a chain
/// stage fails mid-run. Allows UIs to show actionable retry hints (e.g.,
/// "stage 2 of 5 failed — retry from here").
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainFailure {
    /// Human-readable summary of where the failure landed.
    #[schema(example = "stage render failed")]
    pub error: String,
    /// Zero-based index of the stage whose render returned Err.
    #[schema(example = 2)]
    pub failed_stage_idx: u32,
    /// Number of stages that completed successfully before the failure.
    #[schema(example = 2)]
    pub elapsed_stages: u32,
    /// Cumulative generation time across the completed stages, in ms.
    #[schema(example = 12_340)]
    pub elapsed_ms: u64,
    /// Inner error message from the orchestrator (`format!("{e:#}")`).
    #[schema(example = "simulated GPU OOM on stage 2")]
    pub stage_error: String,
}

fn default_motion_tail_frames() -> u32 {
    17
}

fn default_fps() -> u32 {
    24
}

fn default_strength() -> f64 {
    1.0
}

fn default_output_format() -> OutputFormat {
    OutputFormat::Mp4
}

/// Maximum number of stages the v1 orchestrator will accept in a single
/// chain. 16 × 97-frame clips ≈ 1552 frames ≈ 64 s at 24 fps — comfortably
/// past the 400-frame target without risking runaway jobs.
pub const MAX_CHAIN_STAGES: usize = 16;

/// Per-clip frame count an LTX-2 render chain uses for one clip.
///
/// This is the model's *clip size*, not the family's ceiling — see
/// [`crate::validation::ltx2_max_frames_at_fps`] for that. A 97-frame clip is
/// what fits comfortably on a single consumer GPU; the family's real
/// single-request budget is 20 s of runtime, which at 24 fps is 481 frames and
/// would need far more VRAM than most cards have. Users who deliberately want
/// one long clip instead of a stitched sequence raise `--clip-frames`, which
/// is clamped to the family budget rather than to this value.
pub const LTX2_DEFAULT_CLIP_FRAMES: u32 = 97;

/// Per-clip frame count LTX-Video renders in one clip. Mirrors
/// `mold_inference::chain::LTX_VIDEO_FRAMES_PER_CLIP_CAP`, which `mold-core`
/// cannot depend on; the two are pinned together by a contract test in
/// `mold-server`'s `chain_limits`.
pub const LTX_VIDEO_DEFAULT_CLIP_FRAMES: u32 = 97;

/// Per-clip frame count a wan render chain uses for one clip, in pixel frames.
///
/// The two-expert A14B pair measures near the 24 GB envelope well before
/// wan's 257-frame request cap; the single-expert 5B has room for its own
/// shipped 121. Both values sit on wan's `4k+1` grid, so a clip started at the
/// routing default is submittable.
///
/// Those two numbers are a **floor**, not the answer. A tier whose manifest
/// default was raised past its family floor on a measurement has to be able to
/// render that default as one clip — otherwise running the model with no
/// `--frames` at all silently produces a stitched sequence instead of the clip
/// the default advertises. That is exactly what shipped: #776 item 4 raised
/// the Q5/Q4 A14B tiers to the checkpoint's trained 81 frames once block
/// offload made them fit, while the routing default stayed at the pre-offload
/// 53, so `mold run wan22-t2v-a14b:q5` rendered 2 clips and 106 frames rather
/// than one 81-frame clip. Reading the tier's own recorded default keeps the
/// two from drifting again; the floor still covers models whose manifest
/// records a smaller default (Q8 and fp8 A14B stay at 33) and opaque catalog
/// IDs with no manifest at all.
pub fn wan_default_clip_frames(model: &str) -> u32 {
    let floor = if model.to_ascii_lowercase().contains("a14b") {
        53
    } else {
        121
    };
    crate::manifest::find_manifest(&crate::manifest::resolve_model_name(model))
        .and_then(|manifest| manifest.defaults.frames)
        .map_or(floor, |tier_default| tier_default.max(floor))
}

/// The per-model clip size ONE generation renders when mold auto-chains.
///
/// This is deliberately **not** the family's single-request ceiling. The
/// ceiling (`crate::validation::max_frames_for_family_at_fps`) is what a
/// single denoise is *allowed* to ask for — LTX-2's is a 20 s runtime budget,
/// 481 frames at 24 fps. The routing clip size is what mold actually renders
/// per clip: a VRAM envelope for wan, and the model's shipped clip default for
/// the LTX families. Conflating the two is what let a Studio sequence composer
/// offer a single 481-frame LTX-2 clip that the one-shot auto-chain path would
/// have split into five.
///
/// `None` means the family has no routing clip size — it is not chain-capable,
/// so callers keep the family ceiling.
pub fn routing_clip_frames(family: &str, model: &str) -> Option<u32> {
    match family {
        "ltx2" => Some(LTX2_DEFAULT_CLIP_FRAMES),
        "ltx-video" => Some(LTX_VIDEO_DEFAULT_CLIP_FRAMES),
        "wan" => Some(wan_default_clip_frames(model)),
        _ => None,
    }
}

/// The refusal a **one-shot** auto-chain earns on a text-to-video wan tier —
/// the single authority every door renders.
///
/// A wan checkpoint whose advertised `source_image` contract is `Unsupported`
/// has no conditioning channel at all, so nothing crosses a clip boundary:
/// every stage re-derives the scene from the same prompt and the same seed and
/// the "longer" video is the same clip rendered again, with a visible reset at
/// each seam. Measured on `wan22-t2v-a14b:q8` at 219 frames / 3 stages, frames
/// a whole stage apart scored 38.1-44.2 dB PSNR against each other while
/// frames ten apart INSIDE a stage scored 26.0 dB — a stage boundary moved the
/// picture less than ten frames of ordinary motion did. So this is a refusal
/// rather than a zero-length seam (#1508).
///
/// The rule applies to the auto-chain a user did not ask for: `mold run
/// --frames 259`, the Studio's Create rail, and the `ephemeral` chain job both
/// of them post. An AUTHORED sequence is untouched — there, repeated stages
/// are what the author asked for.
///
/// The sentence is deliberately surface-neutral: it reaches a GUI user as the
/// reason a control is disabled as often as it reaches a terminal, so it names
/// no CLI flag. It also names image-to-video TAGS rather than model families,
/// because `wan22-ti2v-5b:dmd` is itself refused — recommending the bare
/// family would send a user to a tier this same rule turns away.
///
/// `family` is a parameter rather than a manifest lookup because an installed
/// `cv:` / `hf:` checkpoint is only classified through the server's sidecar
/// overlay, and because the contract means something different elsewhere: an
/// LTX-2 tier carries latent context across the seam whatever it says about
/// source images. An unclassified contract (`None`) is "unknown", never a
/// declared refusal — #783 added wan auto-chaining precisely so opaque catalog
/// ids route, and refusing them on a guess would undo that.
///
/// `clip_frames` is the size ONE generation renders — [`routing_clip_frames`],
/// or the caller's own `--clip-frames` / `clip_frames` override.
pub fn text_only_wan_auto_chain_refusal(
    family: Option<&str>,
    model: &str,
    source_image: Option<crate::SourceImageCapability>,
    total_frames: u32,
    clip_frames: u32,
) -> Option<String> {
    if family != Some("wan")
        || !matches!(
            source_image,
            Some(crate::SourceImageCapability::Unsupported)
        )
        || total_frames <= clip_frames
    {
        return None;
    }
    Some(format!(
        "'{model}' is text-to-video and cannot continue motion across a clip \
         boundary, so rendering {total_frames} frames would repeat the same \
         ~{clip_frames}-frame clip rather than extend it. Reduce the frame count \
         to {clip_frames} or fewer for one continuous clip, or use an \
         image-to-video tier (wan22-i2v-a14b, wan22-ti2v-5b:turbo), which \
         seeds each continuation with the previous clip's final frame."
    ))
}

impl ChainRequest {
    /// Canonicalize raw prompt text at one chain ingress. Callers that forward
    /// an already-canonical request must protect backslashes on that wire hop
    /// instead of applying this method twice.
    pub fn normalize_prompt_newlines(&mut self) {
        for stage in &mut self.stages {
            stage.prompt = crate::normalize_prompt_newlines(&stage.prompt).into_owned();
            stage.negative_prompt = stage
                .negative_prompt
                .take()
                .map(|prompt| crate::normalize_prompt_newlines(&prompt).into_owned());
        }
        self.original_prompt = self
            .original_prompt
            .take()
            .map(|prompt| crate::normalize_prompt_newlines(&prompt).into_owned());
        self.prompt = self
            .prompt
            .take()
            .map(|prompt| crate::normalize_prompt_newlines(&prompt).into_owned());
    }

    /// Build a synthetic single-clip `GenerateRequest` describing the
    /// stitched output, so gallery rows and embedded metadata can reuse
    /// the existing single-clip schema. `stages[0]` supplies the prompt,
    /// negative prompt, and source image (the row only has one prompt
    /// field — continuation prompts are dropped, acceptable for v1).
    ///
    /// Callers must pass a normalised request (`stages` non-empty).
    /// `actual_format` is the container after encode fallbacks (e.g. a
    /// WebP request that fell back to APNG records APNG).
    pub fn synthetic_generate_request(
        &self,
        actual_format: OutputFormat,
        frames: u32,
        fps: u32,
    ) -> GenerateRequest {
        let first = self
            .stages
            .first()
            .expect("synthetic_generate_request requires a normalised ChainRequest");
        // A sequence with distinct clip prompts must not be recorded under
        // clip 1's prompt alone — join them (one line per clip) so gallery
        // search matches any clip. Uniform prompts (auto-expanded chains)
        // keep the single prompt.
        let prompt = if self.stages.iter().all(|stage| stage.prompt == first.prompt) {
            first.prompt.clone()
        } else {
            self.stages
                .iter()
                .map(|stage| stage.prompt.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        };
        GenerateRequest {
            mesh: None,
            video_only: None,
            // A sequence has exactly one gallery print — the stitched output.
            // Its title and filing come from the chain request and ride the
            // same `OutputMetadata` plumbing as a one-shot's, which is why
            // they belong on the synthetic request rather than being stamped
            // onto the metadata afterwards.
            collection: self.collection.clone(),
            tags: self.tags.clone(),
            title: self.title.clone(),
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            prompt,
            negative_prompt: first.negative_prompt.clone(),
            model: self.model.clone(),
            width: self.width,
            height: self.height,
            steps: self.steps,
            guidance: self.guidance,
            seed: self.seed,
            batch_size: 1,
            output_format: Some(actual_format),
            embed_metadata: Some(false),
            scheduler: None,
            cfg_plus: None,
            edit_images: None,
            references: None,
            source_image: first.source_image.clone(),
            source_image_name: None,
            strength: self.strength,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: self.original_prompt.clone(),
            prompt_transform: self.prompt_transform.clone(),
            batch_id: self.batch_id.clone(),
            batch_index: self.batch_index,
            batch_count: self.batch_count,
            lora: None,
            frames: Some(frames),
            fps: Some(fps),
            upscale_model: None,
            gif_preview: false,
            enable_audio: self.enable_audio,
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
            // Sequences render every clip through the chain stage path, which
            // keeps the pipeline's own guider constants. There is no chain
            // wire field to carry an override, so recording one here would
            // claim a setting the render never used.
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            placement: self.placement.clone(),
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        }
    }

    /// Gallery/PNG metadata for the stitched chain output, derived from
    /// [`Self::synthetic_generate_request`] via
    /// `OutputMetadata::from_generate_request` so chain rows can never
    /// drift from the single-clip metadata semantics (e.g. `strength` is
    /// only recorded when the chain starts from a source image).
    pub fn stitched_output_metadata(
        &self,
        actual_format: OutputFormat,
        frame_count: u32,
        provenance: Option<&ChainProvenance>,
    ) -> OutputMetadata {
        let synth = self.synthetic_generate_request(actual_format, frame_count, self.fps);
        let mut metadata = OutputMetadata::from_generate_request(
            &synth,
            self.seed.unwrap_or(0),
            None,
            crate::build_info::version_string(),
        );
        metadata.chain_job_id = provenance.and_then(|p| p.chain_job_id).map(str::to_string);
        metadata.output_mode = Some(
            self.output_mode
                .unwrap_or(crate::GenerationOutputMode::Sequence),
        );
        let stage_seeds = provenance.and_then(|p| p.stage_seeds);
        metadata.chain = Some(ChainOutputMetadata {
            stage_count: self.stages.len() as u32,
            motion_tail_frames: self.motion_tail_frames,
            stages: self
                .stages
                .iter()
                .enumerate()
                .map(|(idx, stage)| ChainStageMetadata {
                    prompt: stage.prompt.clone(),
                    frames: stage.frames,
                    transition: stage.transition,
                    fade_frames: stage.fade_frames,
                    seed: stage_seeds
                        .and_then(|seeds| seeds.get(idx))
                        .map(u64::to_string),
                    loras: stage.loras.clone(),
                })
                .collect(),
        });
        metadata
    }

    /// Collapse the auto-expand form into a canonical `Vec<ChainStage>` and
    /// validate the result. Called once on the server side immediately after
    /// JSON parsing, before any engine work kicks off.
    ///
    /// Post-conditions on a successful return:
    /// - `self.stages` is non-empty.
    /// - Each stage's `frames` is `8k+1` and `> 0`.
    /// - `self.stages.len() <= MAX_CHAIN_STAGES`.
    /// - All auto-expand fields are `None` (caller must use `self.stages`).
    pub fn normalise(self) -> Result<Self> {
        self.normalise_with_family(None)
    }

    /// [`normalise`](Self::normalise) with the family already resolved.
    ///
    /// The manifest cannot classify an installed `cv:` / `hf:` id, so a
    /// manifest-only lookup fell back to the LTX `8k+1` grid and rejected a
    /// 53-frame wan stage as "this family requires 8k+1" — immediately after
    /// the server had resolved it as wan from the sidecar overlay (#783).
    /// Callers holding a resolved family pass it; `None` keeps the historical
    /// manifest-only behaviour for callers that do not.
    pub fn normalise_with_family(mut self, family_hint: Option<&str>) -> Result<Self> {
        // Resolve the family from the manifest rather than passing `None`.
        // With a family-aware ceiling and grid, a `None` check is not merely
        // loose — it is a different answer: it would reject a 1920x1088 LTX-2
        // sequence the HTTP path admits, and accept a 16-aligned size the
        // LTX-2 VAE's /32 grid cannot render.
        let family = crate::manifest::find_manifest(&self.model)
            .map(|m| m.family.clone())
            .or_else(|| {
                family_hint
                    .filter(|hint| !hint.is_empty())
                    .map(str::to_string)
            });
        // A sequence clip is one generation, so it is bound by exactly the
        // same ceiling — including the composed one. Resolving the
        // composition from the model keeps a 4K sequence admissible wherever a
        // 4K single shot is, and refused wherever it is not.
        let composition = if family.as_deref() == Some("ltx2") {
            crate::validation::ltx2_spatial_composition(&self.model, None)
        } else {
            crate::validation::Ltx2SpatialComposition::SinglePass
        };
        crate::validation::validate_generation_dimensions_for_model(
            &self.model,
            self.width,
            self.height,
            family.as_deref(),
            composition,
        )
        .map_err(MoldError::Validation)?;

        // The stitched print's title and filing are validated here, on the
        // same normalise pass every chain entry point runs, so a bad tag is
        // refused before a durable job dir exists. The filing is also
        // MATERIALIZED: the manifest stores this request verbatim and
        // `stitched_output_metadata` embeds it, so leaving raw spellings here
        // would put a different filing in the print's provenance than the one
        // the row receives — and a chain's request outlives the submission,
        // so the divergence survives every resume.
        if let Some(title) = self.title.as_deref() {
            crate::validate_print_title(title).map_err(MoldError::Validation)?;
        }
        let organization =
            crate::validate_request_organization(self.tags.as_deref(), self.collection.as_ref())
                .map_err(MoldError::Validation)?;
        if self.tags.is_some() {
            self.tags = (!organization.tags.is_empty()).then_some(organization.tags);
        }
        if let Some(Ok(name)) = organization.collection {
            let id = self
                .collection
                .as_ref()
                .and_then(|reference| reference.id.clone());
            self.collection = Some(crate::CollectionRef {
                id,
                name: Some(name),
            });
        }

        if self.stages.is_empty() {
            let prompt = self.prompt.take().ok_or_else(|| {
                MoldError::Validation(
                    "chain request needs either stages[] or prompt + total_frames".into(),
                )
            })?;
            let total_frames = self.total_frames.ok_or_else(|| {
                MoldError::Validation("chain auto-expand requires total_frames".into())
            })?;
            if total_frames == 0 {
                return Err(MoldError::Validation(
                    "chain total_frames must be > 0".into(),
                ));
            }
            // A caller that named no clip size gets the model's own routing
            // clip size — the size ONE generation renders — never the family's
            // single-request ceiling. Wan's is a VRAM envelope on its own
            // `4k+1` grid, so the old flat 97 auto-expanded wan into clips
            // nearly twice the envelope its routing default measures at.
            let clip_frames = self.clip_frames.unwrap_or_else(|| {
                family
                    .as_deref()
                    .and_then(|family| routing_clip_frames(family, &self.model))
                    .unwrap_or(LTX2_DEFAULT_CLIP_FRAMES)
            });
            if clip_frames == 0 {
                return Err(MoldError::Validation(
                    "chain clip_frames must be > 0".into(),
                ));
            }
            // The grid is the family's, not a constant: wan's VAE compresses
            // time by 4 where the LTX families compress by 8. A hardcoded 8
            // rejected every wan auto-chain, including the 53-frame routing
            // default the CLI itself picks.
            let step = family
                .as_deref()
                .and_then(crate::validation::frame_step_for_family)
                .unwrap_or(8);
            if clip_frames % step != 1 {
                let examples: Vec<String> = (1..5).map(|k| (k * step + 1).to_string()).collect();
                return Err(MoldError::Validation(format!(
                    "chain clip_frames ({clip_frames}) must be {step}k+1 ({}, …)",
                    examples.join(", "),
                )));
            }
            let motion_tail = self.motion_tail_frames;
            if motion_tail >= clip_frames {
                return Err(MoldError::Validation(format!(
                    "motion_tail_frames ({motion_tail}) must be strictly less than clip_frames ({clip_frames})",
                )));
            }

            let source_image = self.source_image.take();
            self.stages = build_auto_expand_stages(
                &prompt,
                total_frames,
                clip_frames,
                motion_tail,
                source_image,
                family.as_deref(),
            )?;
        }

        if self.stages.is_empty() {
            return Err(MoldError::Validation("chain request has no stages".into()));
        }
        if self.stages.len() > MAX_CHAIN_STAGES {
            return Err(MoldError::Validation(format!(
                "chain request has {} stages; maximum is {}",
                self.stages.len(),
                MAX_CHAIN_STAGES,
            )));
        }
        // The carryover frames re-encode through the family's own video VAE,
        // so the tail sits on that VAE's temporal grid — 8x causal for LTX-2,
        // 4x for wan.
        let grid_step = family
            .as_deref()
            .and_then(crate::validation::frame_step_for_family)
            .unwrap_or(8);
        if self.motion_tail_frames != 0 && self.motion_tail_frames % grid_step != 1 {
            return Err(MoldError::Validation(format!(
                "motion_tail_frames ({}) must be 0 or {grid_step}k+1 so the carryover RGB frames \
                 re-encode cleanly through this family's video VAE temporal grid",
                self.motion_tail_frames,
            )));
        }
        for (idx, stage) in self.stages.iter().enumerate() {
            if stage.frames == 0 {
                return Err(MoldError::Validation(format!("stage {idx} has 0 frames",)));
            }
            if stage.frames % grid_step != 1 {
                return Err(MoldError::Validation(format!(
                    "stage {idx} has {} frames; this family requires {grid_step}k+1",
                    stage.frames,
                )));
            }
            if self.motion_tail_frames >= stage.frames {
                return Err(MoldError::Validation(format!(
                    "motion_tail_frames ({}) must be strictly less than stage {idx}'s frames ({})",
                    self.motion_tail_frames, stage.frames,
                )));
            }
        }

        // Per-stage LoRAs use the same wire-level constraints as ordinary
        // generation. Paths are server-local except for built-in
        // `camera-control:<preset>` aliases, which the server materializes
        // before execution.
        for (idx, stage) in self.stages.iter().enumerate() {
            if stage.model.is_some() {
                return Err(MoldError::Validation(format!(
                    "stages[{idx}].model is reserved for sub-project C and not yet supported"
                )));
            }
            if stage.loras.len() > 4 {
                return Err(MoldError::Validation(format!(
                    "stages[{idx}].loras exceeds the four-LoRA stack limit"
                )));
            }
            for (lora_idx, lora) in stage.loras.iter().enumerate() {
                if !(0.0..=2.0).contains(&lora.scale) {
                    return Err(MoldError::Validation(format!(
                        "stages[{idx}].loras[{lora_idx}].scale ({}) must be in range [0.0, 2.0]",
                        lora.scale
                    )));
                }
                if !lora.path.ends_with(".safetensors") && !lora.path.starts_with("camera-control:")
                {
                    return Err(MoldError::Validation(format!(
                        "stages[{idx}].loras[{lora_idx}].path must be a .safetensors file or camera-control preset"
                    )));
                }
            }
            if !stage.references.is_empty() {
                return Err(MoldError::Validation(format!(
                    "stages[{idx}].references is reserved for sub-project B and not yet supported"
                )));
            }
        }

        // Stage 0's transition is meaningless (nothing to transition from).
        // Coerce to Smooth with a warn so scripts survive reorders.
        if let Some(first) = self.stages.first_mut() {
            if first.transition != TransitionMode::Smooth {
                tracing::warn!(
                    coerced_from = ?first.transition,
                    "stage 0 transition is meaningless; coercing to Smooth"
                );
                first.transition = TransitionMode::Smooth;
            }
        }

        // Canonicalise: clear auto-expand fields so downstream code only
        // ever reads from `stages`.
        self.prompt = None;
        self.total_frames = None;
        self.clip_frames = None;
        self.source_image = None;

        Ok(self)
    }

    /// Predicted stitched frame count — this IS the delivered length; no
    /// top-level trim exists downstream. Used by UIs for the footer summary
    /// and by the server to size the final buffer.
    ///
    /// Per-boundary rule:
    /// - smooth: drop leading `motion_tail_frames` of the incoming clip
    /// - cut: no trim
    /// - fade: replace `2 * fade_len` frames (trailing of prior + leading of
    ///   next) with `fade_len` blended frames → net `-fade_len`
    ///
    /// Sums [`stage_contributed_frames`] so the per-stage boundary math
    /// exists in exactly one place (the chain-job runner persists the same
    /// values as `frames_emitted`).
    pub fn estimated_total_frames(&self) -> u32 {
        self.stages
            .iter()
            .enumerate()
            .map(|(idx, stage)| {
                let next = self.stages.get(idx + 1);
                stage_contributed_frames(
                    idx,
                    stage.frames,
                    stage.transition,
                    next.map(|next| next.transition),
                    next.and_then(|next| next.fade_frames),
                    self.motion_tail_frames,
                )
            })
            .sum()
    }
}

/// Default crossfade length in pixel frames when a `Fade` stage omits
/// `fade_frames`. Announced to clients via `/api/capabilities/chain-limits`.
pub const DEFAULT_FADE_FRAMES: u32 = 8;

/// Frames stage `idx` contributes to the final stitched video after boundary
/// accounting — the single home of the per-stage boundary math.
/// [`ChainRequest::estimated_total_frames`] sums it and the chain-job runner
/// persists it as each stage's `frames_emitted`.
///
/// Attribution matches the persisted `frames_emitted` wire meaning:
/// - a continuation stage entering with `Smooth` loses its leading
///   `motion_tail_frames` (they duplicate the prior stage's carried tail);
/// - a stage whose NEXT boundary is `Fade` loses its trailing `fade_len`
///   (the blended block replaces them and is attributed to the incoming
///   stage);
/// - a stage entering with `Fade` keeps its full frame count (its leading
///   `fade_len` frames are replaced by the blend in place, not dropped).
pub fn stage_contributed_frames(
    idx: usize,
    stage_frames: u32,
    transition: TransitionMode,
    next_transition: Option<TransitionMode>,
    next_fade_frames: Option<u32>,
    motion_tail_frames: u32,
) -> u32 {
    let mut frames = stage_frames;
    if idx > 0 && transition == TransitionMode::Smooth {
        frames = frames.saturating_sub(motion_tail_frames);
    }
    if next_transition == Some(TransitionMode::Fade) {
        frames = frames.saturating_sub(next_fade_frames.unwrap_or(DEFAULT_FADE_FRAMES));
    }
    frames
}

/// Returns `true` iff `n` has the form `8k + 1` for some non-negative integer
/// `k` (1, 9, 17, 25, …). The LTX-2 pipeline has this constraint on pixel
/// frame counts due to the VAE's 8× temporal compression with a causal first
/// frame.
///
/// Test-only since the Wan wave (#783): production grid checks are family-
/// derived through [`crate::validation::frame_step_for_family`], because Wan's
/// grid is `4k + 1`. This stays as the literal spelling of LTX-2's own
/// constraint so the tests below assert it independently of that lookup.
#[cfg(test)]
fn is_ltx2_frame_count(n: u32) -> bool {
    n % 8 == 1
}

/// Compute the stage count and per-stage frame allocation for the auto-
/// expand form, matching the chain stitch math:
///
/// - Stage 0 contributes `clip_frames` pixel frames.
/// - Each continuation contributes `clip_frames - motion_tail_frames` new
///   frames (the leading `motion_tail_frames` are dropped at stitch time
///   because they duplicate the prior stage's latent tail).
///
/// The LAST stage is exact-fitted to the request (#1509): nothing trims the
/// stitched video afterwards, so a full-clip last stage renders — and the
/// user pays GPU time for — up to a whole clip of frames nobody asked for
/// (`--frames 145` on a 121-frame clip rendered 241). The lattice closes by
/// construction for canonical inputs (`total ≡ clip ≡ tail ≡ 1` on the
/// family's `step·k+1` grid), so the fit is exact there; when it cannot
/// close (a zero motion tail, an off-grid total) the last stage takes the
/// smallest on-grid length that still covers the request, bounding the
/// overshoot by the grid step instead of a clip. Callers disclose any
/// residual difference via [`ChainRequest::estimated_total_frames`].
fn build_auto_expand_stages(
    prompt: &str,
    total_frames: u32,
    clip_frames: u32,
    motion_tail_frames: u32,
    source_image: Option<Vec<u8>>,
    family: Option<&str>,
) -> Result<Vec<ChainStage>> {
    let (stage_count, last_stage_frames) = if total_frames <= clip_frames {
        // Single stage: match the user's requested length exactly so we
        // don't render 97 frames and throw most of them away. The frame
        // count will still be validated as 8k+1 by the caller.
        (1u32, total_frames)
    } else {
        let effective = clip_frames - motion_tail_frames;
        // effective > 0 because the caller has already ensured
        // motion_tail_frames < clip_frames.
        let remainder = total_frames - clip_frames;
        let count = 1 + remainder.div_ceil(effective);
        // Exact-fit the last stage. Every earlier stage is a full clip, so
        // the last must deliver `total - delivered_before_last` new frames
        // on top of the `motion_tail_frames` it re-renders for the seam.
        let delivered_before_last = clip_frames + (count - 2) * effective;
        let needed = motion_tail_frames + (total_frames - delivered_before_last);
        // Snap up to the family's `step·k+1` frame grid. For canonical
        // inputs `needed` is already on it and the snap is the identity;
        // `needed <= clip_frames` and the round-up never passes the next
        // grid point, so the last stage never exceeds the clip envelope.
        let step = family
            .and_then(crate::validation::frame_step_for_family)
            .unwrap_or(8);
        let last = needed + (step + 1 - needed % step) % step;
        // Floor at the smallest non-degenerate grid clip. A zero tail with
        // `total == clip + 1` snaps to a 1-frame last stage — on-grid and
        // above the tail, but a single pixel frame is one latent frame,
        // which wan's continuation conditioning refuses only after every
        // earlier clip has rendered. A non-zero tail is `≥ step + 1` on its
        // own grid rule and already implies `needed > step + 1`, so the
        // floor engages for tail 0 alone; the `clip_frames` cap keeps a
        // degenerate 1-frame-clip request inside its own envelope.
        let last = last.max((step + 1).min(clip_frames));
        (count, last)
    };

    let count_usize = stage_count as usize;
    if count_usize > MAX_CHAIN_STAGES {
        return Err(MoldError::Validation(format!(
            "auto-expand would produce {stage_count} stages; maximum is {MAX_CHAIN_STAGES} \
             (try reducing total_frames or increasing clip_frames)",
        )));
    }

    let mut stages = Vec::with_capacity(count_usize);
    for idx in 0..stage_count {
        // LTX-2 / LTX-2.3 consume the repeated starting image as a soft
        // identity anchor alongside the motion tail; the LTX-2.5 stage
        // renderer drops it again on continuations because that
        // keyframe-trained checkpoint reads the appended token as a keyframe
        // target (`ltx2::pipeline::route_continuation_opening_images`) — the
        // repeat is harmless there. Wan has no such append path at all: a
        // stage-local source image is its hard frame-0 authority, and its
        // renderer deliberately refuses to replace one with the preceding
        // stage's tail. Repeating the opening still therefore restarted every
        // auto-expanded Wan continuation from the same frame instead of
        // producing the promised seamless handoff.
        let stage_source = if family == Some("wan") && idx > 0 {
            None
        } else {
            source_image.clone()
        };
        let frames = if idx + 1 == stage_count {
            last_stage_frames
        } else {
            clip_frames
        };
        stages.push(ChainStage {
            prompt: prompt.to_string(),
            frames,
            source_image: stage_source,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Smooth,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        });
    }
    Ok(stages)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-model routing clip size is the size ONE generation renders when
    /// mold auto-chains — deliberately smaller than the family's
    /// single-request ceiling.
    #[test]
    fn routing_clip_frames_is_the_per_model_clip_size() {
        assert_eq!(
            routing_clip_frames("ltx2", "ltx-2-19b-distilled:fp8"),
            Some(LTX2_DEFAULT_CLIP_FRAMES),
        );
        assert_eq!(LTX2_DEFAULT_CLIP_FRAMES, 97);
        // Opaque catalog ids stay on the family answer.
        assert_eq!(routing_clip_frames("ltx2", "cv:3143864"), Some(97));
        // ltx-video's chain capability publishes a flat 97-frame clip cap.
        assert_eq!(
            routing_clip_frames("ltx-video", "ltx-video-0.9.6:bf16"),
            Some(97),
        );
        // wan is per checkpoint: the two-expert A14B pair measures near the
        // 24 GB envelope well before the single-expert 5B does.
        assert_eq!(
            routing_clip_frames("wan", "wan22-t2v-a14b:q5"),
            Some(wan_default_clip_frames("wan22-t2v-a14b:q5")),
        );
        assert_eq!(routing_clip_frames("wan", "wan22-ti2v-5b:fp16"), Some(121));
        // Non-chain-capable families have no routing clip size at all.
        assert_eq!(routing_clip_frames("flux", "flux-dev:q4"), None);
        assert_eq!(routing_clip_frames("", "whatever"), None);
    }

    /// wan's routing size is the tier's own recorded default over the family
    /// floor, never a flat constant (#776 item 4).
    #[test]
    fn wan_default_clip_frames_takes_the_tier_default_over_the_floor() {
        // A14B floor.
        assert!(wan_default_clip_frames("wan22-t2v-a14b:q8") >= 53);
        assert_eq!(wan_default_clip_frames("cv:unknown-a14b-thing"), 53);
        // Single-expert floor for everything else.
        assert_eq!(wan_default_clip_frames("cv:2041121"), 121);
        assert!(wan_default_clip_frames("wan22-ti2v-5b:fp16") >= 121);
        // Every answer sits on wan's own `4k+1` grid, so a clip started at the
        // routing default is submittable.
        for model in [
            "wan22-t2v-a14b:q5",
            "wan22-t2v-a14b:q8",
            "wan22-ti2v-5b:fp16",
            "wan21-t2v-1.3b:bf16",
            "cv:2041121",
        ] {
            let frames = wan_default_clip_frames(model);
            assert_eq!(
                (frames - 1) % crate::validation::WAN_TEMPORAL_SCALE,
                0,
                "{model}: routing default {frames} is off wan's 4k+1 grid",
            );
        }
    }

    /// The routing clip size is NOT the family's single-request ceiling: the
    /// whole point of the split is that one generation renders 97 LTX-2 frames
    /// where the family budget admits 481 at 24 fps.
    #[test]
    fn routing_clip_frames_is_below_the_family_ceiling() {
        let family_cap = crate::validation::max_frames_for_family_at_fps("ltx2", 24).unwrap();
        assert!(
            routing_clip_frames("ltx2", "ltx-2-19b-distilled:fp8").unwrap() < family_cap,
            "the routing clip size must stay below the family's single-request ceiling",
        );
    }

    /// An installed catalog wan checkpoint normalises on wan's grid (#783).
    ///
    /// `normalise` resolved the family from the built-in manifest alone, and
    /// `find_manifest` cannot classify a `cv:` / `hf:` id — so the grid fell
    /// back to `8k+1` and a 53-frame wan stage was rejected with "this family
    /// requires 8k+1", immediately after the server had correctly resolved it
    /// as wan from the sidecar overlay. Catalog-installed checkpoints are
    /// exactly the models the sequence work targets.
    #[test]
    fn an_installed_catalog_wan_checkpoint_normalises_on_wans_grid() {
        let installed_wan = || ChainRequest {
            model: "cv:2041121".into(),
            stages: vec![
                wan_stage("a paper boat drifting down a rain gutter", 53),
                wan_stage("the boat reaches a storm drain", 53),
            ],
            motion_tail_frames: 1,
            width: 832,
            height: 480,
            fps: 16,
            ..auto_expand_request("unused", 106, 53, 1, None)
        };

        // Without the hint the id is opaque and the LTX grid rejects it —
        // this is the defect, kept explicit so the fix cannot silently lapse.
        let unhinted = installed_wan().normalise();
        assert!(
            unhinted.is_err(),
            "a `cv:` id has no manifest, so an unhinted normalise still cannot know the grid"
        );

        // The server and CLI both resolve the family before calling, so the
        // hint is what they actually have in hand.
        let normalised = installed_wan()
            .normalise_with_family(Some("wan"))
            .expect("53 is 4k+1, which is wan's own grid");
        assert_eq!(normalised.stages.len(), 2);
        assert!(normalised.stages.iter().all(|stage| stage.frames == 53));
        assert_eq!(normalised.motion_tail_frames, 1);

        // Wan's grid is still enforced, just wan's and not LTX-2's.
        let off_grid = ChainRequest {
            stages: vec![wan_stage("one", 50), wan_stage("two", 50)],
            ..installed_wan()
        };
        let error = off_grid.normalise_with_family(Some("wan")).unwrap_err();
        assert!(error.to_string().contains("4k+1"), "got: {error}");

        // An explicit hint never overrides a model the manifest does know.
        let ltx2 = auto_expand_request("a drone shot", 194, 97, 17, None);
        assert!(ltx2.normalise_with_family(Some("ltx2")).is_ok());
    }

    fn wan_stage(prompt: &str, frames: u32) -> ChainStage {
        ChainStage {
            prompt: prompt.into(),
            frames,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Smooth,
            fade_frames: None,
            model: None,
            loras: Vec::new(),
            references: Vec::new(),
        }
    }

    /// Build a minimal auto-expand request with the given knobs. All other
    /// fields use their v1 defaults so tests can focus on the logic under
    /// exercise.
    fn auto_expand_request(
        prompt: &str,
        total_frames: u32,
        clip_frames: u32,
        motion_tail_frames: u32,
        source_image: Option<Vec<u8>>,
    ) -> ChainRequest {
        ChainRequest {
            collection: None,
            tags: None,
            title: None,
            model: "ltx-2-19b-distilled:fp8".into(),
            stages: Vec::new(),
            motion_tail_frames,
            width: 1216,
            height: 704,
            fps: 24,
            seed: Some(42),
            steps: 8,
            guidance: 3.0,
            strength: 1.0,
            output_format: OutputFormat::Mp4,
            placement: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
            prompt: Some(prompt.into()),
            total_frames: Some(total_frames),
            clip_frames: Some(clip_frames),
            source_image,
            enable_audio: None,
            ephemeral: false,
        }
    }

    fn canonical_request(stages: Vec<ChainStage>, motion_tail_frames: u32) -> ChainRequest {
        ChainRequest {
            collection: None,
            tags: None,
            title: None,
            model: "ltx-2-19b-distilled:fp8".into(),
            stages,
            motion_tail_frames,
            width: 1216,
            height: 704,
            fps: 24,
            seed: Some(42),
            steps: 8,
            guidance: 3.0,
            strength: 1.0,
            output_format: OutputFormat::Mp4,
            placement: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
            ephemeral: false,
        }
    }

    fn make_stage(frames: u32) -> ChainStage {
        ChainStage {
            prompt: "test".into(),
            frames,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Smooth,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        }
    }

    #[test]
    fn normalise_splits_single_prompt_into_stages() {
        // total=400, clip=97, tail=9 → effective=88, remainder=303,
        // N = 1 + ceil(303/88) = 1 + 4 = 5 stages. The last stage is
        // exact-fitted (#1509): the first four deliver 97 + 3*88 = 361, so
        // it needs 9 + 39 = 48 frames — off the 8k+1 grid, so it rounds up
        // to 49 and the stitched total is 401, one frame over the (equally
        // off-grid) request instead of a full clip's 48.
        let normalised = auto_expand_request("a cat walking", 400, 97, 9, None)
            .normalise()
            .expect("normalise should succeed");

        assert_eq!(
            normalised
                .stages
                .iter()
                .map(|stage| stage.frames)
                .collect::<Vec<_>>(),
            vec![97, 97, 97, 97, 49],
            "400/97 with a 9-frame motion tail should expand to 5 stages \
             with an exact-fitted last stage",
        );
        assert_eq!(normalised.estimated_total_frames(), 401);
        for stage in &normalised.stages {
            assert_eq!(stage.prompt, "a cat walking");
            assert!(stage.seed_offset.is_none());
        }
        // Auto-expand fields are cleared post-normalisation.
        assert!(normalised.prompt.is_none());
        assert!(normalised.total_frames.is_none());
        assert!(normalised.clip_frames.is_none());
        assert!(normalised.source_image.is_none());
    }

    /// #1509: an auto-expanded chain delivers the requested total EXACTLY
    /// when the lattice closes — the last stage shrinks instead of rendering
    /// a full clip of undisclosed extra frames.
    #[test]
    fn auto_expand_exact_fits_the_last_stage_to_the_requested_total() {
        // LTX: total=145, clip=97, tail=17 → the last stage needs
        // 17 + (145 - 97) = 65 (8k+1 ✓), so the chain is [97, 65] and the
        // stitch is 97 + 48 = 145 — not the [97, 97] → 177 it used to be.
        let normalised = auto_expand_request("a cat walking", 145, 97, 17, None)
            .normalise()
            .expect("normalise should succeed");
        assert_eq!(
            normalised
                .stages
                .iter()
                .map(|stage| stage.frames)
                .collect::<Vec<_>>(),
            vec![97, 65],
        );
        assert_eq!(normalised.estimated_total_frames(), 145);
    }

    /// The issue's own repro (#1509): wan22-ti2v-5b at `--frames 145`
    /// rendered 241 frames because both stages took the full 121-frame clip.
    /// Wan's lattice closes on its `4k+1` grid with the 1-frame handoff, so
    /// the fit must be exact there too.
    #[test]
    fn auto_expand_exact_fits_wan_totals_on_the_4k1_grid() {
        let mut request = auto_expand_request("waves on a beach", 145, 121, 1, None);
        request.model = "hf:opaque-wan-i2v".into();
        request.width = 832;
        request.height = 480;
        let normalised = request
            .normalise_with_family(Some("wan"))
            .expect("wan auto-expand should succeed");
        assert_eq!(
            normalised
                .stages
                .iter()
                .map(|stage| stage.frames)
                .collect::<Vec<_>>(),
            vec![121, 25],
        );
        assert_eq!(normalised.estimated_total_frames(), 145);

        // Multi-continuation: 201 @ clip 73 / tail 1 → [73, 73, 57],
        // 73 + 72 + 56 = 201 (was [73, 73, 73] → 219, +9%).
        let mut request = auto_expand_request("waves on a beach", 201, 73, 1, None);
        request.model = "hf:opaque-wan-i2v".into();
        request.width = 832;
        request.height = 480;
        let normalised = request
            .normalise_with_family(Some("wan"))
            .expect("wan auto-expand should succeed");
        assert_eq!(
            normalised
                .stages
                .iter()
                .map(|stage| stage.frames)
                .collect::<Vec<_>>(),
            vec![73, 73, 57],
        );
        assert_eq!(normalised.estimated_total_frames(), 201);
    }

    /// A zero-tail request one frame past the clip would exact-fit to a
    /// 1-frame last stage — on-grid, above the tail, and refused by the wan
    /// conditioning path only after every earlier clip has rendered. The
    /// fit floors the last stage at the smallest non-degenerate grid value
    /// instead; the extra frames are disclosed like any other residual.
    #[test]
    fn auto_expand_floors_a_degenerate_last_stage() {
        // LTX grid: total = 98 @ clip 97 / tail 0 → needed 1, floored to 9.
        let normalised = auto_expand_request("x", 98, 97, 0, None)
            .normalise()
            .expect("normalise should succeed");
        assert_eq!(
            normalised
                .stages
                .iter()
                .map(|stage| stage.frames)
                .collect::<Vec<_>>(),
            vec![97, 9],
        );
        assert_eq!(normalised.estimated_total_frames(), 106);

        // Wan grid: an opaque checkpoint whose contract could not be probed
        // gets tail 0 from the router; total = 122 @ clip 121 → needed 1,
        // floored to 5 — the 2-latent-frame minimum WanTi2vInpaint accepts.
        let mut request = auto_expand_request("x", 122, 121, 0, None);
        request.model = "hf:opaque-wan".into();
        request.width = 832;
        request.height = 480;
        let normalised = request
            .normalise_with_family(Some("wan"))
            .expect("wan auto-expand should succeed");
        assert_eq!(
            normalised
                .stages
                .iter()
                .map(|stage| stage.frames)
                .collect::<Vec<_>>(),
            vec![121, 5],
        );
        assert_eq!(normalised.estimated_total_frames(), 126);
    }

    /// A layout the lattice cannot close (a zero motion tail makes every
    /// stitched total ≡ stage-count (mod 8), so most totals are unreachable)
    /// still shrinks the last stage to the smallest on-grid clip covering
    /// the request: the overshoot is bounded by the grid step, not by a
    /// whole clip.
    #[test]
    fn auto_expand_bounds_overshoot_by_the_grid_step_when_the_lattice_cannot_close() {
        // tail 0: the last stage needs 150 - 97 = 53 frames, which is off
        // the 8k+1 grid; it rounds up to 57 → [97, 57] delivering 154.
        // Four extra frames, not the 44 of a full 97-frame stage.
        let normalised = auto_expand_request("x", 150, 97, 0, None)
            .normalise()
            .expect("normalise should succeed");
        assert_eq!(
            normalised
                .stages
                .iter()
                .map(|stage| stage.frames)
                .collect::<Vec<_>>(),
            vec![97, 57],
        );
        assert_eq!(normalised.estimated_total_frames(), 154);
    }

    #[test]
    fn normalise_preserves_ltx2_starting_image_across_all_stages() {
        let png = vec![0x89, 0x50, 0x4e, 0x47, 0xde, 0xad, 0xbe, 0xef];
        let normalised = auto_expand_request("test", 200, 97, 9, Some(png.clone()))
            .normalise()
            .expect("normalise should succeed");

        assert!(normalised.stages.len() >= 2);
        for (idx, stage) in normalised.stages.iter().enumerate() {
            // Every stage must carry the starting image. Stage 0 uses it
            // as the i2v replacement at frame 0; continuations use it as a
            // soft identity anchor through the append path so scene and
            // subject identity stay coherent past the motion-tail window.
            assert_eq!(
                stage.source_image.as_deref(),
                Some(png.as_slice()),
                "stage {idx} must carry the starting image for cross-stage identity anchoring",
            );
        }
    }

    #[test]
    fn normalise_seeds_only_the_first_wan_stage_from_the_opening_image() {
        let png = vec![0x89, 0x50, 0x4e, 0x47, 0xde, 0xad, 0xbe, 0xef];
        let mut request = auto_expand_request("test", 105, 53, 1, Some(png.clone()));
        request.model = "hf:opaque-wan-i2v".into();
        request.width = 832;
        request.height = 480;
        let normalised = request
            .normalise_with_family(Some("wan"))
            .expect("wan auto-expand should succeed");

        assert_eq!(normalised.stages.len(), 2);
        assert_eq!(
            normalised.stages[0].source_image.as_deref(),
            Some(png.as_slice())
        );
        assert!(
            normalised.stages[1].source_image.is_none(),
            "the continuation must accept Wan's previous-frame seam carry"
        );
    }

    #[test]
    fn normalise_rejects_empty() {
        let mut req = canonical_request(Vec::new(), 9);
        // No auto-expand fields either.
        req.prompt = None;
        req.total_frames = None;

        let err = req.normalise().expect_err("empty chain should fail");
        assert!(
            matches!(err, MoldError::Validation(_)),
            "empty chain should be a validation error, got {err:?}",
        );
    }

    #[test]
    fn normalise_rejects_non_8k1_frames() {
        // Canonical form with a stage whose frames violates the 8k+1
        // constraint.
        let req = canonical_request(vec![make_stage(50)], 9);
        let err = req.normalise().expect_err("non-8k+1 frames should fail");
        assert!(
            matches!(err, MoldError::Validation(msg) if msg.contains("8k+1")),
            "error must mention the 8k+1 constraint",
        );
    }

    #[test]
    fn normalise_accepts_canonical_form_unchanged() {
        // Caller already built stages; normalise should validate and clear
        // the (already-empty) auto-expand fields without touching stages.
        let stages = vec![make_stage(97), make_stage(97), make_stage(97)];
        let normalised = canonical_request(stages.clone(), 9)
            .normalise()
            .expect("valid canonical form should pass");
        assert_eq!(normalised.stages.len(), 3);
        for (left, right) in normalised.stages.iter().zip(stages.iter()) {
            assert_eq!(left.frames, right.frames);
            assert_eq!(left.prompt, right.prompt);
        }
    }

    #[test]
    fn normalise_single_stage_when_total_leq_clip() {
        // total=9 fits in one clip; don't render a full 97-frame stage and
        // throw most of it away. Use motion_tail=1 (smallest valid 1+8k)
        // so the strict-less-than-stage-frames invariant still holds for
        // the lone 9-frame stage.
        let normalised = auto_expand_request("short", 9, 97, 1, None)
            .normalise()
            .expect("short single-clip chain should pass");
        assert_eq!(normalised.stages.len(), 1);
        assert_eq!(normalised.stages[0].frames, 9);
    }

    #[test]
    fn normalise_rejects_too_many_stages() {
        // 17 canonical stages exceeds MAX_CHAIN_STAGES (16).
        let stages = (0..17).map(|_| make_stage(97)).collect();
        let err = canonical_request(stages, 9)
            .normalise()
            .expect_err("17-stage chain should fail");
        assert!(
            matches!(err, MoldError::Validation(msg) if msg.contains("maximum")),
            "error must mention the max-stages cap",
        );
    }

    #[test]
    fn normalise_rejects_auto_expand_too_long() {
        // 16 × 97 = 1552 max stitched frames before trim; asking for
        // 4000 frames should blow the guardrail.
        let err = auto_expand_request("too long", 4000, 97, 9, None)
            .normalise()
            .expect_err("runaway auto-expand should fail");
        assert!(
            matches!(err, MoldError::Validation(msg) if msg.contains("stages")),
            "error must name the stage count guardrail",
        );
    }

    #[test]
    fn normalise_preserves_optional_prepared_batch_provenance() {
        let mut req = auto_expand_request("expanded prompt", 190, 97, 17, None);
        req.original_prompt = Some("source prompt".into());
        req.batch_id = Some("prepared-batch-1".into());
        req.batch_index = Some(2);
        req.batch_count = Some(3);

        let normalised = req.normalise().unwrap();
        assert_eq!(normalised.original_prompt.as_deref(), Some("source prompt"));
        assert_eq!(normalised.batch_id.as_deref(), Some("prepared-batch-1"));
        assert_eq!(normalised.batch_index, Some(2));
        assert_eq!(normalised.batch_count, Some(3));
    }

    #[test]
    fn normalise_rejects_motion_tail_ge_clip() {
        // motion_tail must leave at least one new frame per continuation.
        let err = auto_expand_request("bad tail", 200, 97, 97, None)
            .normalise()
            .expect_err("motion_tail >= clip should fail");
        assert!(
            matches!(err, MoldError::Validation(msg) if msg.contains("motion_tail_frames")),
            "error must name motion_tail_frames",
        );
    }

    #[test]
    fn enable_audio_defaults_to_none_and_round_trips_when_set() {
        // Wire-conservative default: chains opt in to audio explicitly. A
        // request that omits the field stays None (engine-side resolves to
        // false), so existing chain callers don't suddenly get audio they
        // didn't ask for. Setting `enable_audio: true` on the request must
        // round-trip into the canonical script echo so clients can save and
        // re-render the same chain with audio enabled.
        let req: ChainRequest = serde_json::from_value(serde_json::json!({
            "model": "ltx-2.3-22b-distilled:fp8",
            "stages": [],
            "width": 704,
            "height": 416,
            "steps": 4,
            "guidance": 3.0,
        }))
        .expect("valid minimal chain request");
        assert_eq!(req.enable_audio, None);
        assert_eq!(req.original_prompt, None);
        assert_eq!(req.batch_id, None);
        assert_eq!(req.batch_index, None);
        assert_eq!(req.batch_count, None);

        let req_with_audio: ChainRequest = serde_json::from_value(serde_json::json!({
            "model": "ltx-2.3-22b-distilled:fp8",
            "stages": [{"prompt": "a bird", "frames": 33}],
            "width": 704,
            "height": 416,
            "steps": 4,
            "guidance": 3.0,
            "enable_audio": true,
        }))
        .expect("valid chain request with audio");
        assert_eq!(req_with_audio.enable_audio, Some(true));

        let script = ChainScript::from(&req_with_audio);
        assert_eq!(
            script.chain.enable_audio,
            Some(true),
            "ChainScript echo must preserve enable_audio for round-trip save/reload",
        );
    }

    #[test]
    fn motion_tail_default_lands_on_8k_plus_1_grid() {
        // Server JSON default must satisfy `1 + 8k` so chain tail RGB frames
        // re-encode cleanly through the LTX-2 video VAE. CLI and SPA already
        // default to 17; pin the JSON deserialiser to the same value.
        let req: ChainRequest = serde_json::from_value(serde_json::json!({
            "model": "ltx-2.3-22b-distilled:fp8",
            "stages": [],
            "width": 704,
            "height": 416,
            "steps": 4,
            "guidance": 3.0,
        }))
        .expect("valid minimal chain request");
        assert_eq!(req.motion_tail_frames, 17);
        assert!(is_ltx2_frame_count(req.motion_tail_frames));
    }

    #[test]
    fn normalise_rejects_motion_tail_off_grid() {
        // motion_tail_frames=4 is what the JSON default used to be — it does
        // NOT satisfy `1 + 8k`, so the carryover VAE re-encode would fail
        // deep in the engine with a shape mismatch. Reject with a clear
        // message at the wire boundary instead.
        let req = canonical_request(vec![make_stage(33)], 4);
        let err = req
            .normalise()
            .expect_err("motion_tail_frames=4 must be rejected");
        assert!(
            matches!(err, MoldError::Validation(msg) if msg.contains("8k+1")),
            "error must name the 8k+1 grid constraint",
        );
    }

    #[test]
    fn normalise_accepts_motion_tail_zero() {
        // motion_tail=0 means hard concat, no overlap, no carryover encode.
        // Must be valid so cut/fade chains can opt out of the grid entirely.
        let mut second = make_stage(33);
        second.transition = TransitionMode::Cut;
        let req = canonical_request(vec![make_stage(33), second], 0);
        req.normalise().expect("motion_tail=0 must be accepted");
    }

    #[test]
    fn normalise_rejects_missing_total_frames_in_auto_expand() {
        let mut req = canonical_request(Vec::new(), 4);
        req.prompt = Some("missing total".into());
        // total_frames omitted.
        let err = req
            .normalise()
            .expect_err("missing total_frames should fail");
        assert!(
            matches!(err, MoldError::Validation(msg) if msg.contains("total_frames")),
            "error must name total_frames",
        );
    }

    #[test]
    fn is_ltx2_frame_count_matches_8k_plus_1() {
        for valid in [1u32, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97] {
            assert!(
                is_ltx2_frame_count(valid),
                "{valid} should be a valid LTX-2 frame count",
            );
        }
        for invalid in [0u32, 2, 8, 10, 16, 50, 96, 98, 100] {
            assert!(
                !is_ltx2_frame_count(invalid),
                "{invalid} must not pass the 8k+1 check",
            );
        }
    }

    #[test]
    fn chain_progress_event_roundtrips_json_with_snake_case_tags() {
        let cases = [
            (
                ChainProgressEvent::ChainStart {
                    stage_count: 5,
                    estimated_total_frames: 469,
                },
                r#""type":"chain_start""#,
            ),
            (
                ChainProgressEvent::StageStart { stage_idx: 0 },
                r#""type":"stage_start""#,
            ),
            (
                ChainProgressEvent::DenoiseStep {
                    stage_idx: 2,
                    step: 4,
                    total: 8,
                },
                r#""type":"denoise_step""#,
            ),
            (
                ChainProgressEvent::StageDone {
                    stage_idx: 3,
                    frames_emitted: 97,
                },
                r#""type":"stage_done""#,
            ),
            (
                ChainProgressEvent::Stitching { total_frames: 400 },
                r#""type":"stitching""#,
            ),
        ];
        for (event, expected_tag) in cases {
            let json = serde_json::to_string(&event).expect("serialize");
            assert!(
                json.contains(expected_tag),
                "missing snake_case tag {expected_tag} in {json}",
            );
            let roundtrip: ChainProgressEvent = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(roundtrip, event, "roundtrip must preserve payload");
        }
    }

    #[test]
    fn build_stages_math_matches_stitch_budget() {
        // Auto-expand must produce enough stages that the stitch delivers
        // at least `total_frames` pixel frames. Stitch math:
        //   delivered = clip_frames + (N - 1) * (clip_frames - motion_tail)
        let cases = [
            (400u32, 97u32, 9u32, 5u32), // 97 + 4*88 = 449 ≥ 400
            (200, 97, 9, 3),             // 97 + 2*88 = 273 ≥ 200
            (97, 97, 9, 1),              // single clip hits 97 exactly
            (300, 97, 0, 4),             // zero tail, 4*97 = 388 ≥ 300
        ];
        for (total, clip, tail, expected_n) in cases {
            let req = auto_expand_request("m", total, clip, tail, None)
                .normalise()
                .expect("valid auto-expand should normalise");
            assert_eq!(
                req.stages.len() as u32,
                expected_n,
                "expected {expected_n} stages for total={total}, clip={clip}, tail={tail}",
            );
            let delivered = clip + (expected_n - 1) * (clip - tail);
            assert!(
                delivered >= total,
                "{expected_n} stages deliver {delivered} frames but {total} were requested",
            );
        }
    }

    #[test]
    fn transition_mode_serializes_snake_case() {
        assert_eq!(
            serde_json::to_value(TransitionMode::Smooth).unwrap(),
            serde_json::Value::String("smooth".into())
        );
        assert_eq!(
            serde_json::to_value(TransitionMode::Cut).unwrap(),
            serde_json::Value::String("cut".into())
        );
        assert_eq!(
            serde_json::to_value(TransitionMode::Fade).unwrap(),
            serde_json::Value::String("fade".into())
        );
    }

    #[test]
    fn transition_mode_defaults_to_smooth() {
        assert_eq!(TransitionMode::default(), TransitionMode::Smooth);
    }

    #[test]
    fn lora_spec_serializes_minimal() {
        let spec = LoraSpec {
            path: "./style.safetensors".into(),
            scale: 0.8,
            name: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""path":"./style.safetensors""#));
        assert!(json.contains(r#""scale":0.8"#));
        // name omitted
        assert!(!json.contains(r#""name""#));
    }

    #[test]
    fn named_ref_serializes_minimal() {
        let r = NamedRef {
            name: "hero".into(),
            image: vec![0x89, 0x50],
        };
        let json = serde_json::to_string(&r).unwrap();
        // base64-encoded image via the existing base64 helper
        assert!(json.contains(r#""name":"hero""#));
        assert!(json.contains(r#""image":"#));
    }

    #[test]
    fn chain_stage_defaults_are_backcompat() {
        // Parsing a v1-shaped stage (no new fields) yields the same structure
        // with defaults applied.
        let json = r#"{
            "prompt": "a cat",
            "frames": 97
        }"#;
        let stage: ChainStage = serde_json::from_str(json).unwrap();
        assert_eq!(stage.prompt, "a cat");
        assert_eq!(stage.frames, 97);
        assert_eq!(stage.transition, TransitionMode::Smooth);
        assert_eq!(stage.fade_frames, None);
        assert!(stage.model.is_none());
        assert!(stage.loras.is_empty());
        assert!(stage.references.is_empty());
    }

    #[test]
    fn chain_script_projects_from_request() {
        let req = ChainRequest {
            collection: None,
            tags: None,
            title: None,
            model: "ltx-2-19b-distilled:fp8".into(),
            stages: vec![ChainStage {
                prompt: "a".into(),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            }],
            motion_tail_frames: 25,
            width: 1216,
            height: 704,
            fps: 24,
            seed: Some(42),
            steps: 8,
            guidance: 3.0,
            strength: 1.0,
            output_format: OutputFormat::Mp4,
            placement: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
            ephemeral: false,
        };
        let script = ChainScript::from(&req);
        assert_eq!(script.chain.model, "ltx-2-19b-distilled:fp8");
        assert_eq!(script.chain.seed, Some(42));
        assert_eq!(script.stages.len(), 1);
        assert_eq!(script.stages[0].prompt, "a");
    }

    #[test]
    fn chain_stage_roundtrips_all_fields() {
        let stage = ChainStage {
            prompt: "scene".into(),
            frames: 49,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Cut,
            fade_frames: Some(12),
            model: None,
            loras: vec![],
            references: vec![],
        };
        let json = serde_json::to_string(&stage).unwrap();
        let back: ChainStage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.frames, 49);
        assert_eq!(back.transition, TransitionMode::Cut);
        assert_eq!(back.fade_frames, Some(12));
    }

    #[test]
    fn normalise_coerces_stage_0_transition_to_smooth() {
        let mut req = auto_expand_request("a", 97, 97, 25, None);
        req.stages = vec![
            ChainStage {
                prompt: "scene 0".into(),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Cut, // should coerce
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            },
            ChainStage {
                prompt: "scene 1".into(),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Cut, // preserved
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            },
        ];
        let normalised = req.normalise().unwrap();
        assert_eq!(normalised.stages[0].transition, TransitionMode::Smooth);
        assert_eq!(normalised.stages[1].transition, TransitionMode::Cut);
    }

    #[test]
    fn normalise_rejects_reserved_model_field() {
        let mut req = auto_expand_request("a", 97, 97, 25, None);
        req.stages = vec![ChainStage {
            prompt: "x".into(),
            frames: 97,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Smooth,
            fade_frames: None,
            model: Some("flux-dev:q4".into()),
            loras: vec![],
            references: vec![],
        }];
        let err = req.normalise().unwrap_err().to_string();
        assert!(err.contains("reserved for sub-project C"), "got: {err}");
    }

    #[test]
    fn normalise_accepts_valid_per_stage_loras() {
        let mut req = auto_expand_request("a", 97, 97, 25, None);
        req.stages = vec![ChainStage {
            prompt: "x".into(),
            frames: 97,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Smooth,
            fade_frames: None,
            model: None,
            loras: vec![LoraSpec {
                path: "x.safetensors".into(),
                scale: 1.0,
                name: None,
            }],
            references: vec![],
        }];
        let normalised = req.normalise().unwrap();
        assert_eq!(normalised.stages[0].loras[0].path, "x.safetensors");
        let metadata = normalised.stitched_output_metadata(OutputFormat::Mp4, 97, None);
        assert_eq!(
            metadata.chain.unwrap().stages[0].loras,
            normalised.stages[0].loras
        );
    }

    #[test]
    fn normalise_validates_per_stage_loras() {
        let base = auto_expand_request("a", 97, 97, 25, None)
            .normalise()
            .unwrap();

        let mut invalid_path = base.clone();
        invalid_path.stages[0].loras = vec![LoraSpec {
            path: "camera.bin".into(),
            scale: 1.0,
            name: None,
        }];
        let err = invalid_path.normalise().unwrap_err().to_string();
        assert!(
            err.contains("safetensors file or camera-control"),
            "got: {err}"
        );

        let mut invalid_scale = base.clone();
        invalid_scale.stages[0].loras = vec![LoraSpec {
            path: "camera-control:dolly-in".into(),
            scale: 2.1,
            name: Some("Dolly in".into()),
        }];
        let err = invalid_scale.normalise().unwrap_err().to_string();
        assert!(err.contains("must be in range [0.0, 2.0]"), "got: {err}");

        let mut too_many = base;
        too_many.stages[0].loras = (0..5)
            .map(|idx| LoraSpec {
                path: format!("{idx}.safetensors"),
                scale: 1.0,
                name: None,
            })
            .collect();
        let err = too_many.normalise().unwrap_err().to_string();
        assert!(err.contains("four-LoRA stack limit"), "got: {err}");
    }

    fn stage_list_request(stages: Vec<(TransitionMode, u32, Option<u32>)>) -> ChainRequest {
        ChainRequest {
            collection: None,
            tags: None,
            title: None,
            model: "ltx-2-19b-distilled:fp8".into(),
            stages: stages
                .into_iter()
                .map(|(t, f, fl)| ChainStage {
                    prompt: "x".into(),
                    frames: f,
                    source_image: None,
                    negative_prompt: None,
                    seed_offset: None,
                    transition: t,
                    fade_frames: fl,
                    model: None,
                    loras: vec![],
                    references: vec![],
                })
                .collect(),
            motion_tail_frames: 25,
            width: 1216,
            height: 704,
            fps: 24,
            seed: None,
            steps: 8,
            guidance: 3.0,
            strength: 1.0,
            output_format: OutputFormat::Mp4,
            placement: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
            ephemeral: false,
        }
    }

    #[test]
    fn estimated_total_all_smooth() {
        // 3 × 97-frame smooth = 97 + (97-25) + (97-25) = 241
        let req = stage_list_request(vec![
            (TransitionMode::Smooth, 97, None),
            (TransitionMode::Smooth, 97, None),
            (TransitionMode::Smooth, 97, None),
        ]);
        assert_eq!(req.estimated_total_frames(), 241);
    }

    #[test]
    fn estimated_total_with_cut() {
        // 97 + 97 (cut, no trim) + (97-25) (smooth after cut) = 266
        let req = stage_list_request(vec![
            (TransitionMode::Smooth, 97, None),
            (TransitionMode::Cut, 97, None),
            (TransitionMode::Smooth, 97, None),
        ]);
        assert_eq!(req.estimated_total_frames(), 266);
    }

    /// The per-stage boundary math must live in exactly one place:
    /// `stage_contributed_frames`. `estimated_total_frames` sums it, and the
    /// chain-job runner persists it as `frames_emitted` — attribution matches
    /// the persisted wire meaning (a stage followed by a Fade loses its
    /// trailing `fade_len`; a stage entering with Fade keeps its full count).
    #[test]
    fn stage_contributed_frames_sums_to_estimated_total() {
        let req = stage_list_request(vec![
            (TransitionMode::Smooth, 97, None),
            (TransitionMode::Cut, 97, None),
            (TransitionMode::Fade, 97, Some(8)),
            (TransitionMode::Smooth, 89, None),
            (TransitionMode::Fade, 97, None), // default fade len 8
        ]);
        let per_stage: Vec<u32> = req
            .stages
            .iter()
            .enumerate()
            .map(|(idx, stage)| {
                let next = req.stages.get(idx + 1);
                stage_contributed_frames(
                    idx,
                    stage.frames,
                    stage.transition,
                    next.map(|s| s.transition),
                    next.and_then(|s| s.fade_frames),
                    req.motion_tail_frames,
                )
            })
            .collect();
        // Stage 1 is followed by an explicit 8-frame fade (97-8), stage 3 by
        // a default-length fade (89-25 smooth trim, then -8 outgoing fade).
        assert_eq!(per_stage, vec![97, 89, 97, 89 - 25 - 8, 97]);
        assert_eq!(
            per_stage.iter().sum::<u32>(),
            req.estimated_total_frames(),
            "estimated_total_frames must be the sum of stage_contributed_frames",
        );
    }

    #[test]
    fn estimated_total_with_fade() {
        // 97 + 97 + (97 - fade 8) fade consumes from both sides, net -fade_len
        // Actually: fade replaces the trailing fade_len of clip N + leading
        // fade_len of clip N+1 with fade_len blended frames.
        // Emission = sum - 2*fade_len + fade_len = sum - fade_len
        // = 97+97+97 - 8 = 283
        let req = stage_list_request(vec![
            (TransitionMode::Smooth, 97, None),
            (TransitionMode::Cut, 97, None),
            (TransitionMode::Fade, 97, Some(8)),
        ]);
        assert_eq!(req.estimated_total_frames(), 283);
    }

    /// Ported from the CLI's synth_generate_request regression test:
    /// stage-0 prompt / source image / negative prompt must land in the
    /// synthetic request verbatim, not smeared from the request level.
    #[test]
    fn synthetic_generate_request_reads_stages_zero() {
        let mut req = auto_expand_request("stage zero prompt", 190, 97, 17, None);
        req.original_prompt = Some("source prompt".into());
        req.batch_id = Some("prepared-batch-1".into());
        req.batch_index = Some(2);
        req.batch_count = Some(3);
        req.stages = vec![
            ChainStage {
                prompt: "stage zero prompt".into(),
                frames: 97,
                source_image: Some(vec![1, 2, 3, 4]),
                negative_prompt: Some("no cats".into()),
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            },
            ChainStage {
                prompt: "stage one prompt".into(),
                frames: 97,
                source_image: Some(vec![9, 9, 9]),
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Cut,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            },
        ];
        req.prompt = None;
        req.total_frames = None;
        req.clip_frames = None;

        let synth = req.synthetic_generate_request(OutputFormat::Mp4, 190, 24);
        assert_eq!(
            synth.prompt, "stage zero prompt\nstage one prompt",
            "distinct clip prompts are joined, one line per clip",
        );
        assert_eq!(synth.source_image.as_deref(), Some(&[1, 2, 3, 4][..]));
        assert_eq!(synth.negative_prompt.as_deref(), Some("no cats"));
        assert_eq!(synth.model, "ltx-2-19b-distilled:fp8");
        assert_eq!(synth.seed, Some(42));
        assert_eq!(synth.frames, Some(190));
        assert_eq!(synth.enable_audio, None);
        assert_eq!(synth.original_prompt.as_deref(), Some("source prompt"));
        assert_eq!(synth.batch_id.as_deref(), Some("prepared-batch-1"));
        assert_eq!(synth.batch_index, Some(2));
        assert_eq!(synth.batch_count, Some(3));

        let metadata = req.stitched_output_metadata(OutputFormat::Mp4, 190, None);
        assert_eq!(metadata.original_prompt.as_deref(), Some("source prompt"));
        assert_eq!(metadata.batch_id.as_deref(), Some("prepared-batch-1"));
        assert_eq!(metadata.batch_index, Some(2));
        assert_eq!(metadata.batch_count, Some(3));
    }

    /// A sequence has exactly one gallery print, so the chain request's
    /// title and filing land on the stitched output's metadata exactly the
    /// way a one-shot's do — never on an intermediate clip, which never
    /// reaches the gallery at all.
    #[test]
    fn chain_title_and_filing_reach_the_stitched_output_metadata() {
        let mut req = auto_expand_request("a cat", 190, 97, 17, None);
        req.title = Some("Smurf Village".into());
        req.tags = Some(vec!["smurfs".into(), "village".into()]);
        req.collection = Some(crate::CollectionRef::by_name("Sequences"));
        let req = req.normalise().unwrap();

        let synth = req.synthetic_generate_request(OutputFormat::Mp4, 190, 24);
        assert_eq!(synth.title.as_deref(), Some("Smurf Village"));
        assert_eq!(
            synth.tags.as_deref(),
            Some(["smurfs".to_string(), "village".to_string()].as_slice())
        );
        assert_eq!(
            synth.collection,
            Some(crate::CollectionRef::by_name("Sequences"))
        );

        let metadata = req.stitched_output_metadata(OutputFormat::Mp4, 190, None);
        assert_eq!(metadata.title.as_deref(), Some("Smurf Village"));
        assert_eq!(
            metadata.tags.as_deref(),
            Some(["smurfs".to_string(), "village".to_string()].as_slice())
        );
        assert_eq!(metadata.collection.as_deref(), Some("Sequences"));
    }

    /// The three fields are additive: an older client's body omits them and
    /// an untitled, unfiled sequence serializes exactly as before.
    #[test]
    fn chain_title_and_filing_are_additive_on_the_wire() {
        let bare = auto_expand_request("a cat", 190, 97, 17, None);
        assert_eq!(bare.title, None);
        assert_eq!(bare.tags, None);
        assert_eq!(bare.collection, None);
        let wire = serde_json::to_value(&bare).unwrap();
        for key in ["title", "tags", "collection"] {
            assert!(wire.get(key).is_none(), "{key} should be omitted: {wire}");
        }
        let metadata =
            bare.normalise()
                .unwrap()
                .stitched_output_metadata(OutputFormat::Mp4, 190, None);
        assert_eq!(metadata.title, None);
        assert_eq!(metadata.tags, None);
        assert_eq!(metadata.collection, None);
    }

    /// `normalise` is the one gate every chain entry point runs, so a bad
    /// title or tag is refused before a durable job dir is created.
    #[test]
    fn chain_normalise_refuses_an_invalid_title_or_filing() {
        let with = |mutate: fn(&mut ChainRequest)| {
            let mut req = auto_expand_request("a cat", 190, 97, 17, None);
            mutate(&mut req);
            req.normalise()
        };

        assert!(with(|req| req.title = Some("line\nbreak".into())).is_err());
        assert!(
            with(|req| req.title = Some("x".repeat(crate::PRINT_TITLE_MAX_CHARS + 1))).is_err()
        );
        assert!(with(|req| req.tags = Some(vec!["nul\0".into()])).is_err());
        assert!(with(|req| req.collection = Some(crate::CollectionRef::default())).is_err());

        // …and valid values pass through normalise untouched.
        let ok = with(|req| {
            req.title = Some("  Smurf Village  ".into());
            req.tags = Some(vec!["smurfs".into()]);
            req.collection = Some(crate::CollectionRef::by_name("Sequences"));
        })
        .unwrap();
        assert_eq!(ok.title.as_deref(), Some("  Smurf Village  "));
    }

    /// `normalise` materializes the stitched print's filing, not just checks
    /// it. The manifest keeps this request verbatim and
    /// `stitched_output_metadata` embeds it, so a raw spelling left here
    /// would record a different filing than the row receives — and it would
    /// survive every resume of the job.
    #[test]
    fn chain_normalise_materializes_the_filing_into_the_request() {
        let mut req = auto_expand_request("a cat", 190, 97, 17, None);
        req.tags = Some(vec![
            "  Smurfs ".into(),
            "smurfs".into(),
            "".into(),
            " village  green ".into(),
        ]);
        req.collection = Some(crate::CollectionRef::by_name("  Smurf   Village  "));
        let req = req.normalise().unwrap();

        assert_eq!(
            req.tags.as_deref(),
            Some(["Smurfs".to_string(), "village green".to_string()].as_slice())
        );
        assert_eq!(
            req.collection,
            Some(crate::CollectionRef::by_name("Smurf Village"))
        );

        // The stitched print's provenance agrees with what will be applied.
        let metadata = req.stitched_output_metadata(OutputFormat::Mp4, 190, None);
        assert_eq!(
            metadata.tags.as_deref(),
            Some(["Smurfs".to_string(), "village green".to_string()].as_slice())
        );
        assert_eq!(metadata.collection.as_deref(), Some("Smurf Village"));

        // An unfiled chain gains nothing.
        let bare = auto_expand_request("a cat", 190, 97, 17, None)
            .normalise()
            .unwrap();
        assert_eq!(bare.tags, None);
        assert_eq!(bare.collection, None);
    }

    /// A sequence whose clips carry distinct prompts must not record the
    /// whole video under clip 1's prompt alone — the gallery row joins
    /// every clip prompt (one line per clip) so search matches any of them.
    /// Uniform prompts (auto-expanded chains) keep the single prompt.
    #[test]
    fn synthetic_generate_request_joins_distinct_stage_prompts() {
        let uniform = auto_expand_request("one prompt", 190, 97, 17, None)
            .normalise()
            .unwrap();
        assert_eq!(
            uniform
                .synthetic_generate_request(OutputFormat::Mp4, 190, 24)
                .prompt,
            "one prompt"
        );

        let mut distinct = stage_list_request(vec![
            (TransitionMode::Smooth, 97, None),
            (TransitionMode::Smooth, 33, None),
        ]);
        distinct.stages[0].prompt = "kingfisher waits".into();
        distinct.stages[1].prompt = "it lifts off".into();
        assert_eq!(
            distinct
                .synthetic_generate_request(OutputFormat::Mp4, 113, 24)
                .prompt,
            "kingfisher waits\nit lifts off"
        );
    }

    /// Chain outputs must carry structured per-clip provenance so the
    /// Library can trace a sequence back to its clips (and, later, its
    /// durable job). Stage seeds are recorded as decimal strings.
    #[test]
    fn stitched_metadata_records_chain_block_with_stage_provenance() {
        let mut req = stage_list_request(vec![
            (TransitionMode::Smooth, 97, None),
            (TransitionMode::Fade, 33, Some(8)),
        ]);
        req.stages[0].prompt = "opening".into();
        req.stages[1].prompt = "landing".into();

        let seeds = [7u64, u64::MAX];
        let provenance = ChainProvenance {
            chain_job_id: Some("job-123"),
            stage_seeds: Some(&seeds),
        };
        let meta = req.stitched_output_metadata(OutputFormat::Mp4, 122, Some(&provenance));

        assert_eq!(
            meta.output_mode,
            Some(crate::GenerationOutputMode::Sequence)
        );
        assert_eq!(meta.chain_job_id.as_deref(), Some("job-123"));
        let chain = meta.chain.expect("chain block must be present");
        assert_eq!(chain.stage_count, 2);
        assert_eq!(chain.motion_tail_frames, req.motion_tail_frames);
        assert_eq!(chain.stages.len(), 2);
        assert_eq!(chain.stages[0].prompt, "opening");
        assert_eq!(chain.stages[0].frames, 97);
        assert_eq!(chain.stages[0].transition, TransitionMode::Smooth);
        assert_eq!(chain.stages[0].seed.as_deref(), Some("7"));
        assert_eq!(chain.stages[1].prompt, "landing");
        assert_eq!(chain.stages[1].frames, 33);
        assert_eq!(chain.stages[1].transition, TransitionMode::Fade);
        assert_eq!(chain.stages[1].fade_frames, Some(8));
        assert_eq!(
            chain.stages[1].seed.as_deref(),
            Some("18446744073709551615"),
            "u64 seeds are decimal strings on the wire",
        );
    }

    /// Without provenance (legacy shim path, CLI local render) the chain
    /// block is still recorded — job id and seeds simply stay absent.
    #[test]
    fn stitched_metadata_records_chain_block_without_provenance() {
        let req = auto_expand_request("p", 190, 97, 17, None)
            .normalise()
            .unwrap();
        let meta = req.stitched_output_metadata(OutputFormat::Mp4, 190, None);
        assert_eq!(meta.chain_job_id, None);
        let chain = meta.chain.expect("chain block must be present");
        assert_eq!(chain.stage_count as usize, chain.stages.len());
        assert!(chain.stages.iter().all(|stage| stage.seed.is_none()));
    }

    /// The recorded output_format must be the ACTUAL post-fallback
    /// container, not the requested one (a WebP request that fell back to
    /// APNG previously recorded WebP on the server path).
    #[test]
    fn stitched_metadata_records_actual_format_after_fallback() {
        let req = auto_expand_request("p", 190, 97, 17, None)
            .normalise()
            .unwrap();
        let meta = req.stitched_output_metadata(OutputFormat::Apng, 190, None);
        assert_eq!(meta.output_format, Some(OutputFormat::Apng));
        assert_eq!(meta.frames, Some(190));
        assert_eq!(meta.fps, Some(24));
    }

    /// `strength` is only meaningful when the chain starts from a source
    /// image — text-to-video chains must not record a phantom strength
    /// (the server copies previously wrote Some(strength) unconditionally).
    #[test]
    fn stitched_metadata_strength_only_for_img2img_start() {
        let txt2vid = auto_expand_request("p", 190, 97, 17, None)
            .normalise()
            .unwrap();
        assert_eq!(
            txt2vid
                .stitched_output_metadata(OutputFormat::Mp4, 190, None)
                .strength,
            None
        );

        let img2vid = auto_expand_request("p", 190, 97, 17, Some(vec![1, 2, 3]))
            .normalise()
            .unwrap();
        assert_eq!(
            img2vid
                .stitched_output_metadata(OutputFormat::Mp4, 190, None)
                .strength,
            Some(1.0)
        );
    }

    /// Field-parity guard: the stitched metadata must agree with a
    /// hand-derived from_generate_request over the same synthetic request,
    /// so future OutputMetadata fields can't silently diverge.
    #[test]
    fn stitched_metadata_matches_from_generate_request() {
        let req = auto_expand_request("p", 190, 97, 17, None)
            .normalise()
            .unwrap();
        let synth = req.synthetic_generate_request(OutputFormat::Mp4, 190, req.fps);
        let mut expected = OutputMetadata::from_generate_request(
            &synth,
            req.seed.unwrap_or(0),
            None,
            crate::build_info::version_string(),
        );
        // The chain block and authored Sequence mode are the deliberate
        // additions over the synthetic single-clip projection; everything
        // else must stay in lockstep.
        let mut stitched = req.stitched_output_metadata(OutputFormat::Mp4, 190, None);
        assert!(stitched.chain.is_some());
        assert_eq!(
            stitched.output_mode,
            Some(crate::GenerationOutputMode::Sequence)
        );
        stitched.chain = None;
        expected.output_mode = Some(crate::GenerationOutputMode::Sequence);
        assert_eq!(stitched, expected);
    }

    #[test]
    fn explicit_one_shot_mode_survives_normalization_and_stitched_metadata() {
        let mut request = auto_expand_request("p", 190, 97, 17, None);
        request.output_mode = Some(crate::GenerationOutputMode::OneShot);

        let request = request.normalise().unwrap();
        assert_eq!(
            request.output_mode,
            Some(crate::GenerationOutputMode::OneShot)
        );
        assert_eq!(
            request
                .stitched_output_metadata(OutputFormat::Mp4, 190, None)
                .output_mode,
            Some(crate::GenerationOutputMode::OneShot)
        );
    }

    /// The shared Wan surface-parity fixture (#806) also pins the ONE-SHOT
    /// auto-chain refusal, because three doors render it: the CLI router, the
    /// server's `POST /api/chain-jobs` admission, and
    /// `studio/lib/chainRouting.ts` (whose own test reads the same block).
    ///
    /// A wan tier that declares `source_image: Unsupported` has no channel to
    /// hand anything across a clip boundary, so every stage re-derives the
    /// scene from the same prompt and seed: the "long" video is the same clip
    /// repeated with a visible reset at each seam. Chaining it is not a longer
    /// render, it is the same render three times.
    #[test]
    fn text_only_wan_auto_chain_refusal_matches_the_surface_parity_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/wan/surface-parity-v1.json"
        )))
        .expect("fixture parses");
        let block = &fixture["auto_chain"]["text_only_refusal"];
        let template = block["template"].as_str().expect("template");
        let total = block["total_frames"].as_u64().expect("total_frames") as u32;

        let render = |model: &str, clip: u32| {
            template
                .replace("{model}", model)
                .replace("{total_frames}", &total.to_string())
                .replace("{clip_frames}", &clip.to_string())
        };

        for entry in block["refused"].as_array().expect("refused") {
            let model = entry["model"].as_str().expect("model");
            let clip = entry["clip_frames"].as_u64().expect("clip_frames") as u32;

            // The fixture's contract is the manifest's, not a second opinion.
            let manifest =
                crate::manifest::find_manifest(&crate::manifest::resolve_model_name(model))
                    .unwrap_or_else(|| panic!("{model} is in the manifest"));
            assert_eq!(
                manifest.defaults.source_image,
                Some(crate::SourceImageCapability::Unsupported),
                "{model} must declare an unsupported source-image contract"
            );
            assert_eq!(
                manifest.defaults.frames,
                Some(entry["tier_default_frames"].as_u64().expect("tier default") as u32),
                "{model}'s recorded default frame count drifted from the fixture"
            );
            assert_eq!(
                routing_clip_frames("wan", model),
                Some(clip),
                "{model}'s routing clip size drifted from the fixture"
            );

            assert_eq!(
                text_only_wan_auto_chain_refusal(
                    Some("wan"),
                    model,
                    manifest.defaults.source_image,
                    total,
                    clip,
                ),
                Some(render(model, clip)),
                "{model} must refuse a one-shot auto-chain with the fixture's sentence"
            );
            // At or below the clip size there is no chain to refuse.
            assert_eq!(
                text_only_wan_auto_chain_refusal(
                    Some("wan"),
                    model,
                    manifest.defaults.source_image,
                    clip,
                    clip,
                ),
                None,
                "{model} renders its own clip size as one clip"
            );
        }

        for entry in block["chained"].as_array().expect("chained") {
            let model = entry["model"].as_str().expect("model");
            let clip = entry["clip_frames"].as_u64().expect("clip_frames") as u32;
            let manifest =
                crate::manifest::find_manifest(&crate::manifest::resolve_model_name(model))
                    .unwrap_or_else(|| panic!("{model} is in the manifest"));
            assert_eq!(
                manifest.defaults.source_image,
                Some(crate::SourceImageCapability::Optional),
                "{model} must still advertise an image-conditioned contract"
            );
            assert_eq!(routing_clip_frames("wan", model), Some(clip));
            assert_eq!(
                text_only_wan_auto_chain_refusal(
                    Some("wan"),
                    model,
                    manifest.defaults.source_image,
                    total,
                    clip,
                ),
                None,
                "{model} seeds each continuation and must still chain"
            );
        }

        // An unclassified checkpoint — an opaque `cv:` / `hf:` catalog id — is
        // "unknown", never a declared refusal. #783 added wan auto-chaining
        // precisely so those route; refusing them on a guess would undo it.
        assert_eq!(
            text_only_wan_auto_chain_refusal(Some("wan"), "cv:12345", None, total, 121),
            None,
        );
        // The contract only means this on wan. LTX-2 carries latent context
        // across the seam whatever its source-image contract says.
        assert_eq!(
            text_only_wan_auto_chain_refusal(
                Some("ltx2"),
                "ltx-2-19b-distilled:fp8",
                Some(crate::SourceImageCapability::Unsupported),
                total,
                97,
            ),
            None,
        );
        assert_eq!(
            text_only_wan_auto_chain_refusal(
                None,
                "wan21-t2v-1.3b:turbo",
                Some(crate::SourceImageCapability::Unsupported),
                total,
                121,
            ),
            None,
        );
    }
}
