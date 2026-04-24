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
use crate::types::{DevicePlacement, OutputFormat, VideoData};

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

/// Per-stage LoRA adapter spec. **Reserved for sub-project B** — populating
/// this in a request before B lands causes `ChainRequest::normalise` to
/// return 422. Defined now so scripts that round-trip through v1 clients
/// don't drop fields silently.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct LoraSpec {
    pub path: String,
    pub scale: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Per-stage named reference character/style. **Reserved for sub-project
/// B** — populating this causes `ChainRequest::normalise` to return 422.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
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

    // RESERVED for B/C — populated values are rejected by normalise ───
    /// **Reserved for sub-project C.** Populating this in a request
    /// produces 422 in this release.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// **Reserved for sub-project B.** Non-empty values produce 422.
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
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
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
    /// Defaults to `4` for v1 (matches the CLI default). Must be strictly
    /// less than each stage's `frames`.
    #[serde(default = "default_motion_tail_frames")]
    #[schema(example = 4)]
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

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<DevicePlacement>,

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
            },
            stages: req.stages.clone(),
        }
    }
}

/// VRAM feasibility estimate — populated by sub-project D. `None` in this
/// release.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VramEstimate {
    pub worst_case_bytes: u64,
    pub fits: bool,
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
    /// Base64-encoded stitched video bytes (format per `format` field).
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
    /// consumers the final stage count and the target pre-trim frame total
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
    4
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

impl ChainRequest {
    /// Collapse the auto-expand form into a canonical `Vec<ChainStage>` and
    /// validate the result. Called once on the server side immediately after
    /// JSON parsing, before any engine work kicks off.
    ///
    /// Post-conditions on a successful return:
    /// - `self.stages` is non-empty.
    /// - Each stage's `frames` is `8k+1` and `> 0`.
    /// - `self.stages.len() <= MAX_CHAIN_STAGES`.
    /// - All auto-expand fields are `None` (caller must use `self.stages`).
    pub fn normalise(mut self) -> Result<Self> {
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
            let clip_frames = self.clip_frames.unwrap_or(97);
            if clip_frames == 0 {
                return Err(MoldError::Validation(
                    "chain clip_frames must be > 0".into(),
                ));
            }
            if !is_ltx2_frame_count(clip_frames) {
                return Err(MoldError::Validation(format!(
                    "chain clip_frames ({clip_frames}) must be 8k+1 (9, 17, 25, …, 97)",
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
        for (idx, stage) in self.stages.iter().enumerate() {
            if stage.frames == 0 {
                return Err(MoldError::Validation(format!("stage {idx} has 0 frames",)));
            }
            if !is_ltx2_frame_count(stage.frames) {
                return Err(MoldError::Validation(format!(
                    "stage {idx} has {} frames; LTX-2 requires 8k+1 (9, 17, 25, …, 97)",
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

        // Reserved-field rejection (sub-projects B/C).
        for (idx, stage) in self.stages.iter().enumerate() {
            if stage.model.is_some() {
                return Err(MoldError::Validation(format!(
                    "stages[{idx}].model is reserved for sub-project C and not yet supported"
                )));
            }
            if !stage.loras.is_empty() {
                return Err(MoldError::Validation(format!(
                    "stages[{idx}].loras is reserved for sub-project B and not yet supported"
                )));
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

    /// Predicted stitched frame count *before* any top-level `total_frames`
    /// trim. Used by UIs for the footer summary and by the server to size
    /// the final buffer.
    ///
    /// Per-boundary rule:
    /// - smooth: drop leading `motion_tail_frames` of the incoming clip
    /// - cut: no trim
    /// - fade: replace `2 * fade_len` frames (trailing of prior + leading of
    ///   next) with `fade_len` blended frames → net `-fade_len`
    pub fn estimated_total_frames(&self) -> u32 {
        const DEFAULT_FADE_FRAMES: u32 = 8;
        let mut total: u32 = 0;
        for (idx, stage) in self.stages.iter().enumerate() {
            if idx == 0 {
                total += stage.frames;
                continue;
            }
            match stage.transition {
                TransitionMode::Smooth => {
                    total += stage.frames.saturating_sub(self.motion_tail_frames);
                }
                TransitionMode::Cut => {
                    total += stage.frames;
                }
                TransitionMode::Fade => {
                    let fade_len = stage.fade_frames.unwrap_or(DEFAULT_FADE_FRAMES);
                    total += stage.frames.saturating_sub(fade_len);
                }
            }
        }
        total
    }
}

/// Returns `true` iff `n` has the form `8k + 1` for some non-negative integer
/// `k` (1, 9, 17, 25, …). The LTX-2 pipeline has this constraint on pixel
/// frame counts due to the VAE's 8× temporal compression with a causal first
/// frame.
fn is_ltx2_frame_count(n: u32) -> bool {
    n % 8 == 1
}

/// Compute the stage count and per-stage frame allocation for the auto-
/// expand form, matching Phase 1.4's stitch math:
///
/// - Stage 0 contributes `clip_frames` pixel frames.
/// - Each continuation contributes `clip_frames - motion_tail_frames` new
///   frames (the leading `motion_tail_frames` are dropped at stitch time
///   because they duplicate the prior stage's latent tail).
///
/// Returns enough stages so the stitched total reaches at least
/// `total_frames`; over-production is trimmed from the tail at stitch time
/// per the signed-off decision 2026-04-20.
fn build_auto_expand_stages(
    prompt: &str,
    total_frames: u32,
    clip_frames: u32,
    motion_tail_frames: u32,
    source_image: Option<Vec<u8>>,
) -> Result<Vec<ChainStage>> {
    let (stage_count, per_stage_frames) = if total_frames <= clip_frames {
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
        (count, clip_frames)
    };

    let count_usize = stage_count as usize;
    if count_usize > MAX_CHAIN_STAGES {
        return Err(MoldError::Validation(format!(
            "auto-expand would produce {stage_count} stages; maximum is {MAX_CHAIN_STAGES} \
             (try reducing total_frames or increasing clip_frames)",
        )));
    }

    let mut stages = Vec::with_capacity(count_usize);
    for _ in 0..stage_count {
        // Every stage carries the starting image: stage 0 uses it as the
        // i2v replacement at frame 0, and continuation stages use it as a
        // soft identity anchor through the append path (see
        // `Ltx2Engine::render_chain_stage`). Keeping a durable reference
        // across stages is what stops scene/identity drift past the first
        // clip, whose effects were traced in render-chain v1 as the
        // dominant cause of "strange" continuations — the motion tail
        // alone only carries ~0.7 s of pixel context, nowhere near enough
        // for the model to remember the scene across an 8-stage chain.
        stages.push(ChainStage {
            prompt: prompt.to_string(),
            frames: per_stage_frames,
            source_image: source_image.clone(),
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
            prompt: Some(prompt.into()),
            total_frames: Some(total_frames),
            clip_frames: Some(clip_frames),
            source_image,
        }
    }

    fn canonical_request(stages: Vec<ChainStage>, motion_tail_frames: u32) -> ChainRequest {
        ChainRequest {
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
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
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
        // total=400, clip=97, tail=4 → effective=93, remainder=303,
        // N = 1 + ceil(303/93) = 1 + 4 = 5 stages of 97 frames each.
        // Stitched = 97 + 4*93 = 469, which will be trimmed to 400 at
        // stitch time (per the signed-off "trim from tail" decision).
        let normalised = auto_expand_request("a cat walking", 400, 97, 4, None)
            .normalise()
            .expect("normalise should succeed");

        assert_eq!(
            normalised.stages.len(),
            5,
            "400/97 with a 4-frame motion tail should expand to 5 stages",
        );
        for stage in &normalised.stages {
            assert_eq!(stage.frames, 97);
            assert_eq!(stage.prompt, "a cat walking");
            assert!(stage.seed_offset.is_none());
        }
        // Auto-expand fields are cleared post-normalisation.
        assert!(normalised.prompt.is_none());
        assert!(normalised.total_frames.is_none());
        assert!(normalised.clip_frames.is_none());
        assert!(normalised.source_image.is_none());
    }

    #[test]
    fn normalise_preserves_starting_image_across_all_stages() {
        let png = vec![0x89, 0x50, 0x4e, 0x47, 0xde, 0xad, 0xbe, 0xef];
        let normalised = auto_expand_request("test", 200, 97, 4, Some(png.clone()))
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
    fn normalise_rejects_empty() {
        let mut req = canonical_request(Vec::new(), 4);
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
        let req = canonical_request(vec![make_stage(50)], 4);
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
        let normalised = canonical_request(stages.clone(), 4)
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
        // throw most of it away.
        let normalised = auto_expand_request("short", 9, 97, 4, None)
            .normalise()
            .expect("short single-clip chain should pass");
        assert_eq!(normalised.stages.len(), 1);
        assert_eq!(normalised.stages[0].frames, 9);
    }

    #[test]
    fn normalise_rejects_too_many_stages() {
        // 17 canonical stages exceeds MAX_CHAIN_STAGES (16).
        let stages = (0..17).map(|_| make_stage(97)).collect();
        let err = canonical_request(stages, 4)
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
        let err = auto_expand_request("too long", 4000, 97, 4, None)
            .normalise()
            .expect_err("runaway auto-expand should fail");
        assert!(
            matches!(err, MoldError::Validation(msg) if msg.contains("stages")),
            "error must name the stage count guardrail",
        );
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
            (400u32, 97u32, 4u32, 5u32), // 97 + 4*93 = 469 ≥ 400
            (200, 97, 4, 3),             // 97 + 2*93 = 283 ≥ 200
            (97, 97, 4, 1),              // single clip hits 97 exactly
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
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
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
    fn normalise_rejects_reserved_loras_field() {
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
        let err = req.normalise().unwrap_err().to_string();
        assert!(err.contains("reserved for sub-project B"), "got: {err}");
    }

    fn stage_list_request(stages: Vec<(TransitionMode, u32, Option<u32>)>) -> ChainRequest {
        ChainRequest {
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
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
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
}
