//! Chain orchestration and carryover primitives.
//!
//! Chained video generation renders a multi-stage request as a sequence of
//! per-clip renders and stitches them into one output. To avoid boundary
//! artifacts, the tail of each clip is carried across as decoded RGB frames
//! ([`ChainTail`]) and re-encoded into the next clip's conditioning by the
//! model family's renderer.
//!
//! This module owns the model-agnostic pieces: the per-stage render loop
//! ([`ChainOrchestrator`]), the renderer abstraction
//! ([`ChainStageRenderer`]), and the carryover payload. Family-specific
//! shape math (e.g. the LTX-2 VAE's latent-frame ratio) lives with the
//! family, in `crate::ltx2`.

use crate::audio::NativeAudioTrack;
use crate::chain::stitch::{plan_exr_stage_windows, ExrStageWindow};
use anyhow::{bail, Result};
use image::RgbImage;
use mold_core::chain::{ChainProgressEvent, ChainRequest, ChainStage, TransitionMode};
use mold_core::chain_job::effective_stage_seed;
use mold_core::{GenerateRequest, OutputFormat};
use std::path::PathBuf;

/// Chain-level HDR EXR sidecar configuration, supplied by the CLI's
/// forced-local path. `None` everywhere else keeps ordinary chains untouched.
///
/// An HDR chain is a chained v2v regrade: validation pins `hdr_exr_dir` to
/// `ic_lora_control=hdr`, which needs the IC-LoRA pipeline and an SDR
/// reference video. Carrying only the sidecar directory would LogC3-decode
/// ungraded SDR stages — the "wrongly-graded EXR that looks deliberate"
/// refusal #681 existed to prevent — so this config carries the whole
/// authority: the control id, the resolved adapter LoRA stack, and the
/// reference video every stage slices its own temporal window from.
#[derive(Debug, Clone)]
pub struct ChainHdrConfig {
    /// Directory receiving `frame_%05d.exr`, globally numbered across the
    /// stitched timeline.
    pub exr_dir: PathBuf,
    /// 32-bit float samples instead of the half-float default.
    pub full_float: bool,
    /// Normalized built-in control id (`"hdr"`).
    pub ic_lora_control: String,
    /// Reference video container bytes; each stage conditions on its own
    /// temporal window of this clip via `reference_frame_offset`.
    pub reference_video: Vec<u8>,
    /// Resolved control-adapter LoRA stack, prepended to each stage's own
    /// LoRAs (adapter first, matching the single-clip path).
    pub control_loras: Vec<mold_core::LoraWeight>,
    /// Requested stitched length; EXR windows past it write nothing.
    pub total_frames_cap: Option<u32>,
}

/// Per-stage sidecar view handed to [`ChainStageRenderer::render_stage`].
/// Derived once per run from [`ChainHdrConfig`] by the orchestrator; the
/// renderer stamps it onto its render plan and must not re-derive any of it.
#[derive(Debug, Clone)]
pub struct StageSidecar {
    pub exr_dir: PathBuf,
    pub full_float: bool,
    /// This stage's EXR window on the stitched timeline (cap applied).
    pub window: ExrStageWindow,
    /// Stitched-timeline index of this stage's first *rendered* frame —
    /// where the reference-video slice for this stage begins. Derived from
    /// the uncapped timeline so a capped tail stage still conditions on the
    /// right reference window.
    pub reference_frame_offset: u32,
}

/// Opaque carryover payload handed from one chain stage to the next.
///
/// Holds the last `frames` decoded RGB frames of the emitting stage, not the
/// raw tail latents. The receiving stage re-encodes them fresh through the
/// LTX-2 video VAE so every resulting latent slot has correct causal /
/// continuation semantics in the receiving clip's frame of reference — a
/// direct latent slice from the emitting stage's continuation slots would
/// appear at the receiving stage's position 0/1 with slot-meaning mismatched
/// against the VAE's causal-first-frame convention.
///
/// The VAE encode cost on the receiving side is negligible (≈tens of ms for
/// 17 frames at 704×1216), and it's paid inside a VAE load that's already
/// needed for the source-image anchor path (see the `crate::ltx2` pipeline).
#[derive(Debug, Clone)]
pub struct ChainTail {
    /// Number of *pixel* frames this tail represents (not latent frames).
    /// Clients of [`ChainTail`] work in pixel-frame units because that's
    /// what users think in; the latent-frame count is derived from this
    /// plus the LTX-2 VAE's 8× causal temporal ratio.
    pub frames: u32,

    /// The last `frames` decoded RGB frames of the emitting stage, in
    /// capture order. The receiving stage VAE-encodes this contiguous pixel
    /// window into `tail_latent_frame_count(frames)` latent slots. Each
    /// resulting latent slot then carries correct causal (slot 0, 1 pixel)
    /// or continuation (slots 1+, 8 pixels each) semantics for the receiving
    /// clip's pinned region — monotonic, forward-in-time, no slot meaning
    /// mismatch with the RoPE positions in the receiving clip.
    pub tail_rgb_frames: Vec<RgbImage>,
}

/// Typed error returned by [`ChainOrchestrator::run`].
#[derive(Debug)]
pub enum ChainOrchestratorError {
    /// Request failed pre-flight validation (empty stages, bad motion_tail).
    Invalid(anyhow::Error),
    /// A stage render call returned `Err` mid-chain.
    StageFailed {
        stage_idx: u32,
        elapsed_stages: u32,
        elapsed_ms: u64,
        inner: anyhow::Error,
    },
}

impl std::fmt::Display for ChainOrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(e) => write!(f, "chain validation error: {e:#}"),
            Self::StageFailed {
                stage_idx,
                elapsed_stages,
                inner,
                ..
            } => write!(
                f,
                "chain stage {stage_idx} failed after {elapsed_stages} completed stage(s): {inner:#}"
            ),
        }
    }
}

impl std::error::Error for ChainOrchestratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Invalid(e) | Self::StageFailed { inner: e, .. } => Some(e.as_ref()),
        }
    }
}

// ── Orchestrator: loops stages, drops motion-tail prefix, accumulates frames

/// Per-stage progress events the orchestrator observes from the renderer.
/// The renderer emits these synchronously while a stage is denoising; the
/// orchestrator wraps them with `stage_idx` before forwarding as
/// [`ChainProgressEvent`]s to the chain-level subscriber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageProgressEvent {
    /// Denoise step `step` of `total` completed for the active stage.
    DenoiseStep { step: u32, total: u32 },
}

/// Output of a single stage render: the decoded pixel frames (full clip,
/// before motion-tail trim), the pre-VAE-decode latent tail the next stage
/// needs, the optional rendered audio track, and the wall-clock elapsed time
/// for the render.
#[derive(Debug)]
pub struct StageOutcome {
    pub frames: Vec<RgbImage>,
    pub tail: ChainTail,
    /// Per-stage decoded audio. `None` when the stage's
    /// `GenerateRequest.enable_audio` was false or when the engine renders a
    /// video-only model. Sample-rate and channel count are constant within a
    /// chain (single model, single VAE), so the stitch layer can concatenate
    /// these directly without resampling.
    pub audio: Option<NativeAudioTrack>,
    /// EXR frames this stage's decode wrote, when a sidecar was requested.
    pub hdr_frames_written: Option<usize>,
    pub generation_time_ms: u64,
}

/// Abstraction over "render one chain stage". Production impls: `Ltx2Engine`
/// (latent handoff) and `LtxVideoEngine` (independent-clip fallback); tests inject a fake implementation
/// that fabricates deterministic frames and a synthetic [`ChainTail`]
/// without loading candle weights.
pub trait ChainStageRenderer {
    fn render_stage(
        &mut self,
        stage_req: &GenerateRequest,
        carry: Option<&ChainTail>,
        motion_tail_pixel_frames: u32,
        hdr_sidecar: Option<&StageSidecar>,
        stage_progress: Option<&mut dyn FnMut(StageProgressEvent)>,
    ) -> Result<StageOutcome>;
}

/// Output of an end-to-end chain run.
///
/// The orchestrator no longer trims motion-tail prefixes at run time —
/// that moved into [`super::stitch::StitchPlan::assemble`] so the stitch
/// logic can also implement `Cut` (no trim) and `Fade` (post-stitch alpha
/// blend) on the same per-boundary seam.
#[derive(Debug)]
pub struct ChainRunOutput {
    /// Per-stage frame vectors in stage order, each containing the full
    /// un-trimmed pixel clip emitted by the renderer.
    pub stage_frames: Vec<Vec<RgbImage>>,
    /// Per-stage audio tracks aligned 1:1 with `stage_frames`. `None` for
    /// video-only stages (the v1 chain default and any non-AV model). The
    /// stitch layer pairs each entry with the matching frame block to apply
    /// per-boundary Smooth/Cut/Fade audio concat semantics.
    pub stage_audio: Vec<Option<NativeAudioTrack>>,
    pub stage_count: u32,
    /// Total EXR frames written across all stages when an HDR sidecar was
    /// active; `0` for ordinary chains.
    pub hdr_frames_written: usize,
    pub generation_time_ms: u64,
}

/// Drives the per-stage render loop for a chained generation. Borrows its
/// renderer mutably so the loop can re-enter the engine on the same GPU
/// context across stages.
pub struct ChainOrchestrator<'a, R: ChainStageRenderer + ?Sized> {
    renderer: &'a mut R,
}

impl<'a, R: ChainStageRenderer + ?Sized> ChainOrchestrator<'a, R> {
    pub fn new(renderer: &'a mut R) -> Self {
        Self { renderer }
    }

    /// Run every stage in `req.stages` and return the accumulated frames.
    ///
    /// Behaviour invariants (from the 2026-04-20 sign-off, amended 2026-04-21):
    /// - Per-stage seeds default to the shared `base_seed` so the continuation
    ///   denoise starts from matching noise. Stages can opt in to variation by
    ///   setting `seed_offset`, which XORs into the base seed.
    /// - Stage 0's output is kept whole; continuations drop their leading
    ///   `req.motion_tail_frames` pixel frames because those duplicate the
    ///   prior stage's tail that was threaded back as latent conditioning.
    /// - Mid-chain failure returns the error immediately; partial frames are
    ///   discarded (no partial stitch is ever produced in v1).
    pub fn run(
        &mut self,
        req: &ChainRequest,
        chain_progress: Option<&mut dyn FnMut(ChainProgressEvent)>,
    ) -> std::result::Result<ChainRunOutput, ChainOrchestratorError> {
        self.run_with_hdr(req, chain_progress, None)
    }

    /// [`Self::run`] with an optional HDR EXR sidecar. Kept as a separate
    /// entry point so the many ordinary-chain callers and tests never see
    /// the extra parameter.
    pub fn run_with_hdr(
        &mut self,
        req: &ChainRequest,
        mut chain_progress: Option<&mut dyn FnMut(ChainProgressEvent)>,
        hdr: Option<&ChainHdrConfig>,
    ) -> std::result::Result<ChainRunOutput, ChainOrchestratorError> {
        if req.stages.is_empty() {
            return Err(ChainOrchestratorError::Invalid(anyhow::anyhow!(
                "ChainOrchestrator::run: chain request has no stages"
            )));
        }
        validate_motion_tail(req).map_err(ChainOrchestratorError::Invalid)?;

        // Plan every stage's EXR window up front so an invalid layout (a
        // fade boundary, a clip shorter than its trim) fails before any
        // render burns GPU time. The uncapped pass supplies each stage's
        // stitched-timeline render position, which is where its reference
        // slice begins — a capped tail stage still renders (and conditions)
        // at its real timeline position even when it writes fewer EXRs.
        let stage_sidecars: Option<Vec<StageSidecar>> = match hdr {
            None => None,
            Some(cfg) => {
                let frames: Vec<u32> = req.stages.iter().map(|stage| stage.frames).collect();
                let boundaries: Vec<TransitionMode> =
                    req.stages.iter().skip(1).map(|s| s.transition).collect();
                let capped = plan_exr_stage_windows(
                    &frames,
                    &boundaries,
                    req.motion_tail_frames,
                    cfg.total_frames_cap,
                )
                .map_err(|e| ChainOrchestratorError::Invalid(anyhow::anyhow!(e)))?;
                let uncapped =
                    plan_exr_stage_windows(&frames, &boundaries, req.motion_tail_frames, None)
                        .map_err(|e| ChainOrchestratorError::Invalid(anyhow::anyhow!(e)))?;
                Some(
                    capped
                        .into_iter()
                        .zip(uncapped)
                        .map(|(window, timeline)| StageSidecar {
                            exr_dir: cfg.exr_dir.clone(),
                            full_float: cfg.full_float,
                            window,
                            reference_frame_offset: timeline.start_index - timeline.skip_leading,
                        })
                        .collect(),
                )
            }
        };

        let stage_count = req.stages.len() as u32;
        let estimated_total_frames = estimate_stitched_frames(req);
        if let Some(cb) = chain_progress.as_deref_mut() {
            cb(ChainProgressEvent::ChainStart {
                stage_count,
                estimated_total_frames,
            });
        }

        let base_seed = req.seed.unwrap_or(0);
        let mut stage_frames: Vec<Vec<RgbImage>> = Vec::with_capacity(req.stages.len());
        let mut stage_audio: Vec<Option<NativeAudioTrack>> = Vec::with_capacity(req.stages.len());
        let mut total_generation_ms: u64 = 0;
        let mut hdr_frames_written_total: usize = 0;
        let mut carry: Option<ChainTail> = None;

        for (idx, stage) in req.stages.iter().enumerate() {
            let stage_idx = idx as u32;
            if let Some(cb) = chain_progress.as_deref_mut() {
                cb(ChainProgressEvent::StageStart { stage_idx });
            }

            let stage_seed = derive_stage_seed(base_seed, idx, stage);
            let mut stage_req = build_stage_generate_request(stage, req, stage_seed, idx);
            // The HDR authority overlay: every stage renders as an IC-LoRA
            // regrade of its own temporal window of the reference. The two
            // stage-request builders deliberately keep pinning these fields
            // to None — the orchestrator is the single carrier, so a stage
            // request built anywhere else can never claim HDR by accident.
            if let Some(cfg) = hdr {
                stage_req.pipeline = Some(mold_core::Ltx2PipelineMode::IcLora);
                stage_req.ic_lora_control = Some(cfg.ic_lora_control.clone());
                stage_req.source_video = Some(cfg.reference_video.clone());
                let mut loras = cfg.control_loras.clone();
                loras.extend(stage_req.loras.take().unwrap_or_default());
                stage_req.loras = Some(loras);
            }
            let stage_sidecar = stage_sidecars.as_ref().map(|sidecars| &sidecars[idx]);

            // Cut and Fade transitions produce a visual reset: the prior
            // stage's motion-tail latent is NOT threaded into this stage.
            // Smooth is the only transition that passes carry through.
            // Carry is still captured unconditionally at the end of the loop
            // so a later Smooth stage can pick it up.
            let effective_carry = match stage.transition {
                TransitionMode::Smooth => carry.as_ref(),
                TransitionMode::Cut | TransitionMode::Fade => None,
            };

            // Wrap the chain progress subscriber so per-stage denoise
            // events land on it with `stage_idx` tagged in. The wrapping
            // closure holds a mutable reborrow of the outer callback for
            // just the duration of this call — `render_stage` is
            // synchronous so the reborrow ends before the next iteration.
            let render_result = match chain_progress.as_deref_mut() {
                Some(chain_cb) => {
                    let mut wrapping = |event: StageProgressEvent| match event {
                        StageProgressEvent::DenoiseStep { step, total } => {
                            chain_cb(ChainProgressEvent::DenoiseStep {
                                stage_idx,
                                step,
                                total,
                            });
                        }
                    };
                    self.renderer.render_stage(
                        &stage_req,
                        effective_carry,
                        req.motion_tail_frames,
                        stage_sidecar,
                        Some(&mut wrapping),
                    )
                }
                None => self.renderer.render_stage(
                    &stage_req,
                    effective_carry,
                    req.motion_tail_frames,
                    stage_sidecar,
                    None,
                ),
            };
            let outcome = render_result.map_err(|inner| ChainOrchestratorError::StageFailed {
                stage_idx,
                elapsed_stages: idx as u32,
                elapsed_ms: total_generation_ms,
                inner,
            })?;

            let frames_emitted = outcome.frames.len() as u32;
            stage_frames.push(outcome.frames);
            stage_audio.push(outcome.audio);
            hdr_frames_written_total =
                hdr_frames_written_total.saturating_add(outcome.hdr_frames_written.unwrap_or(0));
            total_generation_ms = total_generation_ms.saturating_add(outcome.generation_time_ms);
            carry = Some(outcome.tail);

            if let Some(cb) = chain_progress.as_deref_mut() {
                cb(ChainProgressEvent::StageDone {
                    stage_idx,
                    frames_emitted,
                });
            }
        }

        if let Some(cb) = chain_progress.as_mut() {
            let total: u32 = stage_frames.iter().map(|s| s.len() as u32).sum();
            cb(ChainProgressEvent::Stitching {
                total_frames: total,
            });
        }

        Ok(ChainRunOutput {
            stage_frames,
            stage_audio,
            stage_count,
            hdr_frames_written: hdr_frames_written_total,
            generation_time_ms: total_generation_ms,
        })
    }
}

fn validate_motion_tail(req: &ChainRequest) -> Result<()> {
    for (idx, stage) in req.stages.iter().enumerate() {
        if req.motion_tail_frames >= stage.frames {
            bail!(
                "motion_tail_frames ({}) must be strictly less than stage {idx}'s frames ({}) \
                 so every continuation emits at least one new frame",
                req.motion_tail_frames,
                stage.frames,
            );
        }
    }
    Ok(())
}

fn estimate_stitched_frames(req: &ChainRequest) -> u32 {
    // delivered = stages[0].frames + Σ (stages[i].frames - motion_tail) for i >= 1
    let tail = req.motion_tail_frames;
    req.stages
        .iter()
        .enumerate()
        .map(|(idx, stage)| {
            if idx == 0 {
                stage.frames
            } else {
                stage.frames.saturating_sub(tail)
            }
        })
        .sum()
}

fn derive_stage_seed(base_seed: u64, _idx: usize, stage: &ChainStage) -> u64 {
    // Keep the seed stable across stages by default. An earlier revision
    // XORed `(idx as u64) << 32` into each stage's seed so the initial
    // noise tensor differed per clip; with the motion tail now re-encoded
    // from the emitting stage's trailing RGB frames (see `ChainTail` +
    // `StagedLatent`) the pinned region is frozen by `video_denoise_mask`
    // anyway, so same-seed noise in the pinned tokens is a no-op, and
    // same-seed noise in the free region lets the continuation settle on a
    // consistent motion profile. Callers who want per-stage variation
    // supply `stage.seed_offset` explicitly.
    effective_stage_seed(base_seed, stage.seed_offset)
}

fn build_stage_generate_request(
    stage: &ChainStage,
    chain: &ChainRequest,
    stage_seed: u64,
    idx: usize,
) -> GenerateRequest {
    GenerateRequest {
        hdr_exr_dir: None,
        hdr_exr_full_float: false,
        guidance_overrides: None,
        sample_shift: None,
        distill_strength_high: None,
        distill_strength_low: None,
        prompt: stage.prompt.clone(),
        negative_prompt: stage.negative_prompt.clone(),
        model: chain.model.clone(),
        width: chain.width,
        height: chain.height,
        steps: chain.steps,
        guidance: chain.guidance,
        seed: Some(stage_seed),
        batch_size: 1,
        // Continuation stages never use the per-chain output_format
        // downstream — the orchestrator decodes to frames regardless —
        // but MP4 is the canonical intermediate for LTX-2.
        output_format: Some(OutputFormat::Mp4),
        embed_metadata: None,
        scheduler: None,
        cfg_plus: None,
        // Every stage carries the starting image. Stage 0 uses it as the
        // i2v replacement at frame 0; continuation stages have their
        // frame-0 slot pinned by the motion-tail carryover latent, so
        // `render_chain_stage` re-routes the staged image into the append
        // path at a non-zero frame with soft strength — turning it into a
        // durable identity anchor rather than a frame-0 replacement.
        source_image: stage.source_image.clone(),
        source_image_name: None,
        edit_images: None,
        references: None,
        // Replacement strength from the chain request is only meaningful
        // for stage 0's frame-0 i2v pin. Continuations override this at
        // `render_chain_stage` time (the anchor uses a lower soft-
        // strength constant there), so the value we plant here is inert
        // on continuations.
        strength: if idx == 0 { chain.strength } else { 1.0 },
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: None,
        original_prompt: chain.original_prompt.clone(),
        prompt_transform: chain.prompt_transform.clone(),
        batch_id: chain.batch_id.clone(),
        batch_index: chain.batch_index,
        batch_count: chain.batch_count,
        lora: None,
        frames: Some(stage.frames),
        fps: Some(chain.fps),
        upscale_model: None,
        gif_preview: false,
        // Chain wire-format default is None ("no preference") which collapses
        // to false here so existing callers don't suddenly start producing
        // audio they didn't ask for. Callers that want chain audio set
        // `chain.enable_audio = Some(true)` explicitly.
        enable_audio: Some(chain.enable_audio.unwrap_or(false)),
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
        loras: (!stage.loras.is_empty()).then(|| {
            stage
                .loras
                .iter()
                .map(|lora| mold_core::LoraWeight {
                    path: lora.path.clone(),
                    scale: lora.scale,

                    expert: None,
                })
                .collect()
        }),
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: chain.placement.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use image::Rgb;
    use mold_core::chain::{ChainStage, LoraSpec, TransitionMode};

    /// Deterministic fake renderer for orchestrator tests. Records every
    /// call so assertions can inspect the per-stage request shape, emits
    /// a solid-color frame block plus a zero-valued latent tail, and
    /// optionally returns errors on pre-configured stage indices.
    struct FakeRenderer {
        calls: Vec<CallRecord>,
        /// If set, fail on the listed stage indices with the given message.
        fail_on: Vec<(usize, String)>,
        /// Per-call override of frame count (default: use stage_req.frames).
        frame_count_override: Option<u32>,
        /// If true, emit one DenoiseStep event per stage so tests can
        /// verify progress forwarding.
        emit_progress: bool,
        /// If true, attach a synthetic NativeAudioTrack to every StageOutcome
        /// so chain-audio plumbing tests can assert the orchestrator captures
        /// per-stage audio without standing up a real LTX-2 vocoder.
        synthesize_audio: bool,
        /// If true, write one marker file per sidecar-window frame via the
        /// real `exr_frame_path` naming so tests can assert the on-disk
        /// sequence tiles the stitched timeline (no holes, no double
        /// writes) without a real VAE decode.
        write_sidecar_markers: bool,
    }

    #[derive(Debug, Clone)]
    struct CallRecord {
        seed: Option<u64>,
        has_source_image: bool,
        has_carry: bool,
        enable_audio: Option<bool>,
        pipeline: Option<mold_core::Ltx2PipelineMode>,
        ic_lora_control: Option<String>,
        has_source_video: bool,
        lora_paths: Vec<String>,
        sidecar: Option<StageSidecar>,
    }

    impl FakeRenderer {
        fn new() -> Self {
            Self {
                calls: Vec::new(),
                fail_on: Vec::new(),
                frame_count_override: None,
                emit_progress: false,
                synthesize_audio: false,
                write_sidecar_markers: false,
            }
        }
    }

    impl ChainStageRenderer for FakeRenderer {
        fn render_stage(
            &mut self,
            stage_req: &GenerateRequest,
            carry: Option<&ChainTail>,
            _motion_tail_pixel_frames: u32,
            hdr_sidecar: Option<&StageSidecar>,
            mut stage_progress: Option<&mut dyn FnMut(StageProgressEvent)>,
        ) -> Result<StageOutcome> {
            let idx = self.calls.len();
            self.calls.push(CallRecord {
                seed: stage_req.seed,
                has_source_image: stage_req.source_image.is_some(),
                has_carry: carry.is_some(),
                enable_audio: stage_req.enable_audio,
                pipeline: stage_req.pipeline,
                ic_lora_control: stage_req.ic_lora_control.clone(),
                has_source_video: stage_req.source_video.is_some(),
                lora_paths: stage_req
                    .loras
                    .as_ref()
                    .map(|loras| loras.iter().map(|lora| lora.path.clone()).collect())
                    .unwrap_or_default(),
                sidecar: hdr_sidecar.cloned(),
            });
            if let Some((_, msg)) = self.fail_on.iter().find(|(stage_idx, _)| *stage_idx == idx) {
                bail!("{msg}");
            }
            let hdr_frames_written = match hdr_sidecar {
                Some(sidecar) if self.write_sidecar_markers => {
                    std::fs::create_dir_all(&sidecar.exr_dir)?;
                    for offset in 0..sidecar.window.write_count {
                        let path = crate::ltx2::exr::exr_frame_path(
                            &sidecar.exr_dir,
                            (sidecar.window.start_index + offset) as usize,
                        );
                        if path.exists() {
                            bail!("sidecar index written twice: {}", path.display());
                        }
                        std::fs::write(&path, b"marker")?;
                    }
                    Some(sidecar.window.write_count as usize)
                }
                Some(sidecar) => Some(sidecar.window.write_count as usize),
                None => None,
            };
            if self.emit_progress {
                if let Some(cb) = stage_progress.as_mut() {
                    cb(StageProgressEvent::DenoiseStep { step: 1, total: 1 });
                }
            }

            let frame_count = self
                .frame_count_override
                .unwrap_or_else(|| stage_req.frames.expect("fake renderer: stage_req.frames"));
            let width = stage_req.width;
            let height = stage_req.height;
            // Colour the frames with the stage index so assertions can
            // verify which stage a frame came from.
            let mut frames = Vec::with_capacity(frame_count as usize);
            for frame_num in 0..frame_count {
                let channel = (idx as u8).wrapping_mul(37).wrapping_add(frame_num as u8);
                frames.push(RgbImage::from_pixel(width, height, Rgb([channel, 0, 0])));
            }

            // Synthesize a 4-pixel-frame tail from the trailing RGB frames
            // so orchestrator tests can assert on the count/shape without
            // loading a real VAE.
            let tail_pixel_frames: u32 = 4;
            let take_from = frames
                .len()
                .saturating_sub(tail_pixel_frames as usize)
                .min(frames.len());
            let tail_rgb_frames = frames[take_from..].to_vec();

            // Synthesize a deterministic per-stage audio track when
            // requested. Sample count is intentionally tied to the stage's
            // pixel-frame count so audio-stitch tests can map sample
            // boundaries back to motion-tail frame counts cleanly.
            let audio = if self.synthesize_audio {
                let samples_per_frame = 100usize;
                let interleaved_samples: Vec<f32> = (0..frame_count as usize * samples_per_frame)
                    .map(|n| ((idx as i32 * 1_000) + n as i32) as f32)
                    .collect();
                Some(NativeAudioTrack {
                    interleaved_samples,
                    sample_rate: 48_000,
                    channels: 2,
                })
            } else {
                None
            };

            Ok(StageOutcome {
                frames,
                tail: ChainTail {
                    frames: tail_pixel_frames,
                    tail_rgb_frames,
                },
                audio,
                hdr_frames_written,
                generation_time_ms: 100,
            })
        }
    }

    fn stage(prompt: &str, frames: u32) -> ChainStage {
        ChainStage {
            prompt: prompt.into(),
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

    fn chain_req(stages: Vec<ChainStage>, motion_tail_frames: u32) -> ChainRequest {
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
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
        }
    }

    #[test]
    fn stage_request_carries_the_ordered_lora_stack() {
        let mut clip = stage("camera move", 97);
        clip.loras = vec![
            LoraSpec {
                path: "camera-control:dolly-in".into(),
                scale: 1.0,
                name: Some("Dolly in".into()),
            },
            LoraSpec {
                path: "/models/style.safetensors".into(),
                scale: 0.6,
                name: None,
            },
        ];
        let chain = chain_req(vec![clip.clone()], 17);
        let request = build_stage_generate_request(&clip, &chain, 42, 0);
        let loras = request.loras.expect("stage LoRAs must reach inference");
        assert_eq!(loras.len(), 2);
        assert_eq!(loras[0].path, "camera-control:dolly-in");
        assert_eq!(loras[0].scale, 1.0);
        assert_eq!(loras[1].path, "/models/style.safetensors");
        assert_eq!(loras[1].scale, 0.6);
    }

    #[test]
    fn chain_orchestrator_emits_full_per_stage_clips() {
        let stages = vec![stage("a", 97), stage("a", 97), stage("a", 97)];
        let req = chain_req(stages, 4);
        let mut renderer = FakeRenderer::new();
        let mut orch = ChainOrchestrator::new(&mut renderer);
        let out = orch.run(&req, None).expect("chain runs");
        // Each stage keeps its full un-trimmed frame count in the per-stage
        // vector. Total across all stages = 97 * 3 = 291.
        let total_frames: usize = out.stage_frames.iter().map(|s| s.len()).sum();
        assert_eq!(total_frames, 97 * 3);
        assert_eq!(out.stage_count, 3);
        assert_eq!(renderer.calls.len(), 3);
        // Stage 0 has no carry; later stages do.
        assert!(!renderer.calls[0].has_carry);
        assert!(renderer.calls[1].has_carry);
        assert!(renderer.calls[2].has_carry);
    }

    #[test]
    fn chain_with_zero_tail_concats_full_clips_without_drop() {
        let stages = vec![stage("a", 97), stage("a", 97)];
        let req = chain_req(stages, 0);
        let mut renderer = FakeRenderer::new();
        let mut orch = ChainOrchestrator::new(&mut renderer);
        let out = orch.run(&req, None).expect("chain runs");
        let total_frames: usize = out.stage_frames.iter().map(|s| s.len()).sum();
        assert_eq!(
            total_frames,
            97 * 2,
            "zero motion tail must keep every frame in each stage's vector",
        );
    }

    #[test]
    fn chain_empty_stages_errors_without_calling_renderer() {
        let req = chain_req(vec![], 4);
        let mut renderer = FakeRenderer::new();
        let mut orch = ChainOrchestrator::new(&mut renderer);
        let err = orch.run(&req, None).expect_err("empty stages must fail");
        match err {
            ChainOrchestratorError::Invalid(inner) => {
                assert!(
                    format!("{inner}").contains("has no stages"),
                    "error must name the missing stages, got: {inner}",
                );
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
        assert!(renderer.calls.is_empty());
    }

    #[test]
    fn chain_fails_closed_mid_chain_discarding_accumulated_frames() {
        // Signed-off decision 2026-04-20: mid-chain failure returns the
        // error immediately and throws away any frames already produced.
        // No partial stitch is ever written to the gallery.
        let stages = vec![stage("a", 97), stage("a", 97), stage("a", 97)];
        let req = chain_req(stages, 4);
        let mut renderer = FakeRenderer::new();
        renderer.fail_on = vec![(1, "simulated GPU OOM on stage 1".into())];
        let mut orch = ChainOrchestrator::new(&mut renderer);
        let err = orch
            .run(&req, None)
            .expect_err("mid-chain failure must bubble up");
        match err {
            ChainOrchestratorError::StageFailed {
                stage_idx: 1,
                elapsed_stages: 1,
                inner,
                ..
            } => {
                assert!(
                    format!("{inner}").contains("simulated GPU OOM"),
                    "inner error must carry the renderer's message, got: {inner}",
                );
            }
            other => panic!("expected StageFailed at stage 1, got {other:?}"),
        }
        // Stage 0 ran (recorded), stage 1 failed (recorded before bail),
        // stage 2 never ran.
        assert_eq!(renderer.calls.len(), 2);
    }

    #[test]
    fn chain_holds_seed_stable_across_stages_by_default() {
        let stages = vec![stage("a", 9), stage("a", 9), stage("a", 9)];
        let mut req = chain_req(stages, 0);
        req.seed = Some(42);
        let mut renderer = FakeRenderer::new();
        renderer.frame_count_override = Some(9);
        let mut orch = ChainOrchestrator::new(&mut renderer);
        orch.run(&req, None).expect("chain runs");
        // The orchestrator used to XOR `(idx as u64) << 32` into each
        // stage's seed so initial noise differed per clip. With the
        // motion-tail pin grounded on a proper causal-first latent,
        // per-stage noise diversity just amplifies drift at the stitch
        // point — same-seed noise stays frozen in the pinned region and
        // produces a more consistent motion profile in the free region.
        assert_eq!(renderer.calls[0].seed, Some(42));
        assert_eq!(renderer.calls[1].seed, Some(42));
        assert_eq!(renderer.calls[2].seed, Some(42));
    }

    #[test]
    fn chain_propagates_source_image_to_every_stage() {
        // Every stage must receive the starting image in its GenerateRequest.
        // Stage 0 uses it as the frame-0 i2v replacement; continuations use
        // it at the engine level as a soft identity anchor (routed through
        // the append path by `Ltx2Engine::render_chain_stage`). Identity
        // drift past the first clip was traced to the prior behaviour of
        // dropping the image on continuations — no long-range identity
        // anchor meant each continuation was anchored only to the drifted
        // last frame of the prior clip, compounding errors stage-over-stage.
        let mut stages = vec![stage("a", 9), stage("a", 9)];
        stages[0].source_image = Some(vec![0x89, 0x50, 0x4e, 0x47]); // PNG magic
        stages[1].source_image = Some(vec![0x89, 0x50, 0x4e, 0x47]);
        let req = chain_req(stages, 0);
        let mut renderer = FakeRenderer::new();
        renderer.frame_count_override = Some(9);
        let mut orch = ChainOrchestrator::new(&mut renderer);
        orch.run(&req, None).expect("chain runs");
        assert!(
            renderer.calls[0].has_source_image,
            "stage 0 must carry source_image (frame-0 i2v replacement)",
        );
        assert!(
            renderer.calls[1].has_source_image,
            "continuation stage must also carry source_image (soft identity anchor)",
        );
    }

    #[test]
    fn chain_forwards_engine_events_with_stage_idx_wrapping() {
        let stages = vec![stage("a", 9), stage("a", 9)];
        let req = chain_req(stages, 0);
        let mut renderer = FakeRenderer::new();
        renderer.frame_count_override = Some(9);
        renderer.emit_progress = true;

        let mut events: Vec<ChainProgressEvent> = Vec::new();
        {
            let mut orch = ChainOrchestrator::new(&mut renderer);
            let mut cb = |e: ChainProgressEvent| events.push(e);
            orch.run(&req, Some(&mut cb)).expect("chain runs");
        }

        // Expected order:
        //   ChainStart, StageStart(0), DenoiseStep(0), StageDone(0),
        //   StageStart(1), DenoiseStep(1), StageDone(1), Stitching
        assert!(matches!(
            events[0],
            ChainProgressEvent::ChainStart { stage_count: 2, .. }
        ));
        assert!(matches!(
            events[1],
            ChainProgressEvent::StageStart { stage_idx: 0 }
        ));
        assert!(matches!(
            events[2],
            ChainProgressEvent::DenoiseStep {
                stage_idx: 0,
                step: 1,
                total: 1
            }
        ));
        assert!(matches!(
            events[3],
            ChainProgressEvent::StageDone {
                stage_idx: 0,
                frames_emitted: 9
            }
        ));
        assert!(matches!(
            events[4],
            ChainProgressEvent::StageStart { stage_idx: 1 }
        ));
        assert!(matches!(
            events[5],
            ChainProgressEvent::DenoiseStep {
                stage_idx: 1,
                step: 1,
                total: 1
            }
        ));
        assert!(matches!(
            events[6],
            ChainProgressEvent::StageDone {
                stage_idx: 1,
                frames_emitted: 9
            }
        ));
        assert!(matches!(
            events[7],
            ChainProgressEvent::Stitching { total_frames: 18 }
        ));
        assert_eq!(events.len(), 8);
    }

    #[test]
    fn chain_rejects_motion_tail_ge_stage_frames_before_running() {
        let stages = vec![stage("a", 9), stage("a", 9)];
        // tail=9 equals stage frames — no net-new content on continuation.
        let req = chain_req(stages, 9);
        let mut renderer = FakeRenderer::new();
        let mut orch = ChainOrchestrator::new(&mut renderer);
        let err = orch.run(&req, None).expect_err("must fail");
        match err {
            ChainOrchestratorError::Invalid(inner) => {
                assert!(
                    format!("{inner}").contains("motion_tail_frames"),
                    "error must name motion_tail_frames, got: {inner}",
                );
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
        // Renderer never gets called because validation runs up-front.
        assert!(renderer.calls.is_empty());
    }

    #[test]
    fn chain_respects_seed_offset_override_when_stage_provides_one() {
        let mut stages = vec![stage("a", 9), stage("a", 9)];
        stages[1].seed_offset = Some(0xDEADBEEF);
        let mut req = chain_req(stages, 0);
        req.seed = Some(100);
        let mut renderer = FakeRenderer::new();
        renderer.frame_count_override = Some(9);
        let mut orch = ChainOrchestrator::new(&mut renderer);
        orch.run(&req, None).expect("runs");
        assert_eq!(renderer.calls[0].seed, Some(100));
        assert_eq!(
            renderer.calls[1].seed,
            Some(100 ^ 0xDEADBEEFu64),
            "seed_offset must XOR into the stable base seed when a stage opts in to variation",
        );
    }

    #[test]
    fn orchestrator_forwards_chain_enable_audio_to_each_stage_request() {
        // The chain wire-format `enable_audio` flag must land on every
        // per-stage GenerateRequest unchanged. Without this, the engine
        // silently renders video-only on every stage even when the chain
        // request asked for audio. Default (None) collapses to Some(false)
        // so existing chain callers don't suddenly start producing audio.
        let stages = vec![stage("a", 9), stage("a", 9), stage("a", 9)];
        let mut req = chain_req(stages, 0);
        req.enable_audio = Some(true);
        let mut renderer = FakeRenderer::new();
        renderer.frame_count_override = Some(9);
        let mut orch = ChainOrchestrator::new(&mut renderer);
        orch.run(&req, None).expect("chain runs");

        for (idx, call) in renderer.calls.iter().enumerate() {
            assert_eq!(
                call.enable_audio,
                Some(true),
                "stage {idx} must inherit chain.enable_audio",
            );
        }
    }

    #[test]
    fn orchestrator_default_enable_audio_resolves_to_false_for_each_stage() {
        // None on the wire = "no preference" = off. Existing chain callers
        // that never set `enable_audio` keep getting video-only output.
        let stages = vec![stage("a", 9), stage("a", 9)];
        let req = chain_req(stages, 0);
        assert_eq!(req.enable_audio, None);
        let mut renderer = FakeRenderer::new();
        renderer.frame_count_override = Some(9);
        let mut orch = ChainOrchestrator::new(&mut renderer);
        orch.run(&req, None).expect("chain runs");
        for call in renderer.calls.iter() {
            assert_eq!(call.enable_audio, Some(false));
        }
    }

    #[test]
    fn orchestrator_collects_per_stage_audio_into_chain_run_output() {
        // When stages emit audio (LTX-2.3 AV path), the orchestrator must
        // expose the per-stage audio tracks in stage_audio so the stitch
        // layer can concatenate them in lockstep with the per-stage video.
        let stages = vec![stage("a", 9), stage("a", 9), stage("a", 9)];
        let mut req = chain_req(stages, 0);
        req.enable_audio = Some(true);
        let mut renderer = FakeRenderer::new();
        renderer.frame_count_override = Some(9);
        renderer.synthesize_audio = true;
        let mut orch = ChainOrchestrator::new(&mut renderer);
        let out = orch.run(&req, None).expect("chain runs");

        assert_eq!(out.stage_audio.len(), 3, "one entry per stage");
        for (idx, audio) in out.stage_audio.iter().enumerate() {
            let track = audio
                .as_ref()
                .unwrap_or_else(|| panic!("stage {idx} must carry an audio track"));
            assert_eq!(track.sample_rate, 48_000);
            assert_eq!(track.channels, 2);
            assert_eq!(track.interleaved_samples.len(), 9 * 100);
            // Sample value encoding: each stage offsets by 1_000 so
            // assertions can confirm stages didn't get crossed in the vec.
            assert_eq!(track.interleaved_samples[0], (idx as i32 * 1_000) as f32);
        }
    }

    #[test]
    fn orchestrator_omits_audio_when_renderer_returns_none() {
        // Video-only chains (the v1 default) still produce a stage_audio vec,
        // but every entry is None so downstream stitch can branch on it.
        let stages = vec![stage("a", 9), stage("a", 9)];
        let req = chain_req(stages, 0);
        let mut renderer = FakeRenderer::new();
        renderer.frame_count_override = Some(9);
        let mut orch = ChainOrchestrator::new(&mut renderer);
        let out = orch.run(&req, None).expect("chain runs");
        assert_eq!(out.stage_audio.len(), 2);
        assert!(out.stage_audio.iter().all(Option::is_none));
    }

    #[test]
    fn chain_run_output_preserves_per_stage_frames() {
        let req = sample_chain_request(3, TransitionMode::Smooth);
        let mut renderer = FakeRenderer::new();
        let mut orch = ChainOrchestrator::new(&mut renderer);
        let out = orch.run(&req, None).unwrap();
        assert_eq!(out.stage_frames.len(), 3);
        // Each stage holds its full un-trimmed frame count.
        assert_eq!(out.stage_frames[0].len() as u32, req.stages[0].frames);
        assert_eq!(out.stage_frames[1].len() as u32, req.stages[1].frames);
        assert_eq!(out.stage_frames[2].len() as u32, req.stages[2].frames);
    }

    #[test]
    fn orchestrator_passes_none_carry_for_cut_transition() {
        let mut renderer = FakeRenderer::new();
        let req = sample_chain_request(3, TransitionMode::Cut);
        let mut orch = ChainOrchestrator::new(&mut renderer);
        orch.run(&req, None).unwrap();
        // Stage 0 always has None; stages 1/2 should also have None because
        // their transition is Cut.
        let has_carry: Vec<bool> = renderer.calls.iter().map(|c| c.has_carry).collect();
        assert_eq!(has_carry, vec![false, false, false]);
    }

    #[test]
    fn orchestrator_passes_some_carry_for_smooth_transition() {
        let mut renderer = FakeRenderer::new();
        let req = sample_chain_request(3, TransitionMode::Smooth);
        let mut orch = ChainOrchestrator::new(&mut renderer);
        orch.run(&req, None).unwrap();
        let has_carry: Vec<bool> = renderer.calls.iter().map(|c| c.has_carry).collect();
        assert_eq!(has_carry, vec![false, true, true]);
    }

    #[test]
    fn mixed_transitions_end_to_end() {
        let mut renderer = FakeRenderer::new();
        let mut req = sample_chain_request(4, TransitionMode::Smooth);
        // sample_chain_request uses motion_tail_frames=0; override so the
        // Smooth boundary trim math has something to trim.
        req.motion_tail_frames = 25;
        req.stages[1].transition = TransitionMode::Smooth;
        req.stages[2].transition = TransitionMode::Cut;
        req.stages[3].transition = TransitionMode::Fade;
        req.stages[3].fade_frames = Some(8);
        let mut orch = ChainOrchestrator::new(&mut renderer);
        let out = orch.run(&req, None).unwrap();

        // Expected frame count after StitchPlan::assemble:
        //   stage 0 kept whole (97)
        //   + stage 1 Smooth (drop motion_tail 25) = 97 - 25 = 72
        //   + stage 2 Cut (keep whole) = 97
        //   + stage 3 Fade (consume 2*fade_len into fade_len, net -fade_len=8)
        //     = 97 - 8 = 89
        //   total = 97 + 72 + 97 + 89 = 355
        let boundaries: Vec<_> = req.stages.iter().skip(1).map(|s| s.transition).collect();
        let fade_lens: Vec<_> = req
            .stages
            .iter()
            .skip(1)
            .map(|s| s.fade_frames.unwrap_or(8))
            .collect();
        let plan = crate::chain::stitch::StitchPlan {
            clips: out.stage_frames,
            boundaries,
            fade_lens,
            motion_tail_frames: req.motion_tail_frames,
        };
        let frames = plan.assemble().unwrap();
        assert_eq!(frames.len(), 355);
    }

    fn hdr_config(dir: &std::path::Path, cap: Option<u32>) -> ChainHdrConfig {
        ChainHdrConfig {
            exr_dir: dir.to_path_buf(),
            full_float: false,
            ic_lora_control: "hdr".to_string(),
            reference_video: vec![0x00, 0x00, 0x00, 0x18, b'f', b't', b'y', b'p'],
            control_loras: vec![mold_core::LoraWeight {
                path: "/models/loras/ltx2-hdr-adapter.safetensors".to_string(),
                scale: 1.0,

                expert: None,
            }],
            total_frames_cap: cap,
        }
    }

    /// The HDR authority overlay must land on every stage request: the
    /// IC-LoRA pipeline, the control id, the shared reference video, and the
    /// adapter LoRA ahead of any per-stage LoRAs. The two stage-request
    /// builders keep pinning these to None — the orchestrator is the single
    /// carrier.
    #[test]
    fn hdr_overlay_reaches_every_stage_request() {
        let dir = tempfile::tempdir().unwrap();
        let mut stages = vec![stage("a", 97), stage("a", 97)];
        stages[1].loras = vec![mold_core::chain::LoraSpec {
            path: "/models/style.safetensors".into(),
            scale: 0.5,
            name: None,
        }];
        let mut req = chain_req(stages, 17);
        req.motion_tail_frames = 17;
        let mut renderer = FakeRenderer::new();
        let mut orch = ChainOrchestrator::new(&mut renderer);
        orch.run_with_hdr(&req, None, Some(&hdr_config(dir.path(), Some(121))))
            .expect("hdr chain runs");

        for (idx, call) in renderer.calls.iter().enumerate() {
            assert_eq!(
                call.pipeline,
                Some(mold_core::Ltx2PipelineMode::IcLora),
                "stage {idx} must render as an IC-LoRA regrade",
            );
            assert_eq!(call.ic_lora_control.as_deref(), Some("hdr"));
            assert!(
                call.has_source_video,
                "stage {idx} must carry the reference"
            );
            assert_eq!(
                call.lora_paths.first().map(String::as_str),
                Some("/models/loras/ltx2-hdr-adapter.safetensors"),
                "adapter LoRA must lead stage {idx}'s stack",
            );
        }
        // Stage 1 keeps its own LoRA behind the adapter.
        assert_eq!(renderer.calls[1].lora_paths.len(), 2);
        assert_eq!(renderer.calls[1].lora_paths[1], "/models/style.safetensors");

        // Windows: [97 @ 0, skip 17 → 24 @ 97 (capped at 121)]. Reference
        // offsets come from the *uncapped* timeline: stage 1 renders from
        // stitched frame 80 even though it writes only 24 EXRs.
        let sidecar0 = renderer.calls[0].sidecar.as_ref().unwrap();
        assert_eq!(sidecar0.window.start_index, 0);
        assert_eq!(sidecar0.window.skip_leading, 0);
        assert_eq!(sidecar0.window.write_count, 97);
        assert_eq!(sidecar0.reference_frame_offset, 0);
        let sidecar1 = renderer.calls[1].sidecar.as_ref().unwrap();
        assert_eq!(sidecar1.window.start_index, 97);
        assert_eq!(sidecar1.window.skip_leading, 17);
        assert_eq!(sidecar1.window.write_count, 24);
        assert_eq!(sidecar1.reference_frame_offset, 80);
    }

    /// End-to-end file-level pin for issue #688's core requirement: the
    /// per-stage sidecar files tile the stitched timeline with no holes and
    /// no index written twice (the FakeRenderer bails on a double write).
    #[test]
    fn hdr_sidecar_files_tile_the_stitched_timeline() {
        let dir = tempfile::tempdir().unwrap();
        let stages = vec![stage("a", 97), stage("a", 41), stage("a", 97)];
        let mut req = chain_req(stages, 17);
        req.motion_tail_frames = 17;
        let mut renderer = FakeRenderer::new();
        renderer.write_sidecar_markers = true;
        let mut orch = ChainOrchestrator::new(&mut renderer);
        let out = orch
            .run_with_hdr(&req, None, Some(&hdr_config(dir.path(), None)))
            .expect("hdr chain runs");

        let boundaries = vec![TransitionMode::Smooth, TransitionMode::Smooth];
        let stitched = crate::chain::stitch::StitchPlan {
            clips: out.stage_frames,
            boundaries,
            fade_lens: vec![0, 0],
            motion_tail_frames: req.motion_tail_frames,
        }
        .assemble()
        .expect("stitch")
        .len();

        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        assert_eq!(
            names.len(),
            stitched,
            "sidecar file count must equal the stitched frame count",
        );
        let expected: Vec<String> = (0..stitched).map(|i| format!("frame_{i:05}.exr")).collect();
        assert_eq!(names, expected, "sequence must be contiguous from zero");
        assert_eq!(out.hdr_frames_written, stitched);
    }

    #[test]
    fn hdr_with_fade_boundary_fails_before_any_render() {
        let dir = tempfile::tempdir().unwrap();
        let mut stages = vec![stage("a", 97), stage("a", 97)];
        stages[1].transition = TransitionMode::Fade;
        stages[1].fade_frames = Some(8);
        let req = chain_req(stages, 17);
        let mut renderer = FakeRenderer::new();
        let mut orch = ChainOrchestrator::new(&mut renderer);
        let err = orch
            .run_with_hdr(&req, None, Some(&hdr_config(dir.path(), None)))
            .expect_err("fade + sidecar must fail");
        assert!(
            matches!(err, ChainOrchestratorError::Invalid(_)),
            "expected Invalid, got {err:?}",
        );
        assert!(
            format!("{err}").contains("fade"),
            "error must name the fade boundary: {err}",
        );
        assert!(renderer.calls.is_empty(), "no stage may render");
    }

    #[test]
    fn ordinary_chains_carry_no_sidecar_and_no_overlay() {
        let stages = vec![stage("a", 97), stage("a", 97)];
        let req = chain_req(stages, 17);
        let mut renderer = FakeRenderer::new();
        let mut orch = ChainOrchestrator::new(&mut renderer);
        let out = orch.run(&req, None).expect("chain runs");
        assert_eq!(out.hdr_frames_written, 0);
        for call in &renderer.calls {
            assert!(call.sidecar.is_none());
            assert_eq!(call.pipeline, None);
            assert_eq!(call.ic_lora_control, None);
            assert!(!call.has_source_video);
        }
    }

    fn sample_chain_request(count: usize, transition: TransitionMode) -> ChainRequest {
        // Use motion_tail_frames=0 so effective=clip_frames=97, which makes
        // build_auto_expand_stages produce exactly `count` stages for
        // total_frames = 97 * count (no ceil rounding complication).
        let req = ChainRequest {
            model: "ltx-2-19b-distilled:fp8".into(),
            stages: Vec::new(),
            motion_tail_frames: 0,
            width: 1216,
            height: 704,
            fps: 24,
            seed: Some(0),
            steps: 8,
            guidance: 3.0,
            strength: 1.0,
            output_format: mold_core::OutputFormat::Mp4,
            placement: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            prompt: Some("x".into()),
            total_frames: Some(97 * count as u32),
            clip_frames: Some(97),
            source_image: None,
            enable_audio: None,
        };
        let mut req = req.normalise().unwrap();
        for s in req.stages.iter_mut().skip(1) {
            s.transition = transition;
        }
        req
    }
}
