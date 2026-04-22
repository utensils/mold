//! LTX-2 chain carryover primitives.
//!
//! Server-side chained video generation stitches multiple per-clip renders
//! into a single output. To avoid a VAE decode → RGB → VAE encode round-trip
//! between clips (which loses information and doubles VAE cost), the tail of
//! each clip is carried across as latent-space tokens and threaded into the
//! next clip's conditioning directly.
//!
//! This module owns the data types and shape math for that handoff. The
//! orchestrator and the `Ltx2Engine::generate_with_carryover` entry point
//! land in sibling commits.
//!
//! See `tasks/render-chain-v1-plan.md` Phase 1.1 for context.

use anyhow::{anyhow, bail, Context, Result};
use candle_core::Tensor;
use image::RgbImage;
use mold_core::chain::{ChainProgressEvent, ChainRequest, ChainStage};
use mold_core::{GenerateRequest, OutputFormat};

use crate::ltx2::model::shapes::SpatioTemporalScaleFactors;

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
/// needed for the source-image anchor path (see pipeline.rs).
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

/// Number of latent frames corresponding to `pixel_frames` pixel frames
/// under the LTX-2 VAE's 8× causal temporal compression. `1` for
/// `1..=8` pixel frames, `2` for `9..=16`, etc. Matches
/// `VideoLatentShape::from_pixel_shape`.
///
/// Panics if `pixel_frames == 0` — a zero-frame tail is nonsensical and
/// would under-flow the formula. Callers must validate upstream.
pub fn tail_latent_frame_count(pixel_frames: u32) -> usize {
    assert!(
        pixel_frames > 0,
        "tail_latent_frame_count: pixel_frames must be > 0",
    );
    let scale = SpatioTemporalScaleFactors::default().time;
    ((pixel_frames as usize - 1) / scale) + 1
}

/// Slice the last `tail_latent_frame_count(pixel_frames)` frames off the
/// time axis of a rank-5 video-latents tensor shaped
/// `[B, C, T, H, W]`.
///
/// The returned tensor is a view/narrow on the input (no copy on candle's
/// current backends) so callers who intend to hand it to a separate engine
/// invocation — which may drop this engine's state and rebuild it — should
/// `.contiguous()` or `.copy()` the result before the original owner goes
/// out of scope.
///
/// Errors if the tensor is not rank-5 or the requested tail exceeds the
/// available time axis — the latter would mean the orchestrator asked for
/// more tail than the stage produced, which indicates a caller bug.
///
/// Kept after the v1.1 decoded-pixel-carryover switch because the utility
/// still reads cleanly from tests and is useful for ad-hoc debugging /
/// future experiments, but the production chain path no longer calls it.
#[allow(dead_code)]
pub fn extract_tail_latents(final_latents: &Tensor, pixel_frames: u32) -> Result<Tensor> {
    let dims = final_latents.dims();
    if dims.len() != 5 {
        return Err(anyhow!(
            "extract_tail_latents: expected rank-5 tensor [B, C, T, H, W], got shape {:?}",
            dims,
        ));
    }
    let time = dims[2];
    let tail = tail_latent_frame_count(pixel_frames);
    if tail > time {
        return Err(anyhow!(
            "extract_tail_latents: tail requests {} latent frames but the stage emitted only {} \
             (pixel_frames={}, tensor shape={:?})",
            tail,
            time,
            pixel_frames,
            dims,
        ));
    }
    let start = time - tail;
    final_latents
        .narrow(2, start, tail)
        .with_context(|| format!("narrow last {tail} latent frames off time axis"))
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
/// needs, and the wall-clock elapsed time for the render.
#[derive(Debug)]
pub struct StageOutcome {
    pub frames: Vec<RgbImage>,
    pub tail: ChainTail,
    pub generation_time_ms: u64,
}

/// Abstraction over "render one chain stage". Production uses the LTX-2
/// engine impl (lands in Phase 1d); tests inject a fake implementation
/// that fabricates deterministic frames and a synthetic [`ChainTail`]
/// without loading candle weights.
pub trait ChainStageRenderer {
    fn render_stage(
        &mut self,
        stage_req: &GenerateRequest,
        carry: Option<&ChainTail>,
        motion_tail_pixel_frames: u32,
        stage_progress: Option<&mut dyn FnMut(StageProgressEvent)>,
    ) -> Result<StageOutcome>;
}

/// Output of an end-to-end chain run: accumulated RGB frames with motion-
/// tail prefix already trimmed on continuations, the number of stages
/// that ran, and the total elapsed render time.
///
/// The orchestrator does *not* trim to a target total frame count or
/// encode the frames into an output video — those are the caller's job
/// (server / CLI). Keeps the orchestrator single-purpose: produce a
/// coherent frame stream from a stages list.
#[derive(Debug)]
pub struct ChainRunOutput {
    pub frames: Vec<RgbImage>,
    pub stage_count: u32,
    pub generation_time_ms: u64,
}

/// Drives the per-stage render loop for a chained generation. Borrows its
/// renderer mutably so the loop can re-enter the engine on the same GPU
/// context across stages.
pub struct Ltx2ChainOrchestrator<'a, R: ChainStageRenderer + ?Sized> {
    renderer: &'a mut R,
}

impl<'a, R: ChainStageRenderer + ?Sized> Ltx2ChainOrchestrator<'a, R> {
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
        mut chain_progress: Option<&mut dyn FnMut(ChainProgressEvent)>,
    ) -> Result<ChainRunOutput> {
        if req.stages.is_empty() {
            bail!("Ltx2ChainOrchestrator::run: chain request has no stages");
        }
        validate_motion_tail(req)?;

        let stage_count = req.stages.len() as u32;
        let estimated_total_frames = estimate_stitched_frames(req);
        if let Some(cb) = chain_progress.as_deref_mut() {
            cb(ChainProgressEvent::ChainStart {
                stage_count,
                estimated_total_frames,
            });
        }

        let base_seed = req.seed.unwrap_or(0);
        let motion_tail_drop = req.motion_tail_frames as usize;
        let mut accumulated_frames: Vec<RgbImage> = Vec::new();
        let mut total_generation_ms: u64 = 0;
        let mut carry: Option<ChainTail> = None;

        for (idx, stage) in req.stages.iter().enumerate() {
            let stage_idx = idx as u32;
            if let Some(cb) = chain_progress.as_deref_mut() {
                cb(ChainProgressEvent::StageStart { stage_idx });
            }

            let stage_seed = derive_stage_seed(base_seed, idx, stage);
            let stage_req = build_stage_generate_request(stage, req, stage_seed, idx);

            // Wrap the chain progress subscriber so per-stage denoise
            // events land on it with `stage_idx` tagged in. The wrapping
            // closure holds a mutable reborrow of the outer callback for
            // just the duration of this call — `render_stage` is
            // synchronous so the reborrow ends before the next iteration.
            let outcome = match chain_progress.as_deref_mut() {
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
                        carry.as_ref(),
                        req.motion_tail_frames,
                        Some(&mut wrapping),
                    )?
                }
                None => self.renderer.render_stage(
                    &stage_req,
                    carry.as_ref(),
                    req.motion_tail_frames,
                    None,
                )?,
            };

            let mut frames = outcome.frames;
            if idx > 0 && motion_tail_drop > 0 {
                if motion_tail_drop >= frames.len() {
                    bail!(
                        "stage {stage_idx}: emitted {} frames but motion_tail_drop={motion_tail_drop} — tail would consume the whole clip",
                        frames.len(),
                    );
                }
                frames.drain(..motion_tail_drop);
            }
            let frames_emitted = frames.len() as u32;
            accumulated_frames.extend(frames);
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
            cb(ChainProgressEvent::Stitching {
                total_frames: accumulated_frames.len() as u32,
            });
        }

        Ok(ChainRunOutput {
            frames: accumulated_frames,
            stage_count,
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
    if let Some(offset) = stage.seed_offset {
        base_seed ^ offset
    } else {
        base_seed
    }
}

fn build_stage_generate_request(
    stage: &ChainStage,
    chain: &ChainRequest,
    stage_seed: u64,
    idx: usize,
) -> GenerateRequest {
    GenerateRequest {
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
        output_format: OutputFormat::Mp4,
        embed_metadata: None,
        scheduler: None,
        // Every stage carries the starting image. Stage 0 uses it as the
        // i2v replacement at frame 0; continuation stages have their
        // frame-0 slot pinned by the motion-tail carryover latent, so
        // `render_chain_stage` re-routes the staged image into the append
        // path at a non-zero frame with soft strength — turning it into a
        // durable identity anchor rather than a frame-0 replacement.
        source_image: stage.source_image.clone(),
        edit_images: None,
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
        original_prompt: None,
        lora: None,
        frames: Some(stage.frames),
        fps: Some(chain.fps),
        upscale_model: None,
        gif_preview: false,
        enable_audio: Some(false), // v1 chain: no audio plumbing yet
        audio_file: None,
        source_video: None,
        keyframes: None,
        pipeline: None,
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: chain.placement.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn tail_latent_frame_count_matches_vae_formula() {
        // Single-frame tail and up to 8 pixel frames fit in 1 latent frame
        // (LTX-2 VAE uses causal first frame + 8× temporal compression).
        for px in [1u32, 2, 4, 8] {
            assert_eq!(tail_latent_frame_count(px), 1, "{px} pixel frames");
        }
        // 9..=16 span 2 latent frames, 17..=24 span 3, etc.
        assert_eq!(tail_latent_frame_count(9), 2);
        assert_eq!(tail_latent_frame_count(16), 2);
        assert_eq!(tail_latent_frame_count(17), 3);
        assert_eq!(tail_latent_frame_count(24), 3);
        // Full-clip tail (97 frames) → 13 latent frames, matching
        // VideoLatentShape::from_pixel_shape under the same VAE ratio.
        assert_eq!(tail_latent_frame_count(97), 13);
    }

    #[test]
    #[should_panic(expected = "pixel_frames must be > 0")]
    fn tail_latent_frame_count_rejects_zero() {
        tail_latent_frame_count(0);
    }

    #[test]
    fn extract_tail_narrows_last_latent_frame_for_4_pixel_frame_tail() {
        // Build a synthetic [1, 2, 3, 1, 1] where channel 0 is the latent-
        // frame index and channel 1 is a sentinel (42, 43, 44) so we can
        // see which frames the narrow returns.
        let data = vec![
            // frame 0
            0.0f32, 42.0, // frame 1
            1.0, 43.0, // frame 2
            2.0, 44.0,
        ];
        // Arrange [B=1, C=2, T=3, H=1, W=1]. `Tensor::from_vec` fills in
        // row-major order — the permute below puts channels on axis 1.
        let raw = Tensor::from_vec(data, (1, 3, 2, 1, 1), &Device::Cpu).expect("build raw tensor");
        // Reshape [1, T, C, H, W] → [1, C, T, H, W]
        let latents = raw
            .permute([0, 2, 1, 3, 4])
            .expect("permute to [B, C, T, H, W]");
        assert_eq!(latents.dims(), &[1, 2, 3, 1, 1]);

        // tail_latent_frame_count(4) = 1 → take the last latent frame only.
        let tail = extract_tail_latents(&latents, 4).expect("extract");
        assert_eq!(tail.dims(), &[1, 2, 1, 1, 1]);
        let values = tail.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            values,
            vec![2.0, 44.0],
            "tail must be the last latent frame (index 2) across all channels",
        );
    }

    #[test]
    fn extract_tail_narrows_two_frames_for_9_pixel_frame_tail() {
        // Simple rank-5 zero tensor with T=3; narrowing the last 2 frames
        // out of 3 is enough to verify the shape without wrestling with
        // permutations again.
        let latents = Tensor::zeros((1, 1, 3, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let tail = extract_tail_latents(&latents, 9).expect("extract");
        assert_eq!(tail.dims(), &[1, 1, 2, 2, 2]);
    }

    #[test]
    fn extract_tail_rejects_rank_4_tensor() {
        let bad = Tensor::zeros((1, 128, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let err = extract_tail_latents(&bad, 4).expect_err("rank 4 must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("rank-5") && msg.contains("T, H, W"),
            "error must identify the rank mismatch, got: {msg}",
        );
    }

    #[test]
    fn extract_tail_rejects_oversize_request() {
        // Tensor has 1 latent frame; asking for a 9-pixel-frame tail needs 2.
        let latents = Tensor::zeros((1, 128, 1, 4, 4), DType::F32, &Device::Cpu).unwrap();
        let err = extract_tail_latents(&latents, 9).expect_err("oversize tail must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("requests 2") && msg.contains("only 1"),
            "error must name the latent-frame mismatch, got: {msg}",
        );
    }

    // ── Orchestrator tests (fake renderer, weight-free) ───────────────

    use image::Rgb;
    use mold_core::chain::ChainStage;

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
    }

    #[derive(Debug, Clone)]
    struct CallRecord {
        seed: Option<u64>,
        has_source_image: bool,
        has_carry: bool,
    }

    impl FakeRenderer {
        fn new() -> Self {
            Self {
                calls: Vec::new(),
                fail_on: Vec::new(),
                frame_count_override: None,
                emit_progress: false,
            }
        }
    }

    impl ChainStageRenderer for FakeRenderer {
        fn render_stage(
            &mut self,
            stage_req: &GenerateRequest,
            carry: Option<&ChainTail>,
            _motion_tail_pixel_frames: u32,
            mut stage_progress: Option<&mut dyn FnMut(StageProgressEvent)>,
        ) -> Result<StageOutcome> {
            let idx = self.calls.len();
            self.calls.push(CallRecord {
                seed: stage_req.seed,
                has_source_image: stage_req.source_image.is_some(),
                has_carry: carry.is_some(),
            });
            if let Some((_, msg)) = self.fail_on.iter().find(|(stage_idx, _)| *stage_idx == idx) {
                bail!("{msg}");
            }
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

            Ok(StageOutcome {
                frames,
                tail: ChainTail {
                    frames: tail_pixel_frames,
                    tail_rgb_frames,
                },
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
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
        }
    }

    #[test]
    fn chain_runs_all_stages_and_drops_tail_prefix_from_continuations() {
        let stages = vec![stage("a", 97), stage("a", 97), stage("a", 97)];
        let req = chain_req(stages, 4);
        let mut renderer = FakeRenderer::new();
        let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
        let out = orch.run(&req, None).expect("chain runs");
        // Stage 0 keeps all 97 frames; each continuation drops the
        // leading 4 frames, so delivered = 97 + 2 * (97 - 4) = 97 + 186 = 283.
        assert_eq!(out.frames.len(), 97 + 93 * 2);
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
        let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
        let out = orch.run(&req, None).expect("chain runs");
        assert_eq!(
            out.frames.len(),
            97 * 2,
            "zero motion tail must keep every frame on continuations",
        );
    }

    #[test]
    fn chain_empty_stages_errors_without_calling_renderer() {
        let req = chain_req(vec![], 4);
        let mut renderer = FakeRenderer::new();
        let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
        let err = orch.run(&req, None).expect_err("empty stages must fail");
        assert!(
            format!("{err}").contains("has no stages"),
            "error must name the missing stages, got: {err}",
        );
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
        let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
        let err = orch
            .run(&req, None)
            .expect_err("mid-chain failure must bubble up");
        assert!(
            format!("{err}").contains("simulated GPU OOM"),
            "error must carry the renderer's message, got: {err}",
        );
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
        let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
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
        let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
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
            let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
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
        let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
        let err = orch.run(&req, None).expect_err("must fail");
        assert!(
            format!("{err}").contains("motion_tail_frames"),
            "error must name motion_tail_frames, got: {err}",
        );
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
        let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
        orch.run(&req, None).expect("runs");
        assert_eq!(renderer.calls[0].seed, Some(100));
        assert_eq!(
            renderer.calls[1].seed,
            Some(100 ^ 0xDEADBEEFu64),
            "seed_offset must XOR into the stable base seed when a stage opts in to variation",
        );
    }
}
