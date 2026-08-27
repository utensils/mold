//! CLI-side render-chain orchestration for LTX-2 distilled models.
//!
//! When `mold run --frames N` exceeds the per-clip cap of the selected model,
//! this module takes over from [`super::generate::run`]: it assembles a
//! [`ChainRequest`] from the user's CLI args and either submits it to a
//! running server via [`MoldClient::generate_chain_stream`] or, in `--local`
//! mode, drives an in-process [`mold_inference::chain::ChainOrchestrator`].
//!
//! Both paths funnel through [`encode_and_save`] so stdout piping, gallery
//! save, metadata DB writes, and preview behaviour match the single-clip
//! path byte-for-byte.

use std::io::Write;
use std::time::Duration;

use anyhow::{Context as _, Result};
use colored::Colorize;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use mold_core::chain::{ChainProgressEvent, ChainRequest};
use mold_core::chain_job::ChainJobState;
#[cfg(any(feature = "cuda", feature = "metal"))]
use mold_core::Config;
use mold_core::{MoldClient, OutputFormat, VideoData};

use crate::control::CliContext;
use crate::output::{is_piped, status};
use crate::theme;

/// Default per-clip frame count when auto-chaining an over-long LTX-2
/// request.
///
/// The per-model clip size lives in `mold_core::chain` because the CLI router
/// is no longer its only reader: `/api/capabilities/chain-limits` advertises
/// the same value, so a Studio composer cannot offer a clip the one-shot path
/// would have split. Re-exported here so the CLI's own call sites and tests
/// keep one import surface.
pub use mold_core::chain::{
    routing_clip_frames, wan_default_clip_frames, LTX2_DEFAULT_CLIP_FRAMES,
};

#[cfg(any(feature = "cuda", feature = "metal", test))]
fn local_chain_planning_frames(request: &ChainRequest) -> u32 {
    // Chain stages execute serially on one owner. Admission must budget the
    // largest individual stage, not the stitched total.
    request
        .stages
        .iter()
        .map(|stage| stage.frames)
        .max()
        .unwrap_or(1)
}

/// Outcome of [`decide_chain_routing`]: either the caller should continue
/// down the single-clip path, build a chain with the given settings, or
/// reject the request because the model family can't be chained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainRoutingDecision {
    /// Go through the normal single-clip path; no chaining required.
    SingleClip,
    /// Submit a chain. `clip_frames` is the clamped per-clip cap.
    Chain { clip_frames: u32, motion_tail: u32 },
    /// Model family doesn't support chaining and `frames` exceeds its cap.
    Rejected { reason: String },
}

/// Whether a wan checkpoint carries context across a clip boundary.
///
/// Mirrors `mold_inference::chain::wan_carryover`, which the CLI cannot call
/// (it does not depend on the inference crate). `Required` is the A14B I2V
/// 36-channel concat, `Optional` the TI2V-5B latent inpaint; both can be
/// seeded from the previous clip. `Unsupported` is text-to-video only, and an
/// unclassified checkpoint is "unknown" — never an assumed handoff.
fn wan_carries_context(source_image: Option<mold_core::SourceImageCapability>) -> bool {
    matches!(
        source_image,
        Some(mold_core::SourceImageCapability::Required)
            | Some(mold_core::SourceImageCapability::Optional)
    )
}

/// Pixel frames a wan continuation duplicates from the clip before it.
///
/// The handoff seeds the continuation with the previous clip's final frame, so
/// it re-renders exactly that one frame and the stitch trims exactly one. This
/// is not LTX-2's 17: that is the pixel window its VAE turns into three latent
/// slots of carryover, which wan has no equivalent of. The value lives in
/// `mold-core`, which both this planner and the engine that enforces it read.
const WAN_HANDOFF_DUPLICATED_FRAMES: u32 = mold_core::validation::WAN_HANDOFF_DUPLICATED_FRAMES;

/// Pure decision function — given a model family, the user's requested
/// `frames`, and the optional `--clip-frames` override, decide whether to
/// chain, stay single-clip, or reject.
///
/// The clamp-to-cap behaviour surfaces through the returned `clip_frames`
/// field; callers warn the user via stderr when they had to clamp.
///
/// Deliberately a flat argument list rather than a struct: every argument is
/// an independent fact the caller already has in hand, and the two call sites
/// (`generate.rs` and the tests) read better spelled out than assembled.
#[allow(clippy::too_many_arguments)]
pub fn decide_chain_routing(
    frames: Option<u32>,
    family: Option<&str>,
    model: &str,
    clip_frames_flag: Option<u32>,
    motion_tail: u32,
    fps: u32,
    pipeline: Option<mold_core::Ltx2PipelineMode>,
    // `source_image` is the model's advertised source-image contract. Only wan
    // reads it, and only to size the seam (#783); `None` keeps the
    // conservative independent-clip behaviour, which is what a model with no
    // advertised contract must get.
    source_image: Option<mold_core::SourceImageCapability>,
) -> ChainRoutingDecision {
    let Some(total_frames) = frames else {
        return ChainRoutingDecision::SingleClip;
    };

    // An audio-only pipeline is never chained. Its `frames` is a duration, not
    // a render shape — the per-clip cap exists because one consumer GPU can
    // only hold so many *video* latents, and there are no clips to stitch. The
    // rule lives here rather than at the call site so a second caller cannot
    // reintroduce it: before this guard, `--pipeline t2a --frames 121` became
    // a two-stage 177-frame video render.
    if pipeline.is_some_and(mold_core::Ltx2PipelineMode::is_audio_only) {
        return ChainRoutingDecision::SingleClip;
    }

    // Auto-chaining is a routing behaviour, not the whole chain capability.
    // Every LTX-2 pipeline renders sequence clips, so the old
    // `model.contains("distilled")` test is gone — it also rejected opaque
    // `cv:` / `hf:` catalog IDs the server and the Studio surfaces both
    // accept. Wan joins it (#783): a wan request past the cap used to be
    // rejected outright rather than chained. ltx-video stays deliberately
    // excluded: it has a chain capability but is not auto-chained here.
    let is_wan = family == Some("wan");
    let is_chain_capable = family == Some("ltx2") || is_wan;

    // The model's real single-request ceiling. LTX-2's is a runtime duration,
    // so it depends on fps; other families report a flat frame count.
    let single_clip_cap = family
        .and_then(|family| mold_core::validation::max_frames_for_family_at_fps(family, fps))
        .unwrap_or(LTX2_DEFAULT_CLIP_FRAMES);

    if !is_chain_capable {
        // Non-chainable families: if the requested frame count is within the
        // family's own single-clip budget, stay on the single-clip path and
        // let the engine decide if it's acceptable. Otherwise, reject with
        // a clear message rather than silently over-producing.
        if total_frames <= single_clip_cap {
            return ChainRoutingDecision::SingleClip;
        }
        return ChainRoutingDecision::Rejected {
            reason: format!(
                "model '{model}' does not support chained video generation; \
                 specify --frames <= {single_clip_cap} per clip for this model",
            ),
        };
    }

    // Auto-chaining uses a default clip length, but an explicit --clip-frames
    // may go all the way to the model's real budget so a user can ask for one
    // long coherent clip instead of a stitched sequence.
    //
    // Wan's default is a VRAM envelope rather than a ceiling: the two-expert
    // A14B measures near the 24 GB limit well before its 257-frame request
    // cap, while the single-expert 5B has room for its own shipped 121. Both
    // sit on wan's `4k+1` grid.
    let routing_default = family
        .and_then(|family| routing_clip_frames(family, model))
        .unwrap_or(LTX2_DEFAULT_CLIP_FRAMES);
    let effective_clip_frames = clip_frames_flag
        .unwrap_or(routing_default)
        .min(single_clip_cap);

    if total_frames <= effective_clip_frames {
        return ChainRoutingDecision::SingleClip;
    }

    // Wan has no latent motion tail, so `--motion-tail` (an LTX-2 pixel window
    // sized for its VAE's 8x carryover) does not apply. Its handoff seeds the
    // continuation with the previous clip's final frame, so exactly one frame
    // is duplicated; a text-to-video checkpoint carries nothing and trims
    // nothing.
    let motion_tail = if is_wan {
        if wan_carries_context(source_image) {
            WAN_HANDOFF_DUPLICATED_FRAMES
        } else {
            0
        }
    } else {
        motion_tail
    };

    if motion_tail > 0 && motion_tail >= effective_clip_frames {
        return ChainRoutingDecision::Rejected {
            reason: format!(
                "--motion-tail ({motion_tail}) must be strictly less than \
                 --clip-frames ({effective_clip_frames}) so every continuation \
                 emits at least one new frame",
            ),
        };
    }

    ChainRoutingDecision::Chain {
        clip_frames: effective_clip_frames,
        motion_tail,
    }
}

/// Emit a stderr warning if `--clip-frames` was above the model's cap and
/// got clamped. Returns the effective value (caller should already have it).
pub fn warn_if_clamped(flag: Option<u32>, cap: u32) {
    if let Some(requested) = flag {
        if requested > cap {
            crate::output::status!(
                "{} --clip-frames {} exceeds model cap {}, clamping to {}",
                theme::prefix_warning(),
                requested,
                cap,
                cap,
            );
        }
    }
}

/// Caller-supplied inputs for a chain run, bundled so the remote + local
/// paths can share a single helper without a 20-arg function signature.
#[allow(clippy::too_many_arguments)]
pub struct ChainInputs {
    pub prompt: String,
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f64,
    pub strength: f64,
    pub seed: Option<u64>,
    pub fps: u32,
    pub output_format: OutputFormat,
    pub total_frames: u32,
    pub clip_frames: u32,
    pub motion_tail: u32,
    pub source_image: Option<Vec<u8>>,
    pub placement: Option<mold_core::DevicePlacement>,
    pub enable_audio: Option<bool>,
    pub original_prompt: Option<String>,
    pub batch_id: Option<String>,
    pub batch_index: Option<u32>,
    pub batch_count: Option<u32>,
}

/// Shrink an auto-expanded all-Smooth chain's LAST stage so the stitched
/// output covers `total_frames` exactly — no overshoot to truncate, and, for
/// the HDR sidecar path that requires this, no stage rendering past the end
/// of the reference video.
///
/// The 8k+1 lattice closes by construction for canonical inputs: `total ≡ 1`,
/// stage 0 delivers `clip ≡ 1`, each full continuation delivers
/// `clip − tail ≡ 0`, and `tail ≡ 1 (mod 8)`, so the remainder-plus-tail last
/// stage is `≡ 1 (mod 8)`. Inputs where the lattice does not close (a
/// zero motion tail, an off-grid total) are refused with the arithmetic in
/// the message rather than silently re-shaped.
pub(crate) fn exact_fit_last_stage_for_sidecar(
    req: &mut ChainRequest,
    total_frames: u32,
) -> Result<()> {
    let tail = req.motion_tail_frames;
    let stage_count = req.stages.len();
    if stage_count <= 1 {
        return Ok(());
    }
    let delivered_before_last: u32 = req
        .stages
        .iter()
        .take(stage_count - 1)
        .enumerate()
        .map(|(idx, stage)| {
            if idx == 0 {
                stage.frames
            } else {
                stage.frames.saturating_sub(tail)
            }
        })
        .sum();
    if delivered_before_last >= total_frames {
        anyhow::bail!(
            "chained EXR export planned {stage_count} stages but the first {} already deliver \
             {delivered_before_last} of {total_frames} frames; this is a stage-planning bug",
            stage_count - 1,
        );
    }
    let last_frames = tail + (total_frames - delivered_before_last);
    if last_frames % 8 != 1 || last_frames <= tail {
        anyhow::bail!(
            "chained EXR export needs the final stage to land on the 8k+1 frame grid: \
             {total_frames} total frames with a {tail}-frame motion tail leaves a \
             {last_frames}-frame last stage. Use --frames on the 8k+1 grid (97, 105, 121, …) \
             and a --motion-tail of 8k+1 (9, 17, 25, …).",
        );
    }
    req.stages
        .last_mut()
        .expect("stage_count > 1 checked above")
        .frames = last_frames;
    Ok(())
}

/// CLI-side HDR EXR sidecar inputs for a forced-local chain. Converted into
/// [`mold_inference::chain::ChainHdrConfig`] by `run_chain_local` after the
/// reference video is probed against the requested timeline.
// Non-GPU builds bail before ever reading past `exr_dir`/`full_float` (the
// local render path is compiled out), so the carrier fields count as dead
// there.
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
pub(crate) struct ChainHdrInputs {
    pub exr_dir: String,
    pub full_float: bool,
    /// Normalized control id (`"hdr"`).
    pub control: String,
    /// Reference video container bytes (`--video`).
    pub reference_video: Vec<u8>,
    /// Resolved control-adapter LoRA stack (adapter first).
    pub control_loras: Vec<mold_core::LoraWeight>,
    /// The requested stitched length.
    pub total_frames: u32,
}

impl ChainInputs {
    pub(crate) fn to_chain_request(&self) -> ChainRequest {
        ChainRequest {
            collection: None,
            tags: None,
            title: None,
            model: self.model.clone(),
            stages: Vec::new(),
            motion_tail_frames: self.motion_tail,
            width: self.width,
            height: self.height,
            fps: self.fps,
            seed: self.seed,
            steps: self.steps,
            guidance: self.guidance,
            strength: self.strength,
            output_format: self.output_format,
            placement: self.placement.clone(),
            original_prompt: self.original_prompt.clone(),
            prompt_transform: None,
            batch_id: self.batch_id.clone(),
            batch_index: self.batch_index,
            batch_count: self.batch_count,
            output_mode: Some(mold_core::GenerationOutputMode::OneShot),
            prompt: Some(self.prompt.clone()),
            total_frames: Some(self.total_frames),
            clip_frames: Some(self.clip_frames),
            source_image: self.source_image.clone(),
            enable_audio: self.enable_audio,
        }
    }
}

/// Run a chain end-to-end, dispatching to the server (streaming) or the
/// local orchestrator based on the `local` flag. Handles encoding, save,
/// preview, and final status messages.
///
/// `req` must already be normalised (stages non-empty, auto-expand fields
/// cleared). The three entry points — `generate.rs` (auto-expand from
/// `--frames`), `run_from_sugar` (repeated `--prompt`), and
/// `run_from_script` (TOML script) — each produce a canonical
/// `ChainRequest`, so this helper doesn't re-project through the lossy
/// auto-expand form (which would drop per-stage prompts and transitions).
#[allow(clippy::too_many_arguments)]
pub async fn run_chain(
    req: ChainRequest,
    hdr: Option<ChainHdrInputs>,
    host: Option<String>,
    output: Option<String>,
    no_metadata: bool,
    preview: bool,
    local: bool,
    gpus: Option<String>,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Option<String>,
    eager: bool,
    offload: bool,
) -> Result<()> {
    debug_assert!(
        !req.stages.is_empty(),
        "run_chain requires a normalised ChainRequest (callers must invoke .normalise())"
    );
    // generate.rs refuses hdr+remote before building the inputs; this guard
    // is the defence for any future caller. The ChainRequest wire carries no
    // HDR fields, so a remote submission could only ever silently drop the
    // sidecar.
    if hdr.is_some() && !local {
        anyhow::bail!(
            "chained EXR export renders locally; re-run with --local — a remote server cannot \
             write the sidecar to your machine"
        );
    }

    // A chain's container comes from the script (or the sugar's MP4 default)
    // and has never seen the filename, so the stitched clip could be saved
    // under an extension naming a different container. Reconcile the two
    // before the first stage renders, exactly as the single-clip path does
    // (#1050).
    let mut req = req;
    // Script/repeated CLI prompts are raw here. Auto-expanded prompts carry
    // `original_prompt` and are already canonical model output, so only the
    // raw CLI shapes take the one normalization pass.
    if req.original_prompt.is_none() {
        req.normalize_prompt_newlines();
    }
    req.output_format = super::generate::reconcile_video_format_with_output_extension(
        req.output_format,
        output.as_deref(),
        false,
        super::generate::delivery_capabilities_for_run(local),
    )
    .map_err(anyhow::Error::msg)?;

    let stage_count = req.stages.len() as u32;
    let estimated_total = req.estimated_total_frames();

    status!(
        "{} Chain mode: {} frames across {} stages (tail {})",
        theme::icon_mode(),
        estimated_total,
        stage_count,
        req.motion_tail_frames,
    );

    let ctx = CliContext::new(host.as_deref());
    let config = ctx.config().clone();
    let embed_metadata = config.effective_embed_metadata(no_metadata.then_some(false));
    let _ = embed_metadata; // reserved for future metadata-embed work on chain output

    let t0 = std::time::Instant::now();
    let hdr_metadata = hdr
        .as_ref()
        .map(|inputs| (inputs.exr_dir.clone(), inputs.full_float));
    let video = if local {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            crate::ui::print_using_local_inference();
            run_chain_local(
                &req,
                hdr,
                &config,
                gpus,
                t5_variant,
                qwen3_variant,
                qwen2_variant,
                qwen2_text_encoder_mode,
                eager,
                offload,
            )
            .await?
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            let _ = (
                hdr,
                gpus,
                t5_variant,
                qwen3_variant,
                qwen2_variant,
                qwen2_text_encoder_mode,
                eager,
                offload,
            );
            anyhow::bail!(
                "No mold server running and this binary was built without GPU support.\n\
                 Either start a server with `mold serve` or rebuild with --features cuda"
            )
        }
    } else {
        run_chain_remote(ctx.client(), &req).await?
    };

    let elapsed_ms = t0.elapsed().as_millis() as u64;
    let base_seed = req.seed.unwrap_or(0);

    encode_and_save(
        &req,
        &video,
        output.as_deref(),
        preview,
        elapsed_ms,
        base_seed,
        hdr_metadata.as_ref(),
    )?;

    mold_db::settings::record_last_model(&req.model);
    Ok(())
}

/// Remote chain: create the durable job, follow its event stream with stacked
/// progress bars, then hydrate the stitched print from the host's gallery.
///
/// A sequence is a durable chain job on every surface now — the compatibility
/// endpoints that ran one as a hidden ephemeral job are gone — so a `--script`
/// run that loses its connection leaves a job the host finishes and
/// `mold chain list` can still find.
async fn run_chain_remote(client: &MoldClient, req: &ChainRequest) -> Result<VideoData> {
    let created = client.create_chain_job(req).await?;
    let job_id = created.job_id;
    status!("{} Sequence job {}", theme::icon_info(), job_id.bold());

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ChainProgressEvent>();
    let stage_labels: Vec<StageLabel> = req.stages.iter().map(StageLabel::from_stage).collect();
    let render = tokio::spawn(render_chain_progress(rx, stage_labels));
    let outcome = client.stream_chain_job_events(&job_id, tx).await;
    let _ = render.await;
    let outcome = outcome?;

    if outcome.state != ChainJobState::Completed {
        let detail = outcome
            .error
            .unwrap_or_else(|| format!("sequence job {job_id} ended as {:?}", outcome.state));
        anyhow::bail!("{detail}");
    }
    let filename = outcome
        .output
        .context("sequence completed without a stitched print")?;
    let bytes = client
        .get_gallery_image(&filename)
        .await
        .with_context(|| format!("could not download stitched print {filename}"))?;
    let item = client
        .gallery_item(&filename)
        .await
        .with_context(|| format!("could not read metadata for {filename}"))?
        .with_context(|| format!("stitched print {filename} is missing from the gallery index"))?;
    let metadata = item.metadata;

    Ok(VideoData {
        data: bytes,
        format: item.format.unwrap_or(OutputFormat::Mp4),
        width: metadata.width,
        height: metadata.height,
        frames: metadata.frames.unwrap_or(0),
        fps: metadata.fps.unwrap_or(0),
        pipeline: metadata.pipeline,
        pipeline_provenance_sha256: metadata.pipeline_provenance_sha256.clone(),
        source_preprocessing: metadata.source_preprocessing.clone(),
        // The gallery serves the stitched MP4 itself. Its derived thumbnail
        // and GIF preview are host-side gallery assets, not part of what a
        // `--script` run writes locally.
        thumbnail: Vec::new(),
        gif_preview: Vec::new(),
        has_audio: false,
        duration_ms: None,
        audio_sample_rate: None,
        audio_channels: None,
    })
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
async fn run_chain_local(
    chain_req: &ChainRequest,
    hdr: Option<ChainHdrInputs>,
    config: &Config,
    gpus: Option<String>,
    t5_variant_override: Option<String>,
    qwen3_variant_override: Option<String>,
    qwen2_variant_override: Option<String>,
    qwen2_text_encoder_mode_override: Option<String>,
    eager: bool,
    offload: bool,
) -> Result<VideoData> {
    use super::local_engine::{
        build_local_engine_from_plan, plan_local_batch, resolve_or_pull_model, EngineOverrides,
        LocalBatchAdmission,
    };

    // Normalise so we have expanded stages locally too. The exact-fit last
    // stage the HDR path applies survives this round trip: its shrunken
    // frame count is 8k+1 by construction and `normalise` on an
    // already-canonical request only re-validates.
    let req = chain_req.clone().normalise()?;

    // Convert the CLI-side sidecar inputs into the orchestrator's config,
    // probing the reference against the requested timeline first: every
    // stage conditions on its own temporal window of this clip, so a short
    // or retimed reference would silently regrade the wrong frames.
    let hdr_config: Option<mold_inference::chain::ChainHdrConfig> = match hdr {
        None => None,
        Some(inputs) => {
            let mut reference_file = tempfile::Builder::new()
                .suffix(".mp4")
                .tempfile()
                .map_err(|e| anyhow::anyhow!("failed to stage the reference video: {e}"))?;
            std::io::Write::write_all(&mut reference_file, &inputs.reference_video)?;
            let probe = mold_inference::ltx2::media::probe_video(reference_file.path())
                .map_err(|e| anyhow::anyhow!("failed to read the HDR reference video: {e}"))?;
            let reference_frames = probe
                .frames
                .ok_or_else(|| anyhow::anyhow!("the HDR reference video reports no frame count"))?;
            if reference_frames < inputs.total_frames {
                anyhow::bail!(
                    "the HDR reference video has {reference_frames} frames but this chained \
                     render covers {} frames; the reference must span the full requested \
                     duration (every stage regrades its own window of it)",
                    inputs.total_frames,
                );
            }
            if probe.fps != req.fps {
                anyhow::bail!(
                    "the HDR reference video runs at {} fps but this render is {} fps; a \
                     retimed reference would regrade the wrong frames (pass --fps {} to match)",
                    probe.fps,
                    req.fps,
                    probe.fps,
                );
            }
            Some(mold_inference::chain::ChainHdrConfig {
                exr_dir: std::path::PathBuf::from(&inputs.exr_dir),
                full_float: inputs.full_float,
                ic_lora_control: inputs.control,
                reference_video: inputs.reference_video,
                control_loras: inputs.control_loras,
                total_frames_cap: Some(inputs.total_frames),
            })
        }
    };

    let model_name = req.model.clone();

    // Ensure the model is pulled + config rows are in place (also runs the
    // missing-assets repair pull the single-clip path gets).
    let (_paths, effective_config, _pulled) = resolve_or_pull_model(&model_name, config).await?;
    let overrides = EngineOverrides {
        gpus,
        t5_variant: t5_variant_override,
        qwen3_variant: qwen3_variant_override,
        qwen2_variant: qwen2_variant_override,
        qwen2_text_encoder_mode: qwen2_text_encoder_mode_override,
        eager,
        offload,
    };
    let planning_request = req.synthetic_generate_request(
        req.output_format,
        local_chain_planning_frames(&req),
        req.fps,
    );
    let local_plan = plan_local_batch(&planning_request, &effective_config, &overrides).await?;
    let mut admission =
        LocalBatchAdmission::new(&local_plan.candidates, 1, local_plan.host_headroom_bytes)?;
    for candidate in &local_plan.candidates {
        admission.owner_ready(candidate.ordinal)?;
    }
    let lease = admission
        .lease_ready()?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no local GPU can admit the frozen chain plan"))?;
    let execution_plan = local_plan
        .execution_plans
        .get(&lease.ordinal)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("local chain lease has no frozen execution plan"))?;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ChainProgressEvent>();
    let stage_labels: Vec<StageLabel> = req.stages.iter().map(StageLabel::from_stage).collect();
    let render = tokio::spawn(render_chain_progress(rx, stage_labels));

    let fps = req.fps;
    let output_format = req.output_format;
    // With the exact-fit last stage, an HDR chain's stitched length already
    // equals the requested total; the trim below is then a no-op kept as a
    // second line of defence beside the window planner's cap.
    let total_frames_opt = Some(
        hdr_config
            .as_ref()
            .and_then(|config| config.total_frames_cap)
            .or(req.total_frames)
            .unwrap_or(u32::MAX),
    );
    let req_clone = req.clone();
    let planning_request_clone = planning_request.clone();
    let effective_config_clone = effective_config.clone();
    let prepared_execution_inputs = local_plan.prepared_execution_inputs.clone();

    let handle = tokio::task::spawn_blocking(move || -> Result<VideoData> {
        // Construction, load, render, and drop are all owned by this one
        // device thread. The chain never constructs a CUDA/Metal engine on a
        // Tokio runtime thread.
        mold_inference::device::init_thread_gpu_ordinal(execution_plan.device_ordinal);
        let mut engine = build_local_engine_from_plan(
            &planning_request_clone,
            &effective_config_clone,
            &execution_plan,
            &prepared_execution_inputs,
        )?;
        engine.load()?;
        let renderer = engine.as_chain_renderer().ok_or_else(|| {
            anyhow::anyhow!(
                "model '{}' does not support chained video generation \
                 (LTX-2, LTX-Video, and Wan Video engines expose a \
                 ChainStageRenderer view)",
                req_clone.model,
            )
        })?;
        let mut orch = mold_inference::chain::ChainOrchestrator::new(renderer);

        let tx = tx;
        let mut chain_cb = move |event: ChainProgressEvent| {
            let _ = tx.send(event);
        };
        let chain_output =
            orch.run_with_hdr(&req_clone, Some(&mut chain_cb), hdr_config.as_ref())?;
        if let Some(config) = hdr_config.as_ref() {
            status!(
                "{} Wrote {} EXR frame(s) to {}",
                theme::icon_done(),
                chain_output.hdr_frames_written,
                config.exr_dir.display(),
            );
        }

        use mold_inference::chain::stitch::{stitch_audio_clips, StitchPlan};
        let boundaries: Vec<_> = req_clone
            .stages
            .iter()
            .skip(1)
            .map(|s| s.transition)
            .collect();
        let fade_lens: Vec<_> = req_clone
            .stages
            .iter()
            .skip(1)
            .map(|s| s.fade_frames.unwrap_or(8))
            .collect();
        let audio = stitch_audio_clips(
            &chain_output.stage_audio,
            &boundaries,
            &fade_lens,
            req_clone.motion_tail_frames,
            req_clone.fps,
        )
        .map_err(|e| anyhow::anyhow!("audio stitch failed: {e}"))?;
        let plan = StitchPlan {
            clips: chain_output.stage_frames,
            boundaries,
            fade_lens,
            motion_tail_frames: req_clone.motion_tail_frames,
        };
        let mut frames = plan
            .assemble()
            .map_err(|e| anyhow::anyhow!("stitch failed: {e}"))?;

        if let Some(target) = total_frames_opt {
            let target = target as usize;
            if frames.len() > target {
                frames.truncate(target);
            }
        }
        if frames.is_empty() {
            anyhow::bail!("chain run emitted zero frames after trim");
        }

        encode_local_frames(&frames, fps, output_format, audio.as_ref())
    });

    let result = handle.await??;
    let _ = render.await;
    Ok(result)
}

/// Encode stitched frames to the requested container via the shared
/// chain encoder in mold-inference (MP4 feature-gating, APNG fallbacks,
/// AAC audio mux). Warnings surface on stderr; this wrapper only adds
/// the CLI-specific `VideoData` assembly and first-frame thumbnail.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn encode_local_frames(
    frames: &[image::RgbImage],
    fps: u32,
    output_format: OutputFormat,
    audio: Option<&mold_inference::chain::NativeAudioTrack>,
) -> Result<VideoData> {
    use mold_inference::ltx_video::video_enc;

    let thumbnail = video_enc::first_frame_png(frames).unwrap_or_default();

    let encoded = mold_inference::chain::encode_chain_frames(frames, fps, output_format, audio)?;
    for warning in &encoded.warnings {
        crate::output::status!("{} {}", theme::prefix_warning(), warning.message());
    }
    let (bytes, actual_format, gif_preview) = (encoded.bytes, encoded.format, encoded.gif_preview);

    let width = frames[0].width();
    let height = frames[0].height();
    let frame_count = frames.len() as u32;
    let duration_ms = if fps == 0 {
        None
    } else {
        Some((frame_count as u64 * 1000) / fps as u64)
    };

    let has_audio = audio.is_some() && actual_format == OutputFormat::Mp4;
    let (audio_sample_rate, audio_channels) = if has_audio {
        let track = audio.expect("has_audio implies Some");
        (Some(track.sample_rate), Some(track.channels as u32))
    } else {
        (None, None)
    };

    Ok(VideoData {
        data: bytes,
        format: actual_format,
        width,
        height,
        frames: frame_count,
        fps,
        pipeline: None,
        pipeline_provenance_sha256: None,
        source_preprocessing: None,
        thumbnail,
        gif_preview,
        has_audio,
        duration_ms,
        audio_sample_rate,
        audio_channels,
    })
}

/// Shared epilogue: write the stitched video to stdout/file/gallery and
/// emit a terminal preview if requested. `req` is the normalised chain
/// request — the gallery metadata row joins the distinct clip prompts and
/// carries the structured per-clip chain block.
fn encode_and_save(
    req: &ChainRequest,
    video: &VideoData,
    output: Option<&str>,
    preview: bool,
    elapsed_ms: u64,
    base_seed: u64,
    hdr_sidecar: Option<&(String, bool)>,
) -> Result<()> {
    let piped = is_piped();

    if piped && output.is_none() {
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(&video.data)?;
        stdout.flush()?;
    } else {
        let filename = match output {
            Some("-") => {
                let mut stdout = std::io::stdout().lock();
                stdout.write_all(&video.data)?;
                stdout.flush()?;
                None
            }
            Some(path) => Some(path.to_string()),
            None => Some(mold_core::default_output_filename(
                &req.model,
                mold_core::time::now_epoch_ms_u64(),
                video.format.extension(),
                1,
                0,
            )),
        };
        if let Some(ref filename) = filename {
            if std::path::Path::new(filename).exists() {
                status!("{} Overwriting: {}", theme::icon_alert(), filename);
            }
            std::fs::write(filename, &video.data)?;
            status!(
                "{} Saved: {} ({} frames, {}x{}, {} fps)",
                theme::icon_done(),
                filename.bold(),
                video.frames,
                video.width,
                video.height,
                video.fps,
            );

            // Persist to the gallery metadata DB with the structured
            // per-clip provenance block (no durable job id on the local
            // render path).
            let mut metadata = req.stitched_output_metadata(video.format, video.frames, None);
            // The chain wire format carries no HDR fields (the sidecar is a
            // CLI-forced-local concern), so the stitched print's saved
            // metadata gets the sidecar overlaid here — matching the
            // single-clip precedent where OutputMetadata records where the
            // EXR sequence went so it stays findable from the Library.
            if let Some((exr_dir, full_float)) = hdr_sidecar {
                metadata.hdr_exr_dir = Some(exr_dir.clone());
                metadata.hdr_exr_full_float = *full_float;
            }
            crate::metadata_db::record_local_save_metadata(
                std::path::Path::new(filename),
                metadata,
                elapsed_ms,
                video.format,
                Some((video.width, video.height)),
            );
        }
    }

    if preview && !piped {
        // Best-effort: show the gif preview or fall back to the thumbnail
        // or the video bytes themselves (GIF/APNG decode as images).
        let bytes_for_preview: &[u8] = if !video.gif_preview.is_empty() {
            &video.gif_preview
        } else if !video.thumbnail.is_empty() {
            &video.thumbnail
        } else {
            &video.data
        };
        super::generate::preview_image(bytes_for_preview);
    }

    status!(
        "{} Done — {} in {:.1}s ({} frames, seed: {})",
        theme::icon_done(),
        req.model.bold(),
        elapsed_ms as f64 / 1000.0,
        video.frames,
        req.seed.unwrap_or(base_seed),
    );

    Ok(())
}

/// Per-stage metadata surfaced in the progress-bar label. Built once per
/// run from the normalised `ChainRequest`, then moved into the render
/// task so the `ChainRequest` doesn't have to be Send-cloned.
#[derive(Clone, Debug)]
struct StageLabel {
    transition_tag: &'static str,
    prompt_preview: String,
}

impl StageLabel {
    fn from_stage(stage: &mold_core::chain::ChainStage) -> Self {
        use mold_core::chain::TransitionMode;
        let transition_tag = match stage.transition {
            TransitionMode::Smooth => "smooth",
            TransitionMode::Cut => "cut",
            TransitionMode::Fade => "fade",
        };
        let prompt_preview: String = stage.prompt.chars().take(40).collect();
        Self {
            transition_tag,
            prompt_preview,
        }
    }
}

/// Stacked progress bars for chain render: a parent "Chain" bar covering
/// all pixel frames and a transient per-stage bar covering denoise steps.
async fn render_chain_progress(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<ChainProgressEvent>,
    stage_labels: Vec<StageLabel>,
) {
    // Always draw to stderr so image bytes piped to stdout stay clean.
    let mp = MultiProgress::with_draw_target(ProgressDrawTarget::stderr());

    let parent = mp.add(ProgressBar::new(0));
    parent.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "{{prefix:.{c}}} [{{bar:30.{c}/dim}}] {{pos}}/{{len}} frames {{msg}}",
                c = theme::SPINNER_STYLE,
            ))
            .unwrap()
            .progress_chars("━╸─"),
    );
    parent.set_prefix("Chain");
    parent.enable_steady_tick(Duration::from_millis(100));

    let mut stage_bar: Option<ProgressBar> = None;
    let mut stage_count: u32 = 0;

    while let Some(event) = rx.recv().await {
        match event {
            ChainProgressEvent::ChainStart {
                stage_count: sc,
                estimated_total_frames,
            } => {
                stage_count = sc;
                parent.set_length(estimated_total_frames as u64);
                parent.set_message(format!("(stages {sc})"));
            }
            ChainProgressEvent::StageStart { stage_idx } => {
                if let Some(old) = stage_bar.take() {
                    old.finish_and_clear();
                }
                let label = stage_labels.get(stage_idx as usize);
                let (tag, preview) = match label {
                    Some(l) => (l.transition_tag, l.prompt_preview.as_str()),
                    None => ("smooth", ""),
                };
                parent.set_message(format!("stage {}/{} [{}]", stage_idx + 1, stage_count, tag,));
                let sb = mp.add(ProgressBar::new(0));
                sb.set_style(
                    ProgressStyle::default_bar()
                        .template(&format!(
                            "  Stage {{prefix}}  [{{bar:30.{c}/dim}}] {{pos}}/{{len}} steps {{msg}}",
                            c = theme::SPINNER_STYLE,
                        ))
                        .unwrap()
                        .progress_chars("━╸─"),
                );
                sb.set_prefix(format!("{}/{} [{}]", stage_idx + 1, stage_count, tag));
                if !preview.is_empty() {
                    sb.set_message(format!("\"{preview}\""));
                }
                sb.enable_steady_tick(Duration::from_millis(100));
                stage_bar = Some(sb);
            }
            ChainProgressEvent::DenoiseStep {
                stage_idx: _,
                step,
                total,
            } => {
                if let Some(ref sb) = stage_bar {
                    if sb.length().unwrap_or(0) == 0 {
                        sb.set_length(total as u64);
                    }
                    sb.set_position(step as u64);
                }
            }
            ChainProgressEvent::StageDone {
                stage_idx: _,
                frames_emitted,
            } => {
                if let Some(sb) = stage_bar.take() {
                    sb.finish_and_clear();
                }
                parent.inc(frames_emitted as u64);
            }
            ChainProgressEvent::Stitching { total_frames } => {
                if let Some(sb) = stage_bar.take() {
                    sb.finish_and_clear();
                }
                parent.set_message(format!("stitching {total_frames} frames…"));
            }
        }
    }

    if let Some(sb) = stage_bar.take() {
        sb.finish_and_clear();
    }
    parent.finish_and_clear();
}

/// Load a TOML script file, normalise it, and either submit or print a
/// dry-run summary. Called from the `Commands::Run` early-return when
/// `--script` is set.
#[allow(clippy::too_many_arguments)]
pub async fn run_from_script(
    path: &std::path::Path,
    host: Option<String>,
    output: Option<String>,
    local: bool,
    dry_run: bool,
    no_metadata: bool,
    preview: bool,
    gpus: Option<String>,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Option<String>,
    eager: bool,
    offload: bool,
) -> anyhow::Result<()> {
    let toml_src = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read script {}: {e}", path.display()))?;
    let script_dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let script = mold_core::chain_toml::read_script_resolving_paths(&toml_src, script_dir)
        .map_err(|e| anyhow::anyhow!("invalid chain TOML in {}: {e}", path.display()))?;

    // Before `normalise()`: a stage shorter than the requested tail would
    // otherwise be rejected for a seam that was never going to be applied,
    // and the dry-run totals below would describe a render that cannot happen.
    let mut built = build_request_from_script(&script)?;
    let authority =
        resolve_chain_model_authority(&built.model, &mold_core::Config::load_or_default());
    let substitution = normalize_script_motion_tail(&mut built, &authority);
    report_motion_tail_substitution(substitution, &built.model);
    let req = built.normalise_with_family(authority.family_hint())?;

    if dry_run {
        print_dry_run_summary(&req);
        return Ok(());
    }

    // Submit the normalised ChainRequest as-is so per-stage prompts and
    // transitions survive intact (previously round-tripped through
    // ChainInputs, which collapsed everything into auto-expand form and
    // silently replicated stages[0].prompt across all continuations).
    run_chain(
        req,
        None,
        host,
        output,
        no_metadata,
        preview,
        local,
        gpus,
        t5_variant,
        qwen3_variant,
        qwen2_variant,
        qwen2_text_encoder_mode,
        eager,
        offload,
    )
    .await
}

/// Round a frame count down onto the family's own `step*k + offset` grid,
/// never below the first renderable clip.
fn snap_down_to_family_grid(frames: u32, family: &str) -> u32 {
    let step = mold_core::validation::frame_step_for_family(family).unwrap_or(1);
    let offset = mold_core::validation::frame_offset_for_family(family).unwrap_or(0);
    if step == 0 || frames <= offset {
        return frames;
    }
    offset + ((frames - offset) / step) * step
}

/// The generation recipe repeated `--prompt` renders a chain with.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SugarRecipe {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub steps: u32,
    pub guidance: f64,
    /// Default frames per clip, on the family's own grid.
    pub clip_frames: u32,
    /// Per-clip ceiling an explicit `--frames-per-clip` is clamped to.
    pub clip_cap: u32,
    /// Family the cap and grid were resolved from, for the clamp warning.
    pub family: Option<String>,
}

/// Resolve the multi-prompt sugar recipe from the model's own configuration.
///
/// This path accepts any known model — `main.rs` gates only MiniMax H3 — but
/// used to hardcode LTX-2's 1216x704, 24 fps, 8 steps, guidance 3.0, a
/// 97-frame clip default, and a cap resolved against a literal `"ltx2"`. Those
/// values reach the engine unchanged through the orchestrator's
/// `build_stage_generate_request`, so `wan22-t2v-a14b:q5` was encoded at 24
/// fps rather than its own 16 and denoised for 8 steps against a 4-step
/// Lightning recipe, which renders noise (#783).
///
/// `resolved_model_config` is the same authority the single-shot `run` path
/// uses, so config overrides and manifest defaults layer identically. A model
/// neither can classify keeps the historical LTX-2 shape rather than an
/// invented one.
pub(crate) fn sugar_recipe(model: &str, config: &mold_core::Config) -> SugarRecipe {
    let resolved = mold_core::manifest::resolve_model_name(model);
    let model_cfg = config.resolved_model_config(&resolved);
    let family = crate::commands::generate::resolve_family(&resolved, config);

    let fps = model_cfg
        .effective_fps()
        .unwrap_or(mold_core::validation::LTX2_DEFAULT_FPS);

    // The cap is a per-clip ceiling, so it is resolved at the fps this chain
    // will actually render at — LTX-2's budget is a runtime, not a count.
    let clip_cap = family
        .as_deref()
        .and_then(|family| mold_core::validation::max_frames_for_family_at_fps(family, fps))
        .unwrap_or(LTX2_DEFAULT_CLIP_FRAMES);

    // The clip default is the routing envelope, not the ceiling: wan's
    // two-expert A14B measures near 24 GB well before its 257-frame cap.
    //
    // `wan_default_clip_frames` reads the tier name, which an opaque `cv:` /
    // `hf:` ID does not carry — it would pick the 121-frame non-A14B floor
    // for an installed A14B checkpoint and blow past the envelope its own
    // sidecar records. Prefer the resolved default there, snapped onto the
    // family's grid so the value stays submittable.
    let clip_frames = match family.as_deref() {
        Some("wan") if mold_core::manifest::find_manifest(&resolved).is_some() => {
            wan_default_clip_frames(&resolved)
        }
        Some("wan") => model_cfg
            .effective_frames()
            .map(|frames| snap_down_to_family_grid(frames, "wan"))
            .unwrap_or_else(|| wan_default_clip_frames(&resolved)),
        _ => model_cfg
            .effective_frames()
            .unwrap_or(LTX2_DEFAULT_CLIP_FRAMES),
    }
    .min(clip_cap);

    SugarRecipe {
        width: model_cfg.effective_width(config),
        height: model_cfg.effective_height(config),
        fps,
        steps: model_cfg.effective_steps(config),
        guidance: model_cfg.effective_guidance(),
        clip_frames,
        clip_cap,
        family,
    }
}

/// Replace a script-authored motion tail with the one the family and the
/// selected checkpoint can actually honour, returning `(original, applied)`
/// when a substitution was made.
///
/// The server has done this since #936 (`routes_chain::validate_and_normalize_
/// chain_family`); the CLI never did, so a forced-local `--script` run,
/// `mold chain validate`, and `--dry-run` all carried the caller's value into
/// the stitcher. `17 % 4 == 1`, so LTX-2's default clears wan's own `4k+1`
/// grid check and then discards sixteen good frames at every Smooth seam —
/// the engine seeds the continuation from one frame while the stitch drops
/// seventeen. Both surfaces now read one authority in `mold-core`.
///
/// This runs **before** `normalise()`: a wan stage shorter than the requested
/// tail would otherwise be rejected for a tail that was never going to be
/// applied, and a dry run would report frame totals for a seam that will not
/// render.
///
/// The family and contract come from the same places the server reads them:
/// the sidecar-derived `ModelConfig` an installed `cv:` / `hf:` checkpoint
/// carries, then the checkpoint's own headers, then the manifest. A
/// manifest-only lookup left every catalog wan checkpoint unclassified, which
/// is not merely conservative — an unclassified family also means the LTX
/// `8k+1` grid, so a valid 53-frame wan chain was rejected outright and a
/// 97-frame one silently kept its 17-frame tail.
pub(crate) fn normalize_script_motion_tail(
    req: &mut ChainRequest,
    authority: &ChainModelAuthority,
) -> Option<(u32, u32)> {
    let applied = mold_core::validation::chain_motion_tail_frames_for_family(
        &authority.family,
        authority.source_image,
        req.motion_tail_frames,
    );
    if applied == req.motion_tail_frames {
        return None;
    }
    let original = req.motion_tail_frames;
    req.motion_tail_frames = applied;
    Some((original, applied))
}

/// What a chain needs to know about the selected checkpoint before it can
/// validate or render: the family that owns the frame grid and per-clip cap,
/// and the conditioning contract that decides the seam.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ChainModelAuthority {
    /// Empty when neither the config nor the manifest can classify the model.
    pub family: String,
    pub source_image: Option<mold_core::SourceImageCapability>,
}

impl ChainModelAuthority {
    /// `None` when the model is unclassified, which callers pass through to
    /// `normalise_with_family` as "no hint".
    pub fn family_hint(&self) -> Option<&str> {
        (!self.family.is_empty()).then_some(self.family.as_str())
    }
}

/// Resolve the family and conditioning contract for a chain's model.
///
/// Mirrors the server's `resolve_chain_family` plus its wan header probe, so
/// a forced-local render and an HTTP submission classify the same checkpoint
/// the same way.
pub(crate) fn resolve_chain_model_authority(
    model: &str,
    config: &mold_core::Config,
) -> ChainModelAuthority {
    let resolved = mold_core::manifest::resolve_model_name(model);
    let manifest = mold_core::manifest::find_manifest(&resolved);
    // The installed sidecar's config wins: it is what an opaque catalog ID
    // resolves to, and the manifest cannot classify one at all.
    let family = config
        .resolved_model_config(model)
        .family
        .clone()
        .or_else(|| manifest.map(|model| model.family.to_string()))
        .unwrap_or_default();

    let manifest_contract = manifest.and_then(|model| model.defaults.source_image);
    // Wan's contract is a property of the weights, so read the headers of the
    // artifacts that will actually load; path overrides can point one manifest
    // name at a different checkpoint.
    let source_image = if family == "wan" {
        mold_core::ModelPaths::resolve(model, config)
            .and_then(|paths| {
                mold_inference::wan_source_image_capability(&paths.transformer, &paths.vae)
            })
            .or(manifest_contract)
    } else {
        manifest_contract
    };

    ChainModelAuthority {
        family,
        source_image,
    }
}

/// Tell the user which seam actually rendered. A substituted tail changes the
/// stitched length, so it is disclosed rather than applied silently.
fn report_motion_tail_substitution(substitution: Option<(u32, u32)>, model: &str) {
    if let Some((original, applied)) = substitution {
        status!(
            "{} {model} carries {} frame(s) across a seam, not {original}; using --motion-tail {}",
            theme::prefix_warning(),
            applied,
            applied,
        );
    }
}

/// Build a canonical `ChainRequest` from the parsed TOML script.
/// The result still needs `normalise()` before use.
pub(crate) fn build_request_from_script(
    script: &mold_core::chain::ChainScript,
) -> anyhow::Result<ChainRequest> {
    Ok(ChainRequest {
        collection: None,
        tags: None,
        title: None,
        model: script.chain.model.clone(),
        stages: script.stages.clone(),
        motion_tail_frames: script.chain.motion_tail_frames,
        width: script.chain.width,
        height: script.chain.height,
        fps: script.chain.fps,
        seed: script.chain.seed,
        steps: script.chain.steps,
        guidance: script.chain.guidance,
        strength: script.chain.strength,
        output_format: script.chain.output_format,
        placement: None,
        original_prompt: None,
        prompt_transform: None,
        batch_id: None,
        batch_index: None,
        batch_count: None,
        output_mode: Some(mold_core::GenerationOutputMode::Sequence),
        prompt: None,
        total_frames: None,
        clip_frames: None,
        source_image: None,
        enable_audio: script.chain.enable_audio,
    })
}

/// Multi-prompt sugar: build a uniform multi-stage chain from a `Vec<String>`
/// of `--prompt` values. All stages share the same frame count, dimensions,
/// FPS, and use `TransitionMode::Smooth`. Model resolution matches the normal
/// `run::run` path via the config default or explicit model positional arg.
#[allow(clippy::too_many_arguments)]
pub async fn run_from_sugar(
    model_or_prompt: Option<String>,
    prompts: Vec<String>,
    frames_per_clip: Option<u32>,
    motion_tail: u32,
    enable_audio: Option<bool>,
    dry_run: bool,
    host: Option<String>,
    output: Option<String>,
    local: bool,
    no_metadata: bool,
    preview: bool,
    gpus: Option<String>,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Option<String>,
    eager: bool,
    offload: bool,
) -> anyhow::Result<()> {
    use mold_core::chain::{ChainStage, TransitionMode};
    use mold_core::manifest::{is_known_model, resolve_model_name};

    let config = mold_core::Config::load_or_default();

    // Resolve the model: positional must be a known model name; otherwise
    // fall back to the config default. All prompts come from --prompt in sugar
    // mode — a positional that is NOT a model name is rejected with a clear error.
    let model_raw = match model_or_prompt.as_deref() {
        Some(m) if is_known_model(m, &config) => m.to_string(),
        Some(m) => {
            anyhow::bail!(
                "unknown model '{m}'; when using repeated --prompt, the first positional arg \
                 must be a known model (or omit it to use the config default)"
            );
        }
        None => config.resolved_default_model(),
    };
    let model = resolve_model_name(&model_raw);

    // Every dimension of the render is the selected model's own, resolved
    // through the same `resolved_model_config` the single-shot path uses.
    // Hardcoding LTX-2's here encoded wan A14B at 24 fps instead of 16 and
    // denoised it for 8 steps against a 4-step recipe (#783).
    let recipe = sugar_recipe(&model, &config);
    let clip_frames = frames_per_clip
        .unwrap_or(recipe.clip_frames)
        .min(recipe.clip_cap);
    if let Some(requested) = frames_per_clip {
        if requested > recipe.clip_cap {
            crate::output::status!(
                "{} --frames-per-clip {} exceeds the per-clip budget for '{}' at {} fps \
                 ({} frames), clamping to {}",
                theme::prefix_warning(),
                requested,
                recipe.family.as_deref().unwrap_or("this model"),
                recipe.fps,
                recipe.clip_cap,
                recipe.clip_cap,
            );
        }
    }

    // Build the canonical ChainRequest from the list of prompts.
    let stages: Vec<ChainStage> = prompts
        .iter()
        .map(|p| ChainStage {
            prompt: p.clone(),
            frames: clip_frames,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Smooth,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        })
        .collect();

    // Geometry, timing, and the denoise recipe all come from the model.
    let req = ChainRequest {
        collection: None,
        tags: None,
        title: None,
        model: model.clone(),
        stages,
        motion_tail_frames: motion_tail,
        width: recipe.width,
        height: recipe.height,
        fps: recipe.fps,
        seed: None,
        steps: recipe.steps,
        guidance: recipe.guidance,
        strength: 1.0,
        output_format: OutputFormat::Mp4,
        placement: None,
        original_prompt: None,
        prompt_transform: None,
        batch_id: None,
        batch_index: None,
        batch_count: None,
        output_mode: Some(mold_core::GenerationOutputMode::Sequence),
        prompt: None,
        total_frames: None,
        clip_frames: None,
        source_image: None,
        enable_audio,
    };

    // `--motion-tail` defaults to LTX-2's 17 for every family, so repeated
    // `--prompt` on a wan model inherited a seam its checkpoint cannot
    // honour. Resolve it before `normalise()` for the same reason the
    // script path does (#783).
    let mut built = req;
    let authority = resolve_chain_model_authority(&built.model, &config);
    let substitution = normalize_script_motion_tail(&mut built, &authority);
    report_motion_tail_substitution(substitution, &built.model);
    let req = built.normalise_with_family(authority.family_hint())?;

    if dry_run {
        print_dry_run_summary(&req);
        return Ok(());
    }

    run_chain(
        req,
        None,
        host,
        output,
        no_metadata,
        preview,
        local,
        gpus,
        t5_variant,
        qwen3_variant,
        qwen2_variant,
        qwen2_text_encoder_mode,
        eager,
        offload,
    )
    .await
}

/// Print a human-readable summary of the normalised chain for `--dry-run`
/// mode. Written to stdout (not through the status! macro) so users can
/// `mold run --script foo.toml --dry-run | less` cleanly.
fn print_dry_run_summary(req: &ChainRequest) {
    use mold_core::chain::TransitionMode;
    let stage_count = req.stages.len();
    let total_frames = req.estimated_total_frames();
    let fps = req.fps.max(1);
    let duration_s = total_frames as f64 / fps as f64;
    println!("{stage_count} stages");
    println!("estimated total frames: {total_frames} ({duration_s:.2}s @ {fps}fps)",);
    for (i, s) in req.stages.iter().enumerate() {
        let tag = match s.transition {
            TransitionMode::Smooth => "smooth",
            TransitionMode::Cut => "cut",
            TransitionMode::Fade => "fade",
        };
        let prompt_preview: String = s.prompt.chars().take(60).collect();
        println!("  [{i}] {tag}  {}f  \"{}\"", s.frames, prompt_preview);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A script-authored wan chain renders with the checkpoint's seam, not
    /// the caller's (#783).
    ///
    /// The server has normalized this since #936; the CLI never did. A wan
    /// TOML carrying LTX-2's 17 passes `normalise()` clean — `17 % 4 == 1`
    /// sits on wan's own `4k+1` grid — and then the Smooth stitch drops all
    /// seventeen incoming frames while the engine seeded the continuation
    /// from one. Sixteen good frames vanish per seam, with correct-looking
    /// validation output. This is the regression test for that.
    #[test]
    fn a_script_authored_wan_chain_takes_the_checkpoints_seam() {
        let wan_script_request = |model: &str, tail: u32| ChainRequest {
            model: model.to_string(),
            stages: vec![
                stage_with_frames("a paper boat drifting down a rain gutter", 49),
                stage_with_frames("the boat reaches a storm drain", 49),
            ],
            motion_tail_frames: tail,
            width: 704,
            height: 384,
            fps: 24,
            ..empty_chain_request()
        };

        let config = mold_core::Config::default();
        let authority_for = |model: &str| resolve_chain_model_authority(model, &config);

        // TI2V-5B is `Optional` — it can be seeded, so the seam is one frame.
        let mut ti2v = wan_script_request("wan22-ti2v-5b:q8", 17);
        let substitution =
            normalize_script_motion_tail(&mut ti2v, &authority_for("wan22-ti2v-5b:q8"));
        assert_eq!(
            ti2v.motion_tail_frames,
            mold_core::validation::WAN_HANDOFF_DUPLICATED_FRAMES,
            "wan re-renders exactly the frame it was seeded with"
        );
        assert_eq!(
            substitution,
            Some((17, 1)),
            "a substituted seam must be reported, never applied silently"
        );
        // And it still normalises — the value we wrote is on wan's grid.
        assert!(ti2v.normalise().is_ok());

        // A text-to-video checkpoint has no channel to be seeded through, so
        // its Smooth boundaries concatenate.
        let mut t2v = wan_script_request("wan21-t2v-1.3b:bf16", 17);
        assert_eq!(
            normalize_script_motion_tail(&mut t2v, &authority_for("wan21-t2v-1.3b:bf16")),
            Some((17, 0))
        );
        assert_eq!(t2v.motion_tail_frames, 0);

        // An already-correct script is left alone and reports nothing.
        let mut correct = wan_script_request("wan22-ti2v-5b:q8", 1);
        assert_eq!(
            normalize_script_motion_tail(&mut correct, &authority_for("wan22-ti2v-5b:q8")),
            None
        );
        assert_eq!(correct.motion_tail_frames, 1);

        // LTX-2's tail is a real latent window; the script still owns it.
        let mut ltx2 = ChainRequest {
            model: "ltx-2-19b-distilled:fp8".to_string(),
            stages: vec![
                stage_with_frames("a drone shot over pine forest", 97),
                stage_with_frames("the trees give way to a lake", 97),
            ],
            motion_tail_frames: 17,
            width: 1216,
            height: 704,
            fps: 24,
            ..empty_chain_request()
        };
        assert_eq!(
            normalize_script_motion_tail(&mut ltx2, &authority_for("ltx-2-19b-distilled:fp8")),
            None
        );
        assert_eq!(ltx2.motion_tail_frames, 17);
    }

    /// Repeated `--prompt` renders with the selected model's own recipe, not
    /// LTX-2's (#783).
    ///
    /// `run_from_sugar` accepts any known model — `main.rs` gates only
    /// MiniMax H3 — and then hardcoded 1216x704, 24 fps, 8 steps, guidance
    /// 3.0, a 97-frame clip default, and a cap resolved against a literal
    /// `"ltx2"`. Those values reach wan unchanged through the orchestrator's
    /// `build_stage_generate_request`, so `wan22-t2v-a14b:q5` was encoded at
    /// 24 fps instead of its own 16 — 1.5x fast playback — and denoised for 8
    /// steps against a 4-step Lightning recipe, which renders noise.
    #[test]
    fn multi_prompt_sugar_uses_the_selected_models_own_recipe() {
        let config = mold_core::Config::default();

        // A14B Lightning: its own geometry, 16 fps, and a 4-step recipe.
        let a14b = sugar_recipe("wan22-t2v-a14b:q5", &config);
        assert_eq!(
            (a14b.width, a14b.height, a14b.fps, a14b.steps, a14b.guidance),
            (832, 480, 16, 4, 1.0),
            "sugar sent 1216x704 / 24 fps / 8 steps / 3.0 for this checkpoint"
        );
        // The Quality tier is the same checkpoint family on a different
        // recipe, which is exactly why one hardcoded tuple cannot serve both.
        let a14b_q8 = sugar_recipe("wan22-t2v-a14b:q8", &config);
        assert_eq!(
            (a14b_q8.fps, a14b_q8.steps, a14b_q8.guidance),
            (16, 20, 3.5)
        );

        assert_eq!(
            (a14b.clip_frames - 1) % mold_core::validation::WAN_TEMPORAL_SCALE,
            0,
            "the clip default must sit on wan's own 4k+1 grid"
        );
        assert!(
            a14b.clip_frames <= a14b.clip_cap,
            "clip default {} exceeds the resolved cap {}",
            a14b.clip_frames,
            a14b.clip_cap
        );
        assert_eq!(
            a14b.clip_cap,
            mold_core::validation::max_frames_for_family_at_fps("wan", 16).unwrap(),
            "the cap is wan's flat request ceiling, not LTX-2's runtime budget"
        );
        // The routing envelope, matching what `decide_chain_routing` picks so
        // sugar and auto-chain cannot disagree about the same model.
        assert_eq!(
            a14b.clip_frames,
            wan_default_clip_frames("wan22-t2v-a14b:q5")
        );

        // TI2V-5B is 24 fps, but for its own reason, and 1280x704.
        let ti2v = sugar_recipe("wan22-ti2v-5b:q8", &config);
        assert_eq!(
            (ti2v.width, ti2v.height, ti2v.fps, ti2v.steps, ti2v.guidance),
            (1280, 704, 24, 20, 5.0)
        );
        assert_eq!(
            (ti2v.clip_frames - 1) % mold_core::validation::WAN_TEMPORAL_SCALE,
            0
        );

        // LTX-2 keeps exactly what it had, so nothing regresses for it.
        let ltx2 = sugar_recipe("ltx-2-19b-distilled:fp8", &config);
        assert_eq!(ltx2.fps, 24);
        assert_eq!(ltx2.steps, 8);
        assert_eq!(ltx2.width, 1216);
        assert_eq!(ltx2.height, 704);
        assert_eq!(ltx2.clip_frames, LTX2_DEFAULT_CLIP_FRAMES);
        assert_eq!(
            ltx2.clip_cap,
            mold_core::validation::max_frames_for_family_at_fps(
                "ltx2",
                mold_core::validation::LTX2_DEFAULT_FPS
            )
            .unwrap()
        );

        // An unresolvable model keeps the historical LTX-2 shape rather than
        // inventing one.
        let unknown = sugar_recipe("cv:2041121", &config);
        assert_eq!(unknown.fps, 24);
        assert_eq!(unknown.clip_frames, LTX2_DEFAULT_CLIP_FRAMES);
    }

    /// Everything a `ChainRequest` needs that these tests do not care about.
    /// `model`, `stages`, `motion_tail_frames`, `width`, `height`, and `fps`
    /// are always spelled out at the call site.
    fn empty_chain_request() -> ChainRequest {
        ChainRequest {
            collection: None,
            tags: None,
            title: None,
            model: String::new(),
            stages: vec![],
            motion_tail_frames: 0,
            width: 0,
            height: 0,
            fps: 24,
            seed: None,
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
            output_mode: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
        }
    }

    /// An installed `cv:` / `hf:` wan checkpoint is classified from its
    /// sidecar, not its name (#783).
    ///
    /// `find_manifest` cannot classify a catalog id, so a manifest-only
    /// lookup left the family empty — and an empty family is not merely
    /// "unknown", it is the LTX `8k+1` grid and a preserved 17-frame tail.
    /// A valid 53-frame wan chain was rejected outright, and a 97-frame one
    /// (which also clears `8k+1`) kept a seam that discards sixteen frames.
    /// The tier name is likewise absent, so the routing default cannot be
    /// sniffed from it either.
    #[test]
    fn an_opaque_catalog_wan_model_is_classified_from_its_sidecar() {
        let mut config = mold_core::Config::default();
        config.models.insert(
            "cv:2041121".to_string(),
            mold_core::ModelConfig {
                family: Some("wan".to_string()),
                default_frames: Some(81),
                default_fps: Some(16),
                default_width: Some(832),
                default_height: Some(480),
                ..Default::default()
            },
        );

        // The family now comes from the sidecar, so wan's grid applies.
        let authority = resolve_chain_model_authority("cv:2041121", &config);
        assert_eq!(authority.family, "wan");
        assert_eq!(authority.family_hint(), Some("wan"));

        // A 53-frame chain is on wan's grid and must survive normalisation.
        let mut req = ChainRequest {
            model: "cv:2041121".into(),
            stages: vec![
                stage_with_frames("a paper boat", 53),
                stage_with_frames("it reaches the drain", 53),
            ],
            motion_tail_frames: 17,
            width: 832,
            height: 480,
            fps: 16,
            ..empty_chain_request()
        };
        normalize_script_motion_tail(&mut req, &authority);
        assert!(
            req.normalise_with_family(authority.family_hint()).is_ok(),
            "53 is 4k+1; the LTX fallback grid rejected it"
        );

        // Its routing default comes from the sidecar rather than the
        // 121-frame non-A14B floor the tier name would have selected.
        let recipe = sugar_recipe("cv:2041121", &config);
        assert_eq!(recipe.family.as_deref(), Some("wan"));
        assert_eq!(recipe.fps, 16);
        assert_eq!(
            recipe.clip_frames, 81,
            "the sidecar records this checkpoint's measured envelope"
        );
        assert_eq!(
            (recipe.clip_frames - 1) % mold_core::validation::WAN_TEMPORAL_SCALE,
            0
        );
    }

    /// An off-grid recorded default is snapped down, never submitted as-is.
    #[test]
    fn a_catalog_default_off_the_family_grid_snaps_down() {
        assert_eq!(snap_down_to_family_grid(80, "wan"), 77);
        assert_eq!(snap_down_to_family_grid(81, "wan"), 81);
        assert_eq!(snap_down_to_family_grid(97, "ltx2"), 97);
        assert_eq!(snap_down_to_family_grid(96, "ltx2"), 89);
        // Never below the first renderable clip.
        assert_eq!(snap_down_to_family_grid(1, "wan"), 1);
    }

    fn stage_with_frames(prompt: &str, frames: u32) -> mold_core::chain::ChainStage {
        mold_core::chain::ChainStage {
            prompt: prompt.to_string(),
            frames,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: mold_core::chain::TransitionMode::Smooth,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        }
    }

    /// Wan auto-chains instead of being rejected, on its own grid and with a
    /// seam its checkpoint can actually honour (#783).
    #[test]
    fn wan_auto_chains_with_a_checkpoint_shaped_seam() {
        use mold_core::SourceImageCapability::{Optional, Required, Unsupported};

        // Before this, any wan request past the cap was Rejected: the family
        // was not chain-capable here at all.
        let five_b = decide_chain_routing(
            Some(300),
            Some("wan"),
            "wan22-ti2v-5b:fp16",
            None,
            24,
            24,
            None,
            Some(Optional),
        );
        // The seam duplicates exactly the one frame the continuation was
        // seeded with -- NOT the caller's LTX-shaped 24.
        assert_eq!(
            five_b,
            ChainRoutingDecision::Chain {
                clip_frames: 121,
                motion_tail: WAN_HANDOFF_DUPLICATED_FRAMES,
            },
        );

        // The two-expert pair gets the tighter 24 GB envelope — 81 on the
        // Q5-backed tiers, which is what their manifest default records once
        // block offload made the checkpoint's trained clip length fit (#776).
        let a14b = decide_chain_routing(
            Some(300),
            Some("wan"),
            "wan22-i2v-a14b:q5",
            None,
            24,
            16,
            None,
            Some(Required),
        );
        assert_eq!(
            a14b,
            ChainRoutingDecision::Chain {
                clip_frames: 81,
                motion_tail: WAN_HANDOFF_DUPLICATED_FRAMES,
            },
        );

        // A text-to-video checkpoint has no conditioning channel at all, so
        // the seam carries nothing however large a tail was requested.
        let t2v = decide_chain_routing(
            Some(300),
            Some("wan"),
            "wan22-t2v-a14b:q5",
            None,
            24,
            16,
            None,
            Some(Unsupported),
        );
        assert_eq!(
            t2v,
            ChainRoutingDecision::Chain {
                clip_frames: 81,
                motion_tail: 0,
            },
        );
        // An unclassified checkpoint is "unknown", not an assumed handoff.
        assert_eq!(
            decide_chain_routing(Some(300), Some("wan"), "cv:12345", None, 24, 16, None, None,),
            ChainRoutingDecision::Chain {
                clip_frames: 121,
                motion_tail: 0,
            },
        );

        // Regression (#776 item 4): a tier's manifest default must render as
        // ONE clip. It shipped otherwise — the Q5/Q4 A14B default was raised
        // to the checkpoint's trained 81 frames while this routing default
        // stayed at the pre-offload 53, so `mold run wan22-t2v-a14b:q5` with
        // no `--frames` produced a 2-stage 106-frame chain in 351.6 s instead
        // of the 81-frame clip its default advertises. Asserted for every wan
        // model the manifest knows, so the next raised default cannot drift
        // away from the routing again.
        let mut checked = 0;
        for manifest in mold_core::manifest::known_manifests()
            .iter()
            .filter(|manifest| manifest.family == "wan")
        {
            let Some(default_frames) = manifest.defaults.frames else {
                continue;
            };
            checked += 1;
            let decision = decide_chain_routing(
                Some(default_frames),
                Some("wan"),
                &manifest.name,
                None,
                0,
                16,
                None,
                manifest.defaults.source_image,
            );
            assert_eq!(
                decision,
                ChainRoutingDecision::SingleClip,
                "{} defaults to {default_frames} frames, which must render as one clip \
                 rather than routing to a stitched sequence",
                manifest.name,
            );
        }
        assert!(
            checked >= 4,
            "only {checked} wan manifests carried a default frame count — the loop above \
             would pass vacuously"
        );

        // Every routed clip length is on wan's 4k+1 grid, so a clip started at
        // the default is submittable.
        for model in ["wan22-ti2v-5b:fp16", "wan22-t2v-a14b:q5", "cv:12345"] {
            let decision =
                decide_chain_routing(Some(300), Some("wan"), model, None, 4, 16, None, None);
            let ChainRoutingDecision::Chain { clip_frames, .. } = decision else {
                panic!("{model} must auto-chain");
            };
            assert_eq!((clip_frames - 1) % 4, 0, "{model} clip {clip_frames}");
        }

        // Under the envelope it still stays on the single-clip path.
        assert_eq!(
            decide_chain_routing(
                Some(53),
                Some("wan"),
                "wan22-t2v-a14b:q5",
                None,
                0,
                16,
                None,
                Some(Unsupported),
            ),
            ChainRoutingDecision::SingleClip,
        );

        // An explicit --clip-frames still wins, clamped to the real cap.
        assert_eq!(
            decide_chain_routing(
                Some(600),
                Some("wan"),
                "wan22-ti2v-5b:fp16",
                Some(999),
                0,
                24,
                None,
                Some(Optional),
            ),
            ChainRoutingDecision::Chain {
                clip_frames: mold_core::validation::MAX_FRAMES_GLOBAL,
                motion_tail: WAN_HANDOFF_DUPLICATED_FRAMES,
            },
        );
    }

    #[test]
    fn routing_single_clip_under_cap() {
        let d = decide_chain_routing(
            Some(97),
            Some("ltx2"),
            "ltx-2-19b-distilled:fp8",
            None,
            4,
            24,
            None,
            None,
        );
        assert_eq!(d, ChainRoutingDecision::SingleClip);
    }

    /// Regression: `--pipeline t2a --frames 121` used to exceed the 97-frame
    /// per-clip *video* cap and get auto-chained, turning a 5-second audio
    /// request into a two-stage 177-frame video render. Audio has no clips to
    /// stitch and no per-clip VRAM ceiling — its `frames` is a duration.
    #[test]
    fn routing_never_chains_an_audio_only_pipeline() {
        for frames in [121u32, 400, 600] {
            let d = decide_chain_routing(
                Some(frames),
                Some("ltx2"),
                "ltx-2-19b-dev:fp8",
                None,
                17,
                24,
                Some(mold_core::Ltx2PipelineMode::T2a),
                None,
            );
            assert_eq!(
                d,
                ChainRoutingDecision::SingleClip,
                "t2a at {frames} frames must stay a single request"
            );
        }

        // The same frame count on a video pipeline still chains, so the guard
        // is scoped to audio rather than disabling auto-chaining outright.
        let video = decide_chain_routing(
            Some(400),
            Some("ltx2"),
            "ltx-2-19b-dev:fp8",
            None,
            17,
            24,
            Some(mold_core::Ltx2PipelineMode::TwoStage),
            None,
        );
        assert!(matches!(video, ChainRoutingDecision::Chain { .. }));
    }

    #[test]
    fn routing_single_clip_when_frames_absent() {
        let d = decide_chain_routing(
            None,
            Some("ltx2"),
            "ltx-2-19b-distilled:fp8",
            None,
            4,
            24,
            None,
            None,
        );
        assert_eq!(d, ChainRoutingDecision::SingleClip);
    }

    #[test]
    fn routing_chain_over_cap_ltx2_distilled() {
        let d = decide_chain_routing(
            Some(200),
            Some("ltx2"),
            "ltx-2-19b-distilled:fp8",
            None,
            4,
            24,
            None,
            None,
        );
        assert_eq!(
            d,
            ChainRoutingDecision::Chain {
                clip_frames: 97,
                motion_tail: 4,
            },
        );
    }

    #[test]
    fn routing_rejects_non_distilled_over_cap() {
        let d = decide_chain_routing(
            Some(200),
            Some("flux"),
            "flux-dev:q4",
            None,
            4,
            24,
            None,
            None,
        );
        match d {
            ChainRoutingDecision::Rejected { reason } => {
                assert!(
                    reason.contains("does not support chained video"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    /// Chain capability is a property of the family. The old
    /// `model.contains("distilled")` test refused a dev checkpoint the server
    /// happily chains, and refused every opaque catalog ID outright.
    #[test]
    fn routing_chains_every_ltx2_checkpoint_over_cap() {
        for model in [
            "ltx-2-19b-dev:fp8",
            "ltx-2.3-22b-dev:fp8",
            "ltx-2-19b-distilled:fp8",
            "cv:3143864",
        ] {
            let decision =
                decide_chain_routing(Some(400), Some("ltx2"), model, None, 17, 24, None, None);
            assert!(
                matches!(decision, ChainRoutingDecision::Chain { .. }),
                "{model} must auto-chain, got {decision:?}"
            );
        }
    }

    #[test]
    fn routing_rejects_non_ltx2_family_over_cap() {
        // ltx-video (not ltx2) is not chainable in v1, so anything past its own
        // single-request ceiling has nowhere to go.
        let d = decide_chain_routing(
            Some(500),
            Some("ltx-video"),
            "ltx-video:0.9.6",
            None,
            4,
            24,
            None,
            None,
        );
        assert!(matches!(d, ChainRoutingDecision::Rejected { .. }));
    }

    /// The CLI used to reject any non-ltx2 request past 97 frames, which was
    /// stricter than the server: ltx-video accepts up to the global ceiling.
    #[test]
    fn routing_keeps_non_chainable_families_single_up_to_their_own_ceiling() {
        let d = decide_chain_routing(
            Some(249),
            Some("ltx-video"),
            "ltx-video:0.9.6",
            None,
            4,
            24,
            None,
            None,
        );
        assert_eq!(d, ChainRoutingDecision::SingleClip);
    }

    /// `--clip-frames` clamps to the model's real budget, not to the routing
    /// default — that is how a user asks for one long clip instead of a
    /// stitched sequence.
    #[test]
    fn routing_clip_frames_may_exceed_the_routing_default() {
        let d = decide_chain_routing(
            Some(300),
            Some("ltx2"),
            "ltx-2-19b-distilled:fp8",
            Some(201),
            4,
            24,
            None,
            None,
        );
        assert_eq!(
            d,
            ChainRoutingDecision::Chain {
                clip_frames: 201,
                motion_tail: 4,
            },
        );
    }

    #[test]
    fn routing_clip_frames_above_the_model_budget_clamps_to_the_budget() {
        // 12 fps → a 20s budget is 244 frames, but 244 is off the 8n+1 grid
        // (243 % 8 == 3), so clamping to it produced a clip-frame count the
        // server rejects. The advertised cap is now grid-snapped to 241.
        let d = decide_chain_routing(
            Some(900),
            Some("ltx2"),
            "ltx-2-19b-distilled:fp8",
            Some(400),
            4,
            12,
            None,
            None,
        );
        assert_eq!(
            d,
            ChainRoutingDecision::Chain {
                clip_frames: 241,
                motion_tail: 4,
            },
        );
    }

    /// Auto-chaining (no --clip-frames) must keep using the conservative
    /// routing default; the corrected ceiling must not silently promote every
    /// long request into one enormous single-clip denoise.
    #[test]
    fn routing_default_clip_size_is_unchanged_by_the_larger_budget() {
        let d = decide_chain_routing(
            Some(400),
            Some("ltx2"),
            "ltx-2-19b-distilled:fp8",
            None,
            4,
            24,
            None,
            None,
        );
        assert_eq!(
            d,
            ChainRoutingDecision::Chain {
                clip_frames: LTX2_DEFAULT_CLIP_FRAMES,
                motion_tail: 4,
            },
        );
    }

    #[test]
    fn routing_clip_frames_under_cap_respected() {
        let d = decide_chain_routing(
            Some(300),
            Some("ltx2"),
            "ltx-2-19b-distilled:fp8",
            Some(65),
            4,
            24,
            None,
            None,
        );
        assert_eq!(
            d,
            ChainRoutingDecision::Chain {
                clip_frames: 65,
                motion_tail: 4,
            },
        );
    }

    #[test]
    fn routing_motion_tail_ge_clip_frames_rejects() {
        let d = decide_chain_routing(
            Some(300),
            Some("ltx2"),
            "ltx-2-19b-distilled:fp8",
            Some(49),
            49,
            24,
            None,
            None,
        );
        match d {
            ChainRoutingDecision::Rejected { reason } => {
                assert!(
                    reason.contains("--motion-tail"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[test]
    fn routing_motion_tail_at_clip_frames_rejects() {
        let d = decide_chain_routing(
            Some(200),
            Some("ltx2"),
            "ltx-2-19b-distilled:fp8",
            None,
            97,
            24,
            None,
            None,
        );
        assert!(matches!(d, ChainRoutingDecision::Rejected { .. }));
    }

    /// 121 total @ clip 97 / tail 17: auto-expand plans [97, 97] and
    /// overshoots to 177; the sidecar path shrinks the last stage to 41 so
    /// the stitch delivers exactly 121 — `97 + (41 − 17) = 121` — and no
    /// stage renders past the reference video's end.
    #[test]
    fn exact_fit_shrinks_the_last_stage_to_the_requested_total() {
        let mut req = ChainRequest {
            collection: None,
            tags: None,
            title: None,
            model: "ltx-2.3-22b-distilled:fp8".into(),
            stages: Vec::new(),
            motion_tail_frames: 17,
            width: 1216,
            height: 704,
            fps: 24,
            seed: Some(7),
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
            prompt: Some("x".into()),
            total_frames: Some(121),
            clip_frames: Some(97),
            source_image: None,
            enable_audio: None,
        }
        .normalise()
        .unwrap();
        assert_eq!(
            req.stages.iter().map(|s| s.frames).collect::<Vec<_>>(),
            vec![97, 97],
        );

        exact_fit_last_stage_for_sidecar(&mut req, 121).unwrap();
        assert_eq!(
            req.stages.iter().map(|s| s.frames).collect::<Vec<_>>(),
            vec![97, 41],
        );
        assert_eq!(req.estimated_total_frames(), 121);
        // The shrunken stage still satisfies every chain invariant, so a
        // re-normalise (run_chain_local does one) is a no-op.
        let renormalised = req.clone().normalise().unwrap();
        assert_eq!(renormalised.stages[1].frames, 41);
    }

    /// The lattice closes for every canonical total: over a spread of 8k+1
    /// totals and tails, the exact-fit last stage is 8k+1, strictly above the
    /// tail, and the stitched total equals the request exactly.
    #[test]
    fn exact_fit_closes_the_lattice_for_canonical_totals() {
        for tail in [9u32, 17, 25] {
            for total in (105..=1537).step_by(8) {
                let clip = 97u32;
                let mut req = ChainRequest {
                    collection: None,
                    tags: None,
                    title: None,
                    model: "ltx-2.3-22b-distilled:fp8".into(),
                    stages: Vec::new(),
                    motion_tail_frames: tail,
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
                    prompt: Some("x".into()),
                    total_frames: Some(total),
                    clip_frames: Some(clip),
                    source_image: None,
                    enable_audio: None,
                };
                // Totals needing more than MAX_CHAIN_STAGES stages reject at
                // normalise; skip them (the property is about the fit, not
                // the stage ceiling).
                let Ok(normalised) = req.clone().normalise() else {
                    continue;
                };
                req = normalised;
                if req.stages.len() < 2 {
                    continue;
                }
                exact_fit_last_stage_for_sidecar(&mut req, total).unwrap_or_else(|e| {
                    panic!("total {total} tail {tail}: {e}");
                });
                let last = req.stages.last().unwrap().frames;
                assert_eq!(last % 8, 1, "total {total} tail {tail}: last {last}");
                assert!(last > tail, "total {total} tail {tail}: last {last}");
                assert_eq!(
                    req.estimated_total_frames(),
                    total,
                    "total {total} tail {tail}",
                );
            }
        }
    }

    /// An off-grid combination is refused with the arithmetic, never
    /// silently re-shaped.
    #[test]
    fn exact_fit_refuses_a_layout_the_lattice_cannot_close() {
        let mut req = ChainRequest {
            collection: None,
            tags: None,
            title: None,
            model: "ltx-2.3-22b-distilled:fp8".into(),
            stages: Vec::new(),
            motion_tail_frames: 0,
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
            prompt: Some("x".into()),
            total_frames: Some(150),
            clip_frames: Some(97),
            source_image: None,
            enable_audio: None,
        }
        .normalise()
        .unwrap();
        let err = exact_fit_last_stage_for_sidecar(&mut req, 150).unwrap_err();
        assert!(
            format!("{err}").contains("8k+1"),
            "error must explain the frame grid: {err}",
        );
    }

    #[test]
    fn ltx2_distilled_cap_matches_engine_constraint() {
        // 97 = 8 * 12 + 1, satisfying the VAE 8k+1 constraint.
        assert_eq!(LTX2_DEFAULT_CLIP_FRAMES % 8, 1);
    }

    #[test]
    fn stage_label_from_stage_builds_tag_and_preview() {
        use mold_core::chain::{ChainStage, TransitionMode};
        let stage = ChainStage {
            prompt: "a long prompt that should be truncated to forty characters here ok".into(),
            frames: 97,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Fade,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        };
        let label = super::StageLabel::from_stage(&stage);
        assert_eq!(label.transition_tag, "fade");
        assert_eq!(label.prompt_preview.chars().count(), 40);
        assert!(label.prompt_preview.starts_with("a long prompt that"));
    }

    #[test]
    fn stage_label_tags_each_transition_variant() {
        use mold_core::chain::{ChainStage, TransitionMode};
        let make = |transition: TransitionMode| {
            super::StageLabel::from_stage(&ChainStage {
                prompt: "p".into(),
                frames: 9,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            })
        };
        assert_eq!(make(TransitionMode::Smooth).transition_tag, "smooth");
        assert_eq!(make(TransitionMode::Cut).transition_tag, "cut");
        assert_eq!(make(TransitionMode::Fade).transition_tag, "fade");
    }

    /// Round-trip: a TOML-style script parsed into a ChainRequest should
    /// come out of `build_request_from_script` + `normalise` with all
    /// stages intact, their prompts and transitions unchanged. This is the
    /// contract `run_chain` relies on (it pulls `stages[0]` in
    /// `encode_and_save` and hands the whole request to the engine).
    #[test]
    fn script_request_preserves_multi_stage_prompts_and_transitions() {
        use mold_core::chain::{ChainScript, ChainScriptChain, ChainStage, TransitionMode};
        let script = ChainScript {
            schema: "mold.chain.v1".into(),
            chain: ChainScriptChain {
                model: "ltx-2-19b-distilled:fp8".into(),
                width: 1216,
                height: 704,
                fps: 24,
                seed: Some(7),
                steps: 8,
                guidance: 3.0,
                strength: 1.0,
                motion_tail_frames: 17,
                output_format: OutputFormat::Mp4,
                enable_audio: None,
            },
            stages: vec![
                ChainStage {
                    prompt: "cat in garden".into(),
                    frames: 97,
                    source_image: None,
                    negative_prompt: None,
                    seed_offset: None,
                    transition: TransitionMode::Smooth,
                    fade_frames: None,
                    model: None,
                    loras: vec![],
                    references: vec![],
                },
                ChainStage {
                    prompt: "cat on rooftop".into(),
                    frames: 97,
                    source_image: None,
                    negative_prompt: None,
                    seed_offset: None,
                    transition: TransitionMode::Cut,
                    fade_frames: None,
                    model: None,
                    loras: vec![],
                    references: vec![],
                },
                ChainStage {
                    prompt: "cat on moon".into(),
                    frames: 97,
                    source_image: None,
                    negative_prompt: None,
                    seed_offset: None,
                    transition: TransitionMode::Fade,
                    fade_frames: Some(6),
                    model: None,
                    loras: vec![],
                    references: vec![],
                },
            ],
        };
        let req = super::build_request_from_script(&script)
            .expect("script → request")
            .normalise()
            .expect("normalise");
        assert_eq!(req.stages.len(), 3);
        assert_eq!(req.stages[0].prompt, "cat in garden");
        assert_eq!(req.stages[1].prompt, "cat on rooftop");
        assert_eq!(req.stages[2].prompt, "cat on moon");
        assert_eq!(req.stages[0].transition, TransitionMode::Smooth);
        assert_eq!(req.stages[1].transition, TransitionMode::Cut);
        assert_eq!(req.stages[2].transition, TransitionMode::Fade);
        assert_eq!(req.stages[2].fade_frames, Some(6));
        assert_eq!(
            local_chain_planning_frames(&req),
            97,
            "local admission budgets one serial stage, not the 291-frame stitch"
        );
        // Auto-expand fields must be cleared by normalise so the server
        // can't confuse the two input shapes on receipt.
        assert!(req.prompt.is_none());
        assert!(req.total_frames.is_none());
        assert!(req.clip_frames.is_none());
    }
}
