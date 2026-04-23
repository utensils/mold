//! CLI-side render-chain orchestration for LTX-2 distilled models.
//!
//! When `mold run --frames N` exceeds the per-clip cap of the selected model,
//! this module takes over from [`super::generate::run`]: it assembles a
//! [`ChainRequest`] from the user's CLI args and either submits it to a
//! running server via [`MoldClient::generate_chain_stream`] or, in `--local`
//! mode, drives an in-process [`Ltx2ChainOrchestrator`].
//!
//! Both paths funnel through [`encode_and_save`] so stdout piping, gallery
//! save, metadata DB writes, and preview behaviour match the single-clip
//! path byte-for-byte.

use std::io::Write;
use std::time::Duration;

use anyhow::Result;
use colored::Colorize;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use mold_core::chain::{ChainProgressEvent, ChainRequest};
use mold_core::{Config, MoldClient, OutputFormat, VideoData};

use crate::control::CliContext;
use crate::output::{is_piped, status};
use crate::theme;

/// Per-clip frame cap for LTX-2 19B/22B distilled. The distilled VAE
/// pipeline maxes at 97 pixel frames (13 latent frames) per clip.
pub const LTX2_DISTILLED_CLIP_CAP: u32 = 97;

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

/// Pure decision function — given a model family, the user's requested
/// `frames`, and the optional `--clip-frames` override, decide whether to
/// chain, stay single-clip, or reject.
///
/// The clamp-to-cap behaviour surfaces through the returned `clip_frames`
/// field; callers warn the user via stderr when they had to clamp.
pub fn decide_chain_routing(
    frames: Option<u32>,
    family: Option<&str>,
    model: &str,
    clip_frames_flag: Option<u32>,
    motion_tail: u32,
) -> ChainRoutingDecision {
    let Some(total_frames) = frames else {
        return ChainRoutingDecision::SingleClip;
    };

    let is_ltx2_distilled = family == Some("ltx2") && model.contains("distilled");

    if !is_ltx2_distilled {
        // Non-chainable families: if the requested frame count is within a
        // conservative single-clip budget, stay on the single-clip path and
        // let the engine decide if it's acceptable. Otherwise, reject with
        // a clear message rather than silently over-producing.
        if total_frames <= LTX2_DISTILLED_CLIP_CAP {
            return ChainRoutingDecision::SingleClip;
        }
        return ChainRoutingDecision::Rejected {
            reason: format!(
                "model '{model}' does not support chained video generation \
                 (only LTX-2 distilled families do); specify --frames <= {} \
                 per clip for this model",
                LTX2_DISTILLED_CLIP_CAP,
            ),
        };
    }

    let cap = LTX2_DISTILLED_CLIP_CAP;
    let effective_clip_frames = clip_frames_flag.unwrap_or(cap).min(cap);

    if total_frames <= effective_clip_frames {
        return ChainRoutingDecision::SingleClip;
    }

    if motion_tail >= effective_clip_frames {
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
}

impl ChainInputs {
    pub(crate) fn to_chain_request(&self) -> ChainRequest {
        ChainRequest {
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
            prompt: Some(self.prompt.clone()),
            total_frames: Some(self.total_frames),
            clip_frames: Some(self.clip_frames),
            source_image: self.source_image.clone(),
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
    let video = if local {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            crate::ui::print_using_local_inference();
            run_chain_local(
                &req,
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
    )?;

    Config::write_last_model(&req.model);
    Ok(())
}

/// Remote chain: streaming SSE with stacked progress bars.
async fn run_chain_remote(client: &MoldClient, req: &ChainRequest) -> Result<VideoData> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ChainProgressEvent>();
    let stage_labels: Vec<StageLabel> = req.stages.iter().map(StageLabel::from_stage).collect();
    let render = tokio::spawn(render_chain_progress(rx, stage_labels));

    let stream_result = client.generate_chain_stream(req, tx).await;
    let _ = render.await;

    match stream_result {
        Ok(Some(resp)) => Ok(resp.video),
        Ok(None) => {
            // Server predates chain endpoint; fall back to non-streaming.
            status!(
                "{} Server SSE chain endpoint unavailable, falling back to blocking endpoint",
                theme::prefix_warning(),
            );
            let resp = client.generate_chain(req).await?;
            Ok(resp.video)
        }
        Err(e) => Err(e),
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
async fn run_chain_local(
    chain_req: &ChainRequest,
    config: &Config,
    gpus: Option<String>,
    t5_variant_override: Option<String>,
    qwen3_variant_override: Option<String>,
    qwen2_variant_override: Option<String>,
    qwen2_text_encoder_mode_override: Option<String>,
    eager: bool,
    offload: bool,
) -> Result<VideoData> {
    use mold_core::manifest::find_manifest;
    use mold_core::ModelPaths;
    use mold_inference::LoadStrategy;

    // Normalise so we have expanded stages locally too.
    let req = chain_req.clone().normalise()?;

    // Apply encoder-variant overrides before constructing the engine so the
    // factory's auto-select picks them up.
    apply_local_engine_env_overrides(
        t5_variant_override.as_deref(),
        qwen3_variant_override.as_deref(),
        qwen2_variant_override.as_deref(),
        qwen2_text_encoder_mode_override.as_deref(),
    );

    let model_name = req.model.clone();

    // Ensure the model is pulled + config rows are in place.
    let (paths, effective_config) = if let Some(p) = ModelPaths::resolve(&model_name, config) {
        (p, config.clone())
    } else if find_manifest(&model_name).is_some() {
        crate::output::status!(
            "{} Model '{}' not found locally, pulling...",
            theme::icon_info(),
            model_name.bold(),
        );
        let updated = super::pull::pull_and_configure(
            &model_name,
            &mold_core::download::PullOptions::default(),
        )
        .await?;
        let p = ModelPaths::resolve(&model_name, &updated).ok_or_else(|| {
            anyhow::anyhow!("model '{model_name}' was pulled but paths could not be resolved")
        })?;
        (p, updated)
    } else {
        anyhow::bail!(
            "no model paths configured for '{model_name}'. Add [models.{model_name}] \
             to ~/.mold/config.toml or pull via `mold pull {model_name}`."
        );
    };

    let is_eager = eager || std::env::var("MOLD_EAGER").is_ok_and(|v| v == "1");
    let load_strategy = if is_eager {
        LoadStrategy::Eager
    } else {
        LoadStrategy::Sequential
    };
    if is_eager {
        std::env::set_var("MOLD_EAGER", "1");
    }
    let is_offload = offload || std::env::var("MOLD_OFFLOAD").is_ok_and(|v| v == "1");

    let gpu_selection = match &gpus {
        Some(s) => mold_core::types::GpuSelection::parse(s)?,
        None => effective_config.gpu_selection(),
    };
    let discovered = mold_inference::device::discover_gpus();
    let available = mold_inference::device::filter_gpus(&discovered, &gpu_selection);
    let gpu_ordinal = mold_inference::device::select_best_gpu(&available)
        .map(|g| g.ordinal)
        .unwrap_or(0);

    let mut engine = mold_inference::create_engine(
        model_name,
        paths,
        &effective_config,
        load_strategy,
        gpu_ordinal,
        is_offload,
    )?;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ChainProgressEvent>();
    let stage_labels: Vec<StageLabel> = req.stages.iter().map(StageLabel::from_stage).collect();
    let render = tokio::spawn(render_chain_progress(rx, stage_labels));

    let fps = req.fps;
    let output_format = req.output_format;
    let total_frames_opt = Some(req.total_frames.unwrap_or(u32::MAX));
    let req_clone = req.clone();

    let handle = tokio::task::spawn_blocking(move || -> Result<VideoData> {
        engine.load()?;
        let renderer = engine.as_chain_renderer().ok_or_else(|| {
            anyhow::anyhow!(
                "model '{}' does not support chained video generation \
                 (only LTX-2 distilled engines expose a ChainStageRenderer view)",
                req_clone.model,
            )
        })?;
        let mut orch = mold_inference::ltx2::Ltx2ChainOrchestrator::new(renderer);

        let tx = tx;
        let mut chain_cb = move |event: ChainProgressEvent| {
            let _ = tx.send(event);
        };
        let chain_output = orch.run(&req_clone, Some(&mut chain_cb))?;

        use mold_inference::ltx2::stitch::StitchPlan;
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

        encode_local_frames(&frames, fps, output_format)
    });

    let result = handle.await??;
    let _ = render.await;
    Ok(result)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn apply_local_engine_env_overrides(
    t5_variant: Option<&str>,
    qwen3_variant: Option<&str>,
    qwen2_variant: Option<&str>,
    qwen2_text_encoder_mode: Option<&str>,
) {
    if let Some(v) = t5_variant {
        std::env::set_var("MOLD_T5_VARIANT", v);
    }
    if let Some(v) = qwen3_variant {
        std::env::set_var("MOLD_QWEN3_VARIANT", v);
    }
    if let Some(v) = qwen2_variant {
        std::env::set_var("MOLD_QWEN2_VARIANT", v);
    }
    if let Some(v) = qwen2_text_encoder_mode {
        std::env::set_var("MOLD_QWEN2_TEXT_ENCODER_MODE", v);
    }
}

/// Encode stitched frames to the requested container. MP4 is feature-gated;
/// fall back to APNG when the CLI was built without `mp4`.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn encode_local_frames(
    frames: &[image::RgbImage],
    fps: u32,
    output_format: OutputFormat,
) -> Result<VideoData> {
    use mold_inference::ltx_video::video_enc;

    let gif_preview = video_enc::encode_gif(frames, fps).unwrap_or_default();
    let thumbnail = video_enc::first_frame_png(frames).unwrap_or_default();

    let (bytes, actual_format) = match output_format {
        OutputFormat::Mp4 => {
            #[cfg(feature = "mp4")]
            {
                (video_enc::encode_mp4(frames, fps)?, OutputFormat::Mp4)
            }
            #[cfg(not(feature = "mp4"))]
            {
                crate::output::status!(
                    "{} MP4 requested but this binary was built without --features mp4; \
                     falling back to APNG",
                    theme::prefix_warning(),
                );
                (
                    video_enc::encode_apng(frames, fps, None)?,
                    OutputFormat::Apng,
                )
            }
        }
        OutputFormat::Apng => (
            video_enc::encode_apng(frames, fps, None)?,
            OutputFormat::Apng,
        ),
        OutputFormat::Gif => (video_enc::encode_gif(frames, fps)?, OutputFormat::Gif),
        OutputFormat::Webp => {
            crate::output::status!(
                "{} WebP chain output not supported locally yet; falling back to APNG",
                theme::prefix_warning(),
            );
            (
                video_enc::encode_apng(frames, fps, None)?,
                OutputFormat::Apng,
            )
        }
        other => anyhow::bail!("{other:?} is not a video output format for chain generation"),
    };

    let width = frames[0].width();
    let height = frames[0].height();
    let frame_count = frames.len() as u32;
    let duration_ms = if fps == 0 {
        None
    } else {
        Some((frame_count as u64 * 1000) / fps as u64)
    };

    Ok(VideoData {
        data: bytes,
        format: actual_format,
        width,
        height,
        frames: frame_count,
        fps,
        thumbnail,
        gif_preview,
        has_audio: false,
        duration_ms,
        audio_sample_rate: None,
        audio_channels: None,
    })
}

/// Shared epilogue: write the stitched video to stdout/file/gallery and
/// emit a terminal preview if requested. `req` is the normalised chain
/// request — `stages[0]` supplies the prompt/source image recorded in the
/// gallery metadata row.
fn encode_and_save(
    req: &ChainRequest,
    video: &VideoData,
    output: Option<&str>,
    preview: bool,
    elapsed_ms: u64,
    base_seed: u64,
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
            None => {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                Some(mold_core::default_output_filename(
                    &req.model,
                    timestamp,
                    video.format.extension(),
                    1,
                    0,
                ))
            }
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

            // Persist to the gallery metadata DB. Build a synthetic
            // GenerateRequest so the existing record_local_save helper can
            // infer dimensions/seed/steps/etc. without a dedicated chain
            // row schema.
            let synth = synth_generate_request(req, video);
            crate::metadata_db::record_local_save(
                std::path::Path::new(filename),
                &synth,
                req.seed.unwrap_or(base_seed),
                elapsed_ms,
                video.format,
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

/// Build a synthetic single-clip `GenerateRequest` from a normalised chain
/// so the gallery metadata DB can record the stitched output with the
/// existing row schema. Uses `stages[0]` for prompt + source image (the
/// gallery row only has one prompt field — multi-prompt chains lose the
/// continuation prompts in the DB, which is acceptable for v1).
fn synth_generate_request(req: &ChainRequest, video: &VideoData) -> mold_core::GenerateRequest {
    let first = req
        .stages
        .first()
        .expect("run_chain callers must pass a normalised ChainRequest");
    mold_core::GenerateRequest {
        prompt: first.prompt.clone(),
        negative_prompt: first.negative_prompt.clone(),
        model: req.model.clone(),
        width: req.width,
        height: req.height,
        steps: req.steps,
        guidance: req.guidance,
        seed: req.seed,
        batch_size: 1,
        output_format: video.format,
        embed_metadata: Some(false),
        scheduler: None,
        edit_images: None,
        source_image: first.source_image.clone(),
        strength: req.strength,
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: None,
        original_prompt: None,
        lora: None,
        frames: Some(video.frames),
        fps: Some(video.fps),
        upscale_model: None,
        gif_preview: false,
        enable_audio: None,
        audio_file: None,
        source_video: None,
        keyframes: None,
        pipeline: None,
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: req.placement.clone(),
    }
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

    let req = build_request_from_script(&script)?.normalise()?;

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

/// Build a canonical `ChainRequest` from the parsed TOML script.
/// The result still needs `normalise()` before use.
pub(crate) fn build_request_from_script(
    script: &mold_core::chain::ChainScript,
) -> anyhow::Result<ChainRequest> {
    Ok(ChainRequest {
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
        prompt: None,
        total_frames: None,
        clip_frames: None,
        source_image: None,
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

    // Cap clip_frames to LTX2_DISTILLED_CLIP_CAP if the user didn't override.
    let clip_frames = frames_per_clip
        .unwrap_or(LTX2_DISTILLED_CLIP_CAP)
        .min(LTX2_DISTILLED_CLIP_CAP);
    if let Some(requested) = frames_per_clip {
        if requested > LTX2_DISTILLED_CLIP_CAP {
            crate::output::status!(
                "{} --frames-per-clip {} exceeds LTX-2 cap {}, clamping to {}",
                theme::prefix_warning(),
                requested,
                LTX2_DISTILLED_CLIP_CAP,
                LTX2_DISTILLED_CLIP_CAP,
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

    // LTX-2 19B/22B distilled defaults: 1216×704, 24 fps, 8 steps, 3.0 guidance.
    let req = ChainRequest {
        model: model.clone(),
        stages,
        motion_tail_frames: motion_tail,
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
    .normalise()?;

    if dry_run {
        print_dry_run_summary(&req);
        return Ok(());
    }

    run_chain(
        req,
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

    #[test]
    fn routing_single_clip_under_cap() {
        let d = decide_chain_routing(Some(97), Some("ltx2"), "ltx-2-19b-distilled:fp8", None, 4);
        assert_eq!(d, ChainRoutingDecision::SingleClip);
    }

    #[test]
    fn routing_single_clip_when_frames_absent() {
        let d = decide_chain_routing(None, Some("ltx2"), "ltx-2-19b-distilled:fp8", None, 4);
        assert_eq!(d, ChainRoutingDecision::SingleClip);
    }

    #[test]
    fn routing_chain_over_cap_ltx2_distilled() {
        let d = decide_chain_routing(Some(200), Some("ltx2"), "ltx-2-19b-distilled:fp8", None, 4);
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
        let d = decide_chain_routing(Some(200), Some("flux"), "flux-dev:q4", None, 4);
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

    #[test]
    fn routing_rejects_non_ltx2_family_over_cap() {
        // ltx-video (not ltx2) is not chainable in v1.
        let d = decide_chain_routing(Some(200), Some("ltx-video"), "ltx-video:0.9.6", None, 4);
        assert!(matches!(d, ChainRoutingDecision::Rejected { .. }));
    }

    #[test]
    fn routing_clip_frames_above_cap_clamps_to_cap() {
        let d = decide_chain_routing(
            Some(300),
            Some("ltx2"),
            "ltx-2-19b-distilled:fp8",
            Some(200),
            4,
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
    fn routing_clip_frames_under_cap_respected() {
        let d = decide_chain_routing(
            Some(300),
            Some("ltx2"),
            "ltx-2-19b-distilled:fp8",
            Some(65),
            4,
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
        let d = decide_chain_routing(Some(200), Some("ltx2"), "ltx-2-19b-distilled:fp8", None, 97);
        assert!(matches!(d, ChainRoutingDecision::Rejected { .. }));
    }

    #[test]
    fn ltx2_distilled_cap_matches_engine_constraint() {
        // 97 = 8 * 12 + 1, satisfying the VAE 8k+1 constraint.
        assert_eq!(LTX2_DISTILLED_CLIP_CAP % 8, 1);
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

    /// Regression: `run_chain` used to round-trip multi-stage requests
    /// through `ChainInputs`, which collapsed everything into auto-expand
    /// form and silently replicated `stages[0].prompt` across every
    /// continuation. The gallery DB row (built by `synth_generate_request`)
    /// should now carry the authored stage-0 prompt and source image
    /// verbatim — and by construction, the caller has already preserved the
    /// downstream per-stage data in the `ChainRequest` it hands us.
    #[test]
    fn synth_generate_request_reads_stages_zero() {
        use mold_core::chain::{ChainStage, TransitionMode};
        let req = ChainRequest {
            model: "ltx-2-19b-distilled:fp8".into(),
            stages: vec![
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
                // A continuation stage with a DIFFERENT prompt and its own
                // source image — the old lossy code would have dropped both
                // and replicated stage-0's prompt. We're not asserting the
                // continuation shows up in the synth row (v1 gallery schema
                // only has one prompt field) — only that the stage-0 data
                // isn't overwritten by something smeared from the request.
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
            ],
            motion_tail_frames: 4,
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
        let video = VideoData {
            data: vec![],
            format: OutputFormat::Mp4,
            width: 1216,
            height: 704,
            frames: 190,
            fps: 24,
            thumbnail: vec![],
            gif_preview: vec![],
            has_audio: false,
            duration_ms: None,
            audio_sample_rate: None,
            audio_channels: None,
        };
        let synth = super::synth_generate_request(&req, &video);
        assert_eq!(synth.prompt, "stage zero prompt");
        assert_eq!(synth.source_image.as_deref(), Some(&[1, 2, 3, 4][..]));
        assert_eq!(synth.negative_prompt.as_deref(), Some("no cats"));
        assert_eq!(synth.model, "ltx-2-19b-distilled:fp8");
        assert_eq!(synth.seed, Some(42));
        assert_eq!(synth.frames, Some(190));
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
                motion_tail_frames: 4,
                output_format: OutputFormat::Mp4,
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
        // Auto-expand fields must be cleared by normalise so the server
        // can't confuse the two input shapes on receipt.
        assert!(req.prompt.is_none());
        assert!(req.total_frames.is_none());
        assert!(req.clip_frames.is_none());
    }
}
