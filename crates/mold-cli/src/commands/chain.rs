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
    fn to_chain_request(&self) -> ChainRequest {
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
#[allow(clippy::too_many_arguments)]
pub async fn run_chain(
    inputs: ChainInputs,
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
    // Validate the auto-expand form before touching the network / GPU so
    // obvious mistakes (bad clip_frames math, too many stages) fail fast.
    let chain_req = inputs.to_chain_request();
    let normalised = chain_req.clone().normalise()?;
    let stage_count = normalised.stages.len() as u32;

    status!(
        "{} Chain mode: {} frames → {} stages × {} frames (tail {})",
        theme::icon_mode(),
        inputs.total_frames,
        stage_count,
        inputs.clip_frames,
        inputs.motion_tail,
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
                &chain_req,
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
        run_chain_remote(ctx.client(), &chain_req).await?
    };

    let elapsed_ms = t0.elapsed().as_millis() as u64;
    let base_seed = inputs.seed.unwrap_or(0);

    encode_and_save(
        &inputs,
        &video,
        output.as_deref(),
        preview,
        elapsed_ms,
        base_seed,
    )?;

    Config::write_last_model(&inputs.model);
    Ok(())
}

/// Remote chain: streaming SSE with stacked progress bars.
async fn run_chain_remote(client: &MoldClient, req: &ChainRequest) -> Result<VideoData> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ChainProgressEvent>();
    let render = tokio::spawn(render_chain_progress(rx));

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
    let render = tokio::spawn(render_chain_progress(rx));

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

        let mut frames = chain_output.frames;
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
/// emit a terminal preview if requested.
fn encode_and_save(
    inputs: &ChainInputs,
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
                    &inputs.model,
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
            let req = synth_generate_request(inputs, video);
            crate::metadata_db::record_local_save(
                std::path::Path::new(filename),
                &req,
                inputs.seed.unwrap_or(base_seed),
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
        inputs.model.bold(),
        elapsed_ms as f64 / 1000.0,
        video.frames,
        inputs.seed.unwrap_or(base_seed),
    );

    Ok(())
}

fn synth_generate_request(inputs: &ChainInputs, video: &VideoData) -> mold_core::GenerateRequest {
    mold_core::GenerateRequest {
        prompt: inputs.prompt.clone(),
        negative_prompt: None,
        model: inputs.model.clone(),
        width: inputs.width,
        height: inputs.height,
        steps: inputs.steps,
        guidance: inputs.guidance,
        seed: inputs.seed,
        batch_size: 1,
        output_format: video.format,
        embed_metadata: Some(false),
        scheduler: None,
        edit_images: None,
        source_image: inputs.source_image.clone(),
        strength: inputs.strength,
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
        placement: inputs.placement.clone(),
    }
}

/// Stacked progress bars for chain render: a parent "Chain" bar covering
/// all pixel frames and a transient per-stage bar covering denoise steps.
async fn render_chain_progress(mut rx: tokio::sync::mpsc::UnboundedReceiver<ChainProgressEvent>) {
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
                parent.set_message(format!("stage {}/{}", stage_idx + 1, stage_count));
                let sb = mp.add(ProgressBar::new(0));
                sb.set_style(
                    ProgressStyle::default_bar()
                        .template(&format!(
                            "  Stage {{prefix}}  [{{bar:30.{c}/dim}}] {{pos}}/{{len}} steps",
                            c = theme::SPINNER_STYLE,
                        ))
                        .unwrap()
                        .progress_chars("━╸─"),
                );
                sb.set_prefix(format!("{}", stage_idx + 1));
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

    // Submit via the existing run_chain path.
    let inputs = chain_inputs_from_request(&req);
    run_chain(
        inputs,
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

/// Re-build a `ChainInputs` bundle from a normalised `ChainRequest` so the
/// existing `run_chain` helper can dispatch to the server or the local
/// orchestrator. Assumes `req.stages` is non-empty (guaranteed by
/// `normalise`).
fn chain_inputs_from_request(req: &ChainRequest) -> ChainInputs {
    let first = &req.stages[0];
    ChainInputs {
        prompt: first.prompt.clone(),
        model: req.model.clone(),
        width: req.width,
        height: req.height,
        steps: req.steps,
        guidance: req.guidance,
        strength: req.strength,
        seed: req.seed,
        fps: req.fps,
        output_format: req.output_format,
        total_frames: req.estimated_total_frames(),
        clip_frames: first.frames,
        motion_tail: req.motion_tail_frames,
        source_image: first.source_image.clone(),
        placement: req.placement.clone(),
    }
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
}
