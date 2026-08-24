use anyhow::Result;
#[cfg(any(feature = "preview", test))]
use base64::{engine::general_purpose, Engine as _};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use mold_core::{
    classify_generate_error, fit_to_model_dimensions_aligned, fit_to_target_area, manifest, Config,
    DevicePlacement, GenerateRequest, GenerateResponse, GenerateServerAction, GenerationReference,
    GenerationReferenceAuthority, ImageData, KeyframeCondition, LoraWeight, Ltx2GuidanceOverrides,
    Ltx2PipelineMode, Ltx2SpatialUpscale, Ltx2TemporalUpscale, MoldClient, OutputFormat,
    ReferenceUploadLease, ReferenceUploadSource, Scheduler, TimeRange,
};
use rand::Rng;
#[cfg(feature = "preview")]
use std::io::IsTerminal;
use std::io::Write;
use std::time::Duration;

use crate::commands::h3::ReferenceUpload;
use crate::control::{stream_server_pull, CliContext};
use crate::errors::RemoteInferenceError;
use crate::output::{is_piped, status};
use crate::theme;
use crate::ui::{print_server_pull_missing_model, print_using_local_inference, render_progress};

/// Tag an error as originating from the remote server so the top-level handler
/// can produce a remote-context diagnostic (e.g. distinguishing a server-side
/// CUDA OOM from a local Metal OOM on the client).
fn tag_remote(client: &MoldClient, e: anyhow::Error) -> anyhow::Error {
    RemoteInferenceError::wrap(client.host(), e)
}

/// Fit source image dimensions to the model's native resolution, preserving
/// aspect ratio on the resolved model's pixel grid (16 for most checkpoints,
/// 32 for e.g. `wan22-ti2v-5b`).
fn source_image_model_dimensions(
    bytes: &[u8],
    model_w: u32,
    model_h: u32,
    align: u32,
) -> Result<(u32, u32)> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode source image: {e}"))?;
    let orig_w = img.width();
    let orig_h = img.height();
    let (w, h) = fit_to_model_dimensions_aligned(orig_w, orig_h, model_w, model_h, align);
    if w != orig_w || h != orig_h {
        let is_upscale = w > orig_w || h > orig_h;
        let icon = if is_upscale {
            theme::icon_info()
        } else {
            theme::icon_warn()
        };
        status!(
            "{} Source image {}x{} -> {}x{} (fit to {}x{} model bounds, {}px aligned)",
            icon,
            orig_w,
            orig_h,
            w,
            h,
            model_w,
            model_h,
            align
        );
    }
    Ok((w, h))
}

fn qwen_image_edit_dimensions(bytes: &[u8]) -> Result<(u32, u32)> {
    const TARGET_AREA: u32 = 1024 * 1024;
    const ALIGN: u32 = 16;

    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode source image: {e}"))?;
    let orig_w = img.width().max(1);
    let orig_h = img.height().max(1);
    let (width, height) = fit_to_target_area(orig_w, orig_h, TARGET_AREA, ALIGN);

    if width != orig_w || height != orig_h {
        status!(
            "{} Edit image {}x{} -> {}x{} (target ~1024x1024 area, 16px aligned)",
            theme::icon_info(),
            orig_w,
            orig_h,
            width,
            height
        );
    }

    Ok((width, height))
}

pub(crate) fn resolve_family(model: &str, config: &Config) -> Option<String> {
    config
        .resolved_model_config(model)
        .family
        .or_else(|| manifest::find_manifest(model).map(|m| m.family.clone()))
}

fn require_local_request_model_activation(req: &GenerateRequest, config: &Config) -> Result<()> {
    let family = resolve_family(&req.model, config);
    let models_root = config.resolved_models_dir();
    mold_core::require_generate_request_model_activation(
        req,
        Some(&models_root),
        family.as_deref(),
    )?;
    crate::catalog_bridge::require_known_model_activation(config, &req.model)?;
    for identity in [req.control_model.as_deref(), req.upscale_model.as_deref()]
        .into_iter()
        .flatten()
    {
        crate::catalog_bridge::require_known_model_activation(config, identity)?;
    }
    Ok(())
}

/// Validate a request on the forced-local path.
///
/// Feeds the config/manifest-resolved family through as a hint so `cv:` / `hf:`
/// catalog IDs keep their family-gated features (audio, keyframes, retake,
/// optional prompts, …) instead of failing with `unknown model family`. This
/// mirrors what the HTTP server does with its catalog family hint.
#[cfg(any(feature = "cuda", feature = "metal", test))]
fn validate_local_request(req: &GenerateRequest, config: &Config) -> Result<()> {
    require_local_request_model_activation(req, config)?;
    mold_core::validate_generate_request_with_family(
        req,
        resolve_family(&req.model, config).as_deref(),
    )
    .map_err(|e| anyhow::anyhow!(e))?;
    if let Some(profile) = local_generation_profile(config, &req.model) {
        mold_core::validate_request_against_generation_profile(&profile, req)
            .map_err(anyhow::Error::msg)?;
    }
    Ok(())
}

/// Ask the server whether it understands this request's additive identity
/// shapes, and refuse rather than submit when it does not.
///
/// Returns immediately — with no round trip at all — for every request that
/// does not carry several photographs or engage true CFG, which is every
/// ordinary render and every single-photograph identity render.
///
/// A probe that cannot REACH the server is deliberately not a refusal. The
/// caller's next step is the ordinary remote attempt, which fails the same way
/// and falls back to local inference — where both shapes are honoured in full.
/// Turning an unreachable host into a hard error here would take that fallback
/// away from exactly the requests that need it most.
///
/// Every OTHER failure is a refusal, and that distinction is the whole point. A
/// reachable server that predates `/api/capabilities`, or whose response cannot
/// be decoded, answers this probe with a 404 or a parse error — and then
/// ACCEPTS the generate request while dropping the very fields we asked about.
/// Treating that like an unreachable host is the silent accept-and-ignore all
/// over again, so it is read as "identity unsupported" instead and refused
/// through the same named message. `MoldClient::is_connection_error` is the
/// existing authority for which failures mean "server unreachable, fall back to
/// local"; this reuses it rather than inventing a second rule.
async fn require_remote_identity_capabilities(
    client: &mold_core::MoldClient,
    request: &mold_core::GenerateRequest,
) -> Result<()> {
    if !crate::commands::identity::request_needs_identity_capabilities(request) {
        return Ok(());
    }
    let capabilities = match client.server_capabilities().await {
        Ok(capabilities) => capabilities,
        Err(error) if mold_core::MoldClient::is_connection_error(&error) => return Ok(()),
        // Reachable, but it could not tell us — which is exactly how an older
        // server answers. Absence is NO.
        Err(_) => mold_core::ServerCapabilities::default(),
    };
    crate::commands::identity::ensure_server_understands_identity(
        request,
        &capabilities,
        client.host(),
    )
}

fn local_generation_delivery_capabilities() -> mold_core::GenerationDeliveryCapabilities {
    mold_core::GenerationDeliveryCapabilities::new(cfg!(feature = "mp4"), cfg!(feature = "webp"))
}

/// The encoder that will produce this run's bytes: this build's linked
/// features when it renders locally, and everything when a server does —
/// there the remote refuses at admission what it cannot deliver, still
/// before anything is written to the path the user named.
pub(crate) fn delivery_capabilities_for_run(
    local: bool,
) -> mold_core::GenerationDeliveryCapabilities {
    if local {
        local_generation_delivery_capabilities()
    } else {
        mold_core::GenerationDeliveryCapabilities::new(true, true)
    }
}

fn local_generation_profile(
    config: &Config,
    model: &str,
) -> Option<mold_core::GenerationProfileSet> {
    let family = resolve_family(model, config);
    mold_core::require_model_activation(model, family.as_deref()).ok()?;
    let canonical = manifest::resolve_model_name(model);
    let mut catalog = mold_core::build_model_catalog(config, None, false);
    mold_core::qualify_catalog_generation_delivery(
        &mut catalog,
        local_generation_delivery_capabilities(),
    );
    catalog
        .into_iter()
        .find(|entry| entry.info.name == model || entry.info.name == canonical)
        .and_then(|entry| entry.generation_profile)
}

#[allow(clippy::too_many_arguments)]
fn effective_dimensions(
    config: &Config,
    model_cfg: &mold_core::ModelConfig,
    model: &str,
    family: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
    source_image: Option<&[u8]>,
    edit_images: Option<&[Vec<u8>]>,
) -> Result<(u32, u32)> {
    if family.is_some_and(mold_core::minimax_h3::is_family) {
        return match (width, height, source_image) {
            (Some(width), Some(height), _) => Ok((width, height)),
            (Some(width), None, _) => Ok((width, model_cfg.effective_height(config))),
            (None, Some(height), _) => Ok((model_cfg.effective_width(config), height)),
            (None, None, Some(source_image)) => {
                let image = image::load_from_memory(source_image)
                    .map_err(|error| anyhow::anyhow!("failed to decode H3 first frame: {error}"))?;
                // Compact layouts (base + Turbo tags) must submit the fixed
                // reviewed envelope regardless of source aspect; only the
                // official BF16 reference keeps the aspect-derived canvas.
                Ok(mold_core::minimax_h3::source_fit_dimensions(
                    model,
                    image.width(),
                    image.height(),
                ))
            }
            (None, None, None) => Ok((
                mold_core::minimax_h3::DEFAULT_WIDTH,
                mold_core::minimax_h3::DEFAULT_HEIGHT,
            )),
        };
    }
    match (width, height, source_image) {
        (Some(width), Some(height), _) => Ok((width, height)),
        (Some(width), None, _) => Ok((width, model_cfg.effective_height(config))),
        (None, Some(height), _) => Ok((model_cfg.effective_width(config), height)),
        (None, None, _) if family == Some("qwen-image-edit") => {
            let first = edit_images
                .and_then(|images| images.first())
                .ok_or_else(|| {
                    anyhow::anyhow!("qwen-image-edit requires at least one input image")
                })?;
            qwen_image_edit_dimensions(first)
        }
        (None, None, Some(source_image)) => {
            let model_w = model_cfg.effective_width(config);
            let model_h = model_cfg.effective_height(config);
            // Per-model grid: `wan22-ti2v-5b` needs 32, most models 16.
            let align = mold_core::dimension_alignment_for_model(model, family);
            source_image_model_dimensions(source_image, model_w, model_h, align)
        }
        (None, None, None) => Ok((
            model_cfg.effective_width(config),
            model_cfg.effective_height(config),
        )),
    }
}

/// Steps the request submits when the user passed no `--steps`.
///
/// MiniMax H3's step authority is per manifest tier — 21 for the compact
/// base, 9/5 for the reviewed Turbo tags, 50 for the hidden official BF16
/// references — so the resolved model config's manifest-merged default is the
/// answer. Neither the official-BF16 constant nor the global image default
/// may leak in: the server refuses off-envelope steps only after the whole
/// admission cost has been paid.
fn effective_generation_steps(
    is_h3: bool,
    model: &str,
    model_cfg: &mold_core::ModelConfig,
    config: &Config,
    steps: Option<u32>,
) -> u32 {
    if let Some(steps) = steps {
        return steps;
    }
    if is_h3 {
        // Compact layouts (base + Turbo tags) run exactly their tier's
        // reviewed step count, so it comes from the identity — never from
        // `default_steps`, which `build_model_catalog` launders a user
        // `model_pref` into and which can therefore be stale. Only the
        // official BF16 reference keeps the stored value; its ladder really
        // is adjustable. An unrecognized identity takes the compact answer,
        // the same fail-toward-the-stricter-contract rule as
        // `source_fit_dimensions` above.
        if mold_core::minimax_h3::layout_for_model(model)
            != Some(mold_core::minimax_h3::Layout::OfficialBf16)
        {
            return mold_core::minimax_h3::turbo_tier_for_model(model)
                .map(|tier| tier.steps)
                .unwrap_or(mold_core::minimax_h3::COMFY_DEFAULT_STEPS);
        }
        model_cfg
            .default_steps
            .unwrap_or(mold_core::minimax_h3::COMFY_DEFAULT_STEPS)
    } else {
        model_cfg.effective_steps(config)
    }
}

fn effective_negative_prompt(
    family: Option<&str>,
    guidance: f64,
    negative_prompt: Option<String>,
) -> Option<String> {
    if family == Some("qwen-image-edit") && guidance > 1.0 && negative_prompt.is_none() {
        // Keep CFG's negative branch explicitly populated for Qwen edit models.
        Some(" ".to_string())
    } else {
        negative_prompt
    }
}

fn validate_cli_batch_for_family(family: Option<&str>, batch: u32) -> Result<()> {
    if family == Some("qwen-image-edit") && batch > 1 {
        anyhow::bail!(
            "qwen-image-edit only supports --batch 1. Run separate commands for multiple edits."
        );
    }
    Ok(())
}

fn validate_batch_stdout(batch: u32, piped: bool, output: &Option<String>) -> Result<()> {
    if batch > 1 {
        let stdout_output = (piped && output.is_none()) || output.as_deref() == Some("-");
        if stdout_output {
            anyhow::bail!(
                "--batch with more than 1 output is not supported with stdout output. \
                 Use --output <path> to save batch outputs to separate files."
            );
        }
    }
    Ok(())
}

/// Re-derive request defaults after an auto-pull refreshed the model
/// config. `cli_*` carry the user's explicit flags — `None` means the
/// field was defaulted at request-build time and must now track the
/// refreshed model defaults. Reuses `effective_dimensions`, so img2img
/// sources are re-fit to the refreshed canvas and qwen-image-edit sizes
/// from its first edit image.
#[cfg(any(feature = "cuda", feature = "metal", test))]
#[allow(clippy::too_many_arguments)]
fn refit_request_after_pull(
    req: &mut GenerateRequest,
    config: &Config,
    model_cfg: &mold_core::ModelConfig,
    family: Option<&str>,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
) -> Result<()> {
    if cli_width.is_none() && cli_height.is_none() {
        let (w, h) = effective_dimensions(
            config,
            model_cfg,
            &req.model,
            family,
            None,
            None,
            req.source_image.as_deref(),
            req.edit_images.as_deref(),
        )?;
        req.width = w;
        req.height = h;
    } else {
        if cli_width.is_none() {
            req.width = model_cfg.effective_width(config);
        }
        if cli_height.is_none() {
            req.height = model_cfg.effective_height(config);
        }
    }
    if cli_steps.is_none() {
        req.steps = effective_generation_steps(
            family.is_some_and(mold_core::minimax_h3::is_family),
            &req.model,
            model_cfg,
            config,
            None,
        );
    }
    if cli_guidance.is_none() {
        req.guidance = model_cfg.effective_guidance();
    }
    Ok(())
}

pub struct Ltx2Options {
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    /// Per-clip cap for chained rendering. `None` = use the model-family default
    /// (currently 97 for LTX-2 distilled). Only read when `frames > cap`.
    pub clip_frames: Option<u32>,
    /// Motion-tail overlap between chained clips (pixel frames).
    pub motion_tail: u32,
    pub enable_audio: Option<bool>,
    pub audio_file: Option<Vec<u8>>,
    pub source_video: Option<Vec<u8>>,
    /// Existing video to continue. Makes the request a continuation: the
    /// delivered output is this clip followed by the newly rendered frames.
    pub extend_video: Option<Vec<u8>>,
    /// Pixel-frame overlap the continuation conditions on. `None` uses the
    /// shared chain motion-tail default.
    pub extend_overlap_frames: Option<u32>,
    pub keyframes: Option<Vec<KeyframeCondition>>,
    pub pipeline: Option<Ltx2PipelineMode>,
    pub ic_lora_control: Option<String>,
    pub hdr_exr_dir: Option<String>,
    pub hdr_exr_full_float: bool,
    pub loras: Option<Vec<LoraWeight>>,
    pub retake_range: Option<TimeRange>,
    pub spatial_upscale: Option<Ltx2SpatialUpscale>,
    pub temporal_upscale: Option<Ltx2TemporalUpscale>,
    /// Advanced multimodal-guider overrides. `None` keeps every pipeline
    /// constant, which is what makes existing seeds reproducible.
    pub guidance_overrides: Option<Ltx2GuidanceOverrides>,
    /// Wan flow shift (#782). `None` keeps the per-tier default.
    pub sample_shift: Option<f64>,
    /// Wan Lightning distill strengths (#795). `None` = 1.0.
    pub distill_strength_high: Option<f64>,
    pub distill_strength_low: Option<f64>,
    /// Display-safe first-frame provenance. Never a client path.
    pub source_image_name: Option<String>,
    /// Payload-free ordered H3 Ref2VA descriptors.
    pub references: Option<Vec<GenerationReference>>,
    /// Local files corresponding one-for-one with `references`. They are
    /// streamed only after an authenticated request-bound session is created.
    pub reference_uploads: Vec<ReferenceUpload>,
}

fn require_local_hdr_exr_dir(hdr_exr_dir: Option<String>, local: bool) -> Result<Option<String>> {
    if hdr_exr_dir.is_some() && !local {
        anyhow::bail!(
            "--hdr-exr-dir renders locally; re-run with --local — a remote server cannot write \
             the EXR sidecar to your machine"
        );
    }
    Ok(hdr_exr_dir)
}

/// Creation-time filing chosen on the command line: `--tag`, `--collection`,
/// and `--no-auto-tag`.
///
/// Bundled rather than three more positional parameters on a signature that
/// already carries fifty, and because the three are resolved together —
/// whether the title becomes a tag depends on all of them plus config.
#[derive(Debug, Clone, Default)]
pub struct FilingOptions {
    /// Tags from repeated `--tag`, already validated by the value parser.
    pub tags: Vec<String>,
    /// Collection display name from `--collection`, already validated.
    pub collection: Option<String>,
    /// `--no-auto-tag`: never add the title as a tag, whatever
    /// `generate.auto_tag_title` says.
    pub no_auto_tag: bool,
}

impl FilingOptions {
    /// Resolve the filing into the wire fields a request carries, plus the
    /// disclosure line for a tag the user did not type.
    ///
    /// `auto_tag_title` is the effective `generate.auto_tag_title` setting;
    /// `--no-auto-tag` overrides it for this invocation only.
    fn resolve(
        &self,
        title: Option<&str>,
        auto_tag_title: bool,
    ) -> Result<ResolvedFiling, anyhow::Error> {
        let composed =
            mold_core::compose_client_tags(&self.tags, title, auto_tag_title && !self.no_auto_tag)
                .map_err(|error| anyhow::anyhow!(error))?;
        Ok(ResolvedFiling {
            tags: (!composed.tags.is_empty()).then_some(composed.tags),
            collection: self
                .collection
                .as_deref()
                .map(mold_core::CollectionRef::by_name),
            auto_tagged: composed.auto_tagged,
        })
    }
}

/// [`FilingOptions`] resolved against the title and the effective config.
struct ResolvedFiling {
    tags: Option<Vec<String>>,
    collection: Option<mold_core::CollectionRef>,
    /// The tag added from the title, when one was — disclosed on stderr so a
    /// tag the user did not type is never a surprise found later.
    auto_tagged: Option<String>,
}

#[allow(clippy::too_many_arguments)]
pub async fn run(
    prompt: &str,
    model: &str,
    output: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    guidance: Option<f64>,
    seed: Option<u64>,
    batch: u32,
    ltx2: Ltx2Options,
    host: Option<String>,
    format: OutputFormat,
    no_metadata: bool,
    title: Option<String>,
    filing: FilingOptions,
    preview: bool,
    local: bool,
    gpus: Option<String>,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Option<String>,
    scheduler: Option<Scheduler>,
    cfg_plus: Option<bool>,
    eager: bool,
    offload: bool,
    placement: Option<DevicePlacement>,
    source_image: Option<Vec<u8>>,
    edit_images: Option<Vec<Vec<u8>>>,
    strength: f64,
    mask_image: Option<Vec<u8>>,
    control_image: Option<Vec<u8>>,
    control_model: Option<String>,
    control_scale: f64,
    negative_prompt: Option<String>,
    original_prompt: Option<String>,
    batch_prompts: Option<Vec<String>>,
    lora: Option<LoraWeight>,
    expand: Option<bool>,
    // PuLID face identity, already read and validated by
    // `commands::identity`. The SAME value serves the remote and the
    // forced-local path, which is what makes the two requests identical by
    // construction rather than by review.
    identity: crate::commands::identity::IdentityOptions,
) -> Result<()> {
    let Ltx2Options {
        frames,
        fps,
        clip_frames,
        motion_tail,
        enable_audio,
        audio_file,
        source_video,
        extend_video,
        extend_overlap_frames,
        keyframes,
        pipeline,
        ic_lora_control,
        hdr_exr_dir,
        hdr_exr_full_float,
        loras,
        retake_range,
        spatial_upscale,
        temporal_upscale,
        guidance_overrides,
        sample_shift,
        distill_strength_high,
        distill_strength_low,
        source_image_name,
        references,
        reference_uploads,
    } = ltx2;
    // Keep the filesystem path out of every remote request shape. Local
    // generation retains the exact user path so export metadata remains
    // faithful to the sidecar written on this machine.
    let hdr_exr_dir = require_local_hdr_exr_dir(hdr_exr_dir, local)?;

    // Load config and pull model-specific defaults.
    let ctx = CliContext::new(host.as_deref());
    let mut config = ctx.config().clone();
    // Catalog bridge: when the user passed `cv:<id>` / `hf:<author>/<name>`,
    // resolve it (sidecar-first, then live) and inject the synthesized
    // `ModelConfig` into `config.models` so the downstream
    // `ModelPaths::resolve` and engine factory accept it. A no-op for
    // non-catalog model names. Catalog lookup failures now surface here
    // rather than being swallowed and re-reported as a generic "unknown
    // model" downstream.
    crate::catalog_bridge::ensure_catalog_model(&mut config, model).await?;
    // `--no-metadata` is an opt-out override, so we only pass `Some(false)` when set.
    // Otherwise we defer to env/config/default precedence inside Config.
    let embed_metadata = config.effective_embed_metadata(no_metadata.then_some(false));
    let model_cfg = config.resolved_model_config(model);
    let family = resolve_family(model, &config);
    let is_h3 = family
        .as_deref()
        .is_some_and(mold_core::minimax_h3::is_family);
    if is_h3 && !reference_uploads.is_empty() && batch != 1 {
        anyhow::bail!(
            "MiniMax H3 ordered references require batch 1 because upload handles are one-use and request-bound"
        );
    }
    if is_h3
        && references.as_ref().is_some_and(|references| {
            references.iter().any(|reference| {
                matches!(reference.media(), GenerationReferenceAuthority::Descriptor)
            })
        })
        && reference_uploads.is_empty()
    {
        anyhow::bail!(
            "MiniMax H3 reference descriptors require authenticated streaming upload sources"
        );
    }
    let effective_frames = if is_h3 {
        Some(frames.unwrap_or(mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES))
    } else {
        frames.or_else(|| model_cfg.effective_frames())
    };
    let effective_fps = if is_h3 {
        Some(mold_core::minimax_h3::FIXED_FPS)
    } else {
        fps.or_else(|| model_cfg.effective_fps())
    };
    let is_ltx2 = family.as_deref() == Some("ltx2");
    validate_cli_batch_for_family(family.as_deref(), batch)?;

    // Default video models to a sensible container unless the user explicitly picked one.
    // An audio-only pipeline goes to WAV instead: it emits no frames, so both
    // the raster default and the video default would be rejected outright.
    let audio_only_pipeline = ltx2
        .pipeline
        .is_some_and(mold_core::Ltx2PipelineMode::is_audio_only);
    if local {
        // Ask the activation question before the profile lookup. A model this
        // build cannot execute has no runtime recipe *because* it is refused,
        // so looking the recipe up first reports the symptom ("no generation
        // profile") instead of the cause ("no runtime for this layout").
        mold_core::require_model_activation(model, family.as_deref())?;
    }
    let local_profile = if local {
        let profile = local_generation_profile(&config, model).ok_or_else(|| {
            anyhow::anyhow!("no generation profile is available for local model '{model}'")
        })?;
        // Resolve the recipe even for an explicit format. This rejects an
        // MP4-only model such as H3 immediately in a featureless local build,
        // before model download or inference can begin.
        mold_core::generation_profile_default_output_format(&profile, ltx2.pipeline)
            .map_err(anyhow::Error::msg)?;
        Some(profile)
    } else {
        None
    };
    let output_format = if format == OutputFormat::Png && effective_frames.is_some() {
        if let Some(profile) = local_profile.as_ref() {
            mold_core::generation_profile_default_output_format(profile, ltx2.pipeline)
                .map_err(anyhow::Error::msg)?
        } else {
            default_output_format(
                family.as_deref(),
                format,
                effective_frames,
                audio_only_pipeline,
                is_h3,
                cfg!(feature = "mp4"),
            )
        }
    } else {
        default_output_format(
            family.as_deref(),
            format,
            effective_frames,
            audio_only_pipeline,
            is_h3,
            cfg!(feature = "mp4"),
        )
    };
    // The container above is a family/recipe decision that never saw the
    // filename. Reconcile the two here — above the family policy, so wan,
    // ltx-video and LTX-2 all get it, and before any weight is read (#1050).
    let output_format = reconcile_video_format_with_output_extension(
        output_format,
        output.as_deref(),
        format != OutputFormat::Png,
        delivery_capabilities_for_run(local),
    )
    .map_err(anyhow::Error::msg)?;

    // ── Chain routing ─────────────────────────────────────────────────────
    // When --frames exceeds the per-clip cap, auto-build a ChainRequest and
    // delegate to the chain helper. LTX-2 and wan auto-chain; other video
    // families error fast rather than silently over-producing.
    // `decide_chain_routing` declines outright for an audio-only pipeline.
    {
        use super::chain::{decide_chain_routing, warn_if_clamped, ChainRoutingDecision};
        let routing_fps = effective_fps.unwrap_or(mold_core::validation::LTX2_DEFAULT_FPS);
        // Wan's seam is last-frame image conditioning, so the routing needs
        // the checkpoint's own source-image contract (#783). The manifest is
        // where it is recorded; an unknown model yields `None`, which routes
        // to independent clips rather than an assumed handoff.
        let source_image_contract =
            mold_core::manifest::find_manifest(&mold_core::manifest::resolve_model_name(model))
                .and_then(|manifest| manifest.defaults.source_image);
        let decision = decide_chain_routing(
            effective_frames,
            family.as_deref(),
            model,
            clip_frames,
            motion_tail,
            routing_fps,
            ltx2.pipeline,
            source_image_contract,
        );
        match decision {
            ChainRoutingDecision::SingleClip => {
                // Fall through to the existing single-clip path below.
            }
            ChainRoutingDecision::Rejected { reason } => {
                anyhow::bail!(reason);
            }
            ChainRoutingDecision::Chain {
                clip_frames: cf,
                motion_tail: mt,
            } => {
                // Chained EXR export reaches this arm only after the shared
                // local-only gate above. The forced-local orchestrator carries
                // the whole HDR authority (control id, adapter LoRAs, and
                // per-stage reference windows), so every stage renders a true
                // regrade of its own slice of the reference (#688).
                warn_if_clamped(
                    clip_frames,
                    family
                        .as_deref()
                        .and_then(|family| {
                            mold_core::validation::max_frames_for_family_at_fps(family, routing_fps)
                        })
                        .unwrap_or(super::chain::LTX2_DEFAULT_CLIP_FRAMES),
                );
                let (eff_w, eff_h) = effective_dimensions(
                    &config,
                    &model_cfg,
                    model,
                    family.as_deref(),
                    width,
                    height,
                    source_image.as_deref(),
                    edit_images.as_deref(),
                )?;
                let eff_steps = steps.unwrap_or_else(|| model_cfg.effective_steps(&config));
                let eff_guidance = guidance.unwrap_or_else(|| model_cfg.effective_guidance());
                let eff_fps = effective_fps.unwrap_or(24);
                let total_frames = effective_frames
                    .expect("decide_chain_routing only returns Chain when frames is Some");

                // Chain path doesn't use batch/edit_images/mask/control/loras —
                // those are single-clip concepts. If the user set them, warn and
                // continue (we don't hard-error to keep the UX lenient).
                if batch > 1 {
                    status!(
                        "{} --batch has no effect in chain mode; rendering a single stitched video",
                        theme::icon_warn(),
                    );
                }

                let inputs = super::chain::ChainInputs {
                    prompt: prompt.to_string(),
                    model: model.to_string(),
                    width: eff_w,
                    height: eff_h,
                    steps: eff_steps,
                    guidance: eff_guidance,
                    strength,
                    seed,
                    fps: eff_fps,
                    output_format,
                    total_frames,
                    clip_frames: cf,
                    motion_tail: mt,
                    source_image: source_image.clone(),
                    placement: placement.clone(),
                    enable_audio,
                    original_prompt: original_prompt.clone(),
                    batch_id: None,
                    batch_index: None,
                    batch_count: None,
                };
                // Guidance overrides are a single-clip concept: chain stages
                // render through their own pipeline constants, so say so
                // rather than pretend the flags took effect.
                if guidance_overrides.is_some() {
                    status!(
                        "{} guidance overrides have no effect in chain mode; each clip keeps its pipeline defaults",
                        theme::icon_warn(),
                    );
                }
                // Consume otherwise-unused LTX-2 knobs that chain v1 ignores so
                // clippy doesn't fire `unused_variables` on the early return.
                let _ = (
                    &guidance_overrides,
                    &audio_file,
                    &source_video,
                    &keyframes,
                    &pipeline,
                    &loras,
                    &retake_range,
                    &spatial_upscale,
                    &temporal_upscale,
                    &mask_image,
                    &control_image,
                    &control_model,
                    control_scale,
                    &negative_prompt,
                    &original_prompt,
                    &batch_prompts,
                    &lora,
                    &scheduler,
                    expand,
                );
                // Chained HDR: resolve the adapter + validate the request
                // shape through the same production path a single-clip HDR
                // render uses (which also downloads the gated adapter), then
                // hand the whole authority to the orchestrator.
                let hdr_chain = if let Some(dir) = hdr_exr_dir.clone() {
                    let reference_video = source_video.clone().ok_or_else(|| {
                        anyhow::anyhow!(
                            "--hdr-exr-dir requires --video <reference>: the HDR adapter \
                             regrades an SDR reference clip"
                        )
                    })?;
                    let control = ic_lora_control.clone().unwrap_or_else(|| "hdr".to_string());
                    let mut probe_req = GenerateRequest {
                        collection: None,
                        tags: None,
                        title: None,
                        source_fit: None,
                        hdr_exr_dir: Some(dir.clone()),
                        hdr_exr_full_float,
                        guidance_overrides: None,
                        sample_shift: None,
                        distill_strength_high: None,
                        distill_strength_low: None,
                        prompt: prompt.to_string(),
                        negative_prompt: None,
                        model: model.to_string(),
                        width: eff_w,
                        height: eff_h,
                        steps: eff_steps,
                        guidance: eff_guidance,
                        seed,
                        batch_size: 1,
                        output_format: Some(OutputFormat::Mp4),
                        embed_metadata: None,
                        scheduler: None,
                        cfg_plus: None,
                        source_image: None,
                        source_image_name: None,
                        edit_images: None,
                        references: None,
                        strength: 1.0,
                        mask_image: None,
                        control_image: None,
                        control_model: None,
                        control_scale: 1.0,
                        expand: None,
                        original_prompt: None,
                        prompt_transform: None,
                        batch_id: None,
                        batch_index: None,
                        batch_count: None,
                        lora: lora.clone(),
                        frames: Some(cf),
                        fps: Some(eff_fps),
                        upscale_model: None,
                        gif_preview: false,
                        enable_audio: Some(false),
                        audio_file: None,
                        audio_file_path: None,
                        source_video: Some(reference_video.clone()),
                        source_video_path: None,
                        extend_video: None,
                        extend_video_path: None,
                        extend_overlap_frames: None,
                        keyframes: None,
                        pipeline: Some(pipeline.unwrap_or(mold_core::Ltx2PipelineMode::IcLora)),
                        ic_lora_control: Some(control.clone()),
                        loras: loras.clone(),
                        retake_range: None,
                        spatial_upscale: None,
                        temporal_upscale: None,
                        placement: placement.clone(),
                        id_image: None,
                        id_image_name: None,
                        id_weight: None,
                        id_start_step: None,
                        id_images: None,
                        id_image_names: None,
                        true_cfg: None,
                        cfg_start_step: None,
                    };
                    materialize_local_builtin_control(&mut probe_req, &config).await?;
                    let control_loras = probe_req.loras.take().unwrap_or_default();
                    Some(super::chain::ChainHdrInputs {
                        exr_dir: dir,
                        full_float: hdr_exr_full_float,
                        control,
                        reference_video,
                        control_loras,
                        total_frames,
                    })
                } else {
                    None
                };

                // Expand the single-prompt inputs into a canonical
                // ChainRequest and normalise before handing off to run_chain.
                let mut chain_req = inputs.to_chain_request().normalise()?;
                if hdr_chain.is_some() {
                    // The EXR sequence and the reference slices both follow
                    // the stitched timeline, so the chain must deliver
                    // exactly the requested total — no overshoot to
                    // truncate, no stage rendering past the reference's end.
                    super::chain::exact_fit_last_stage_for_sidecar(&mut chain_req, total_frames)?;
                }
                return super::chain::run_chain(
                    chain_req,
                    hdr_chain,
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
                .await;
            }
        }
    }

    let piped = is_piped();

    validate_batch_stdout(batch, piped, &output)?;

    // Default to the source image size for img2img/inpainting when neither
    // dimension was provided. We still normalize to the validation envelope
    // (multiples of 16, megapixel clamp) to avoid invalid requests and OOMs.
    let (effective_width, effective_height) = effective_dimensions(
        &config,
        &model_cfg,
        model,
        family.as_deref(),
        width,
        height,
        source_image.as_deref(),
        edit_images.as_deref(),
    )?;
    let effective_steps = effective_generation_steps(is_h3, model, &model_cfg, &config, steps);
    let guidance_caps = mold_core::GuidanceCapabilities::for_recipe(
        family.as_deref().unwrap_or_default(),
        model,
        pipeline,
    );
    let effective_guidance = if is_h3 {
        0.0
    } else {
        guidance_caps.resolve_scale(guidance, model_cfg.effective_guidance())
    };
    let effective_negative_prompt = if is_h3 {
        None
    } else {
        effective_negative_prompt(
            family.as_deref(),
            effective_guidance,
            negative_prompt.clone(),
        )
    };

    // Creation-time filing is a client decision (see
    // `mold_core::compose_client_tags`), resolved here so both the remote and
    // the forced-local path submit the identical tag list.
    let resolved_filing = filing.resolve(title.as_deref(), config.generate.auto_tag_title)?;
    if let Some(auto_tagged) = resolved_filing.auto_tagged.as_deref() {
        // A tag the user did not type must be visible now, not discovered
        // later in the Library.
        status!("filing under tag \"{auto_tagged}\"");
    }

    let mut req = GenerateRequest {
        collection: resolved_filing.collection,
        tags: resolved_filing.tags,
        title,
        source_fit: None,
        hdr_exr_dir,
        hdr_exr_full_float,
        guidance_overrides,
        sample_shift,
        distill_strength_high,
        distill_strength_low,
        prompt: prompt.to_string(),
        negative_prompt: effective_negative_prompt.clone(),
        model: model.to_string(),
        width: effective_width,
        height: effective_height,
        steps: effective_steps,
        guidance: effective_guidance,
        seed,
        batch_size: batch,
        output_format: Some(output_format),
        embed_metadata: Some(embed_metadata),
        scheduler: (!is_h3).then_some(scheduler).flatten(),
        cfg_plus: (!is_h3).then_some(cfg_plus).flatten(),
        edit_images: edit_images.clone(),
        references,
        source_image: source_image.clone(),
        source_image_name,
        strength: if is_h3 { 1.0 } else { strength },
        mask_image: mask_image.clone(),
        control_image: control_image.clone(),
        control_model: control_model.clone(),
        control_scale,
        expand,
        original_prompt,
        prompt_transform: None,
        batch_id: None,
        batch_index: None,
        batch_count: None,
        lora: lora.clone(),
        frames: effective_frames,
        fps: effective_fps,
        upscale_model: None,
        gif_preview: preview,
        enable_audio: if is_h3 { Some(true) } else { enable_audio },
        audio_file,
        audio_file_path: None,
        source_video,
        source_video_path: None,
        extend_video,
        extend_video_path: None,
        extend_overlap_frames,
        keyframes,
        pipeline,
        ic_lora_control,
        loras,
        retake_range,
        spatial_upscale,
        temporal_upscale,
        placement,
        id_image: identity.id_image,
        id_image_name: identity.id_image_name,
        id_images: identity.id_images,
        id_image_names: identity.id_image_names,
        id_weight: identity.id_weight,
        id_start_step: identity.id_start_step,
        true_cfg: identity.true_cfg,
        cfg_start_step: identity.cfg_start_step,
    };
    // A continuation that named no overlap renders with its family's own
    // carryover, and the metadata `record_local_save` builds resolves the
    // family through the manifest — which an installed `cv:` / `hf:` wan
    // checkpoint has none of, so a forced-local wan extend was saved as
    // having used LTX-2's 17 (#783). Server admission materializes the same
    // field from the same helper, so a remote run records the same value.
    mold_core::validation::materialize_extend_overlap_frames(&mut req, family.as_deref());
    // The forced-local path builds its own metadata and seeds its own row, so
    // it needs the same canonical filing server admission materializes — a
    // local run must record what a remote one would.
    mold_core::validation::materialize_request_organization(&mut req)
        .map_err(|error| anyhow::anyhow!(error))?;
    let base_seed = req.seed.unwrap_or_else(|| rand::thread_rng().gen());
    let mut reference_session = None;
    if local {
        require_local_request_model_activation(&req, &config)?;
        materialize_local_builtin_control(&mut req, &config).await?;
        materialize_local_builtin_camera_controls(&mut req, &config).await?;
    } else {
        // Several photographs and true CFG are additive fields an older server
        // would DROP rather than reject, rendering a print with no face in it
        // or with no negative branch and saying nothing. Ask before submitting.
        // Only these two shapes pay the round trip; a single `--id-image`
        // predates the capability block and every identity-capable server
        // understands it.
        require_remote_identity_capabilities(ctx.client(), &req).await?;

        if !reference_uploads.is_empty() {
            // Upload sessions bind the complete request. Freeze a random seed
            // now rather than changing it between session creation and
            // generation.
            req.seed = Some(base_seed);
            reference_session =
                bind_remote_reference_uploads(ctx.client(), &mut req, reference_uploads).await?;
        }
    }

    // Warn if user-provided dimensions don't match model recommendations.
    // Only warn locally — in remote mode the server sends the warning via SSE/header.
    if local && (width.is_some() || height.is_some()) {
        if let Some(ref family) = family {
            let composition = if family == "ltx2" {
                mold_core::validation::ltx2_spatial_composition(&req.model, req.pipeline)
            } else {
                mold_core::validation::Ltx2SpatialComposition::SinglePass
            };
            if let Some(warning) = mold_core::dimension_warning_composed(
                effective_width,
                effective_height,
                family,
                composition,
            ) {
                status!("{} {}", theme::icon_warn(), warning);
            }
        }
    }

    if let Some(desc) = &model_cfg.description {
        status!(
            "{} {} — {}",
            theme::icon_ok(),
            model.bold(),
            crate::output::colorize_description(desc)
        );
    }
    // Show truncated prompt so the user can confirm their input
    let display_prompt = if prompt.chars().count() > 60 {
        let truncated: String = prompt.chars().take(57).collect();
        format!("{truncated}...")
    } else {
        prompt.to_string()
    };
    status!("{} \"{}\"", theme::icon_info(), display_prompt.dimmed());
    if !guidance_caps.adjustable {
        status!(
            "{} Distilled recipe fixes CFG at 1.0 and does not use negative-prompt guidance; choose a Dev checkpoint with Auto or a guided pipeline to adjust it",
            theme::icon_info()
        );
    }
    if guidance_caps.supports_negative_prompt {
        if let Some(ref neg) = effective_negative_prompt {
            let display_neg = if neg.chars().count() > 50 {
                let truncated: String = neg.chars().take(47).collect();
                format!("{truncated}...")
            } else {
                neg.clone()
            };
            status!(
                "{} Negative: \"{}\"",
                theme::icon_info(),
                display_neg.dimmed()
            );
        }
    }
    if is_h3 {
        let mode = match mold_core::minimax_h3::task_for_model(&req.model) {
            Some(mold_core::minimax_h3::Task::Ref2va) => format!(
                "Ref2VA with {} ordered reference{}",
                req.references.as_ref().map_or(0, Vec::len),
                if req.references.as_ref().map_or(0, Vec::len) == 1 {
                    ""
                } else {
                    "s"
                }
            ),
            Some(mold_core::minimax_h3::Task::Fl2va) => {
                let first = req.source_image.is_some();
                let last = req
                    .keyframes
                    .as_ref()
                    .is_some_and(|items| !items.is_empty());
                match (first, last) {
                    (false, false) => "T2VA".to_string(),
                    (true, false) => "FL2VA with first frame".to_string(),
                    (false, true) => "FL2VA with last frame".to_string(),
                    (true, true) => "FL2VA with first and last frames".to_string(),
                }
            }
            None => "unknown task".to_string(),
        };
        status!("{} MiniMax H3 {mode}", theme::icon_mode());
    } else if mask_image.is_some() {
        status!(
            "{} inpainting mode (strength: {:.2})",
            theme::icon_mode(),
            strength,
        );
    } else if let Some(ref images) = edit_images {
        status!(
            "{} image edit mode ({} input image{})",
            theme::icon_mode(),
            images.len(),
            if images.len() == 1 { "" } else { "s" }
        );
    } else if source_image.is_some() {
        status!(
            "{} img2img mode (strength: {:.2})",
            theme::icon_mode(),
            strength,
        );
    }
    if let Some(ref cm) = control_model {
        status!(
            "{} ControlNet: {} (scale: {:.2})",
            theme::icon_mode(),
            cm.bold(),
            control_scale
        );
    }
    // An audio-only pipeline still reads `frames`/`fps` — that is how its
    // duration is expressed — but it renders no frames, so nothing downstream
    // should describe it as video.
    let is_video = effective_frames.is_some() && !audio_only_pipeline;
    if let Some(f) = effective_frames {
        let effective_fps = effective_fps.unwrap_or(24);
        if is_h3 {
            status!(
                "{} Synchronized AV: {} frames @ {} fps ({:.2}s), 32 kHz stereo",
                theme::icon_mode(),
                f,
                effective_fps,
                f64::from(f) / f64::from(effective_fps.max(1)),
            );
        } else if audio_only_pipeline {
            status!(
                "{} Audio mode: {:.2}s ({} frames @ {} fps)",
                theme::icon_mode(),
                f as f64 / effective_fps.max(1) as f64,
                f,
                effective_fps,
            );
        } else {
            status!(
                "{} Video mode: {} frames @ {} fps",
                theme::icon_mode(),
                f,
                effective_fps,
            );
        }
        if is_ltx2 && !audio_only_pipeline {
            let audio_mode = if enable_audio == Some(false) {
                "silent"
            } else {
                "audio+video"
            };
            status!("{} LTX-2 pipeline: {}", theme::icon_mode(), audio_mode);
        }
    }
    let displayed_guidance = guidance_caps.fixed_scale.unwrap_or(effective_guidance);
    if is_h3 {
        status!(
            "{} Generating {}x{} ({} sigma points / {} model evaluations, no CFG)",
            theme::icon_info(),
            effective_width,
            effective_height,
            effective_steps,
            effective_steps.saturating_sub(1),
        );
    } else if audio_only_pipeline {
        // No raster to report. Printing the request's width and height here
        // would describe a frame this pipeline never renders.
        status!(
            "{} Generating audio ({} steps, guidance {:.1})",
            theme::icon_info(),
            effective_steps,
            displayed_guidance,
        );
    } else {
        status!(
            "{} Generating {}x{} ({} steps, guidance {:.1})",
            theme::icon_info(),
            effective_width,
            effective_height,
            effective_steps,
            displayed_guidance,
        );
    }
    status!("{}", "─".repeat(40).dimmed());

    let response = if local {
        print_using_local_inference();
        generate_local_batch(
            &req,
            &config,
            gpus,
            t5_variant.clone(),
            qwen3_variant.clone(),
            qwen2_variant.clone(),
            qwen2_text_encoder_mode.clone(),
            eager,
            offload,
            width,
            height,
            steps,
            guidance,
            batch,
            base_seed,
            batch_prompts.as_deref(),
            &output,
            output_format,
            preview,
        )
        .await?
    } else {
        let save_ctx = BatchSaveContext {
            output: &output,
            model,
            output_format,
            preview,
            batch,
        };
        let mut collected = BatchOutputs::new(batch, base_seed);

        for i in 0..batch {
            let mut iter_req = req.clone();
            if reference_session.is_none() {
                iter_req.seed = Some(base_seed.wrapping_add(i as u64));
                iter_req.batch_size = 1;

                // Use per-batch expanded prompt if available
                if let Some(ref prompts) = batch_prompts {
                    if let Some(p) = prompts.get(i as usize) {
                        iter_req.prompt = p.clone();
                    }
                }
            }

            if batch > 1 {
                status!(
                    "{} Generating image {}/{} (seed: {})",
                    theme::icon_info(),
                    i + 1,
                    batch,
                    iter_req.seed.unwrap(),
                );
            }

            let response = generate_remote(
                ctx.client(),
                &iter_req,
                &config,
                model,
                piped,
                effective_width,
                effective_height,
                effective_steps,
                gpus.clone(),
                t5_variant.clone(),
                qwen3_variant.clone(),
                qwen2_variant.clone(),
                qwen2_text_encoder_mode.clone(),
                eager,
                offload,
                width,
                height,
                steps,
                guidance,
            )
            .await;
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    if let Some(lease) = reference_session.as_mut() {
                        let _ = lease.cancel().await;
                    }
                    return Err(error);
                }
            };
            if let Some(lease) = reference_session.as_mut() {
                lease.mark_consumed();
            }

            // Persist every successful artifact before the next batch item can
            // fail. The aggregate retains the last clip or track only for the
            // common response shape; it is not the durability boundary.
            let persist = PersistArgs {
                request: &iter_req,
                seed_used: response.seed_used,
                generation_time_ms: response.generation_time_ms,
            };
            collected.absorb(i, response, &save_ctx, Some(persist))?;
        }

        collected.finish()
    };

    // Output: audio, video, or image.
    if let Some(ref audio) = response.audio {
        // --- Audio-only output (LTX-2 text-to-audio) ---
        if batch > 1 {
            // Batch tracks were persisted per item before aggregation.
        } else if piped && output.is_none() {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(&audio.data)?;
            stdout.flush()?;
        } else {
            save_and_preview_audio(
                audio,
                &output,
                model,
                1,
                0,
                preview,
                Some(PersistArgs {
                    request: &req,
                    seed_used: response.seed_used,
                    generation_time_ms: response.generation_time_ms,
                }),
            )?;
        }
    } else if let Some(ref video) = response.video {
        // --- Video output ---
        if batch > 1 {
            // Batch clips were persisted per item before aggregation.
        } else if piped && output.is_none() {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(&video.data)?;
            stdout.flush()?;
        } else {
            save_and_preview_video(
                video,
                &output,
                model,
                1,
                0,
                preview,
                Some(PersistArgs {
                    request: &req,
                    seed_used: response.seed_used,
                    generation_time_ms: response.generation_time_ms,
                }),
            )?;
        }
    } else {
        // --- Image output ---
        // For batch > 1, images were already saved inside the batch loop above.
        if piped && output.is_none() {
            let mut stdout = std::io::stdout().lock();
            for img in &response.images {
                stdout.write_all(&img.data)?;
            }
            stdout.flush()?;
        } else if batch == 1 {
            for img in &response.images {
                save_and_preview_image(
                    img,
                    &output,
                    model,
                    batch,
                    output_format,
                    preview,
                    Some(PersistArgs {
                        request: &req,
                        seed_used: response.seed_used,
                        generation_time_ms: response.generation_time_ms,
                    }),
                )?;
            }
        }
        // batch > 1: already saved+previewed inside the batch loop
    }

    let secs = response.generation_time_ms as f64 / 1000.0;
    if is_video {
        let frame_count = response
            .video
            .as_ref()
            .map(|v| v.frames)
            .unwrap_or_default();
        status!(
            "{} Done — {} in {:.1}s ({} frames, seed: {})",
            theme::icon_done(),
            model.bold(),
            secs,
            frame_count,
            response.seed_used,
        );
    } else if batch > 1 {
        status!(
            "{} Done — {} in {:.1}s ({} images, base seed: {})",
            theme::icon_done(),
            model.bold(),
            secs,
            batch,
            base_seed,
        );
    } else {
        status!(
            "{} Done — {} in {:.1}s (seed: {})",
            theme::icon_done(),
            model.bold(),
            secs,
            response.seed_used,
        );
    }

    // Remember the model for next time (best-effort, non-fatal)
    mold_db::settings::record_last_model(&response.model);

    Ok(())
}

async fn materialize_local_builtin_control(
    request: &mut GenerateRequest,
    config: &Config,
) -> Result<()> {
    require_local_request_model_activation(request, config)?;
    let Some(control) = request.ic_lora_control.as_deref() else {
        return Ok(());
    };
    let model_config = config.resolved_model_config(&request.model);
    mold_core::validate_generate_request_with_family(request, model_config.family.as_deref())
        .map_err(anyhow::Error::msg)?;
    let profile = mold_core::ltx2_control::control_profile_for_model(&request.model, &model_config)
        .map_err(anyhow::Error::msg)?;
    let adapter = mold_core::ltx2_control::resolve_control_adapter(profile, control)
        .map_err(anyhow::Error::msg)?;
    let manifest = mold_core::manifest::find_manifest(adapter.download_model)
        .expect("control registry and hidden manifests must stay in sync");
    let file = manifest
        .files
        .iter()
        .find(|file| file.hf_filename == adapter.hf_filename)
        .expect("control registry and hidden manifests must stay in sync");
    let path = config
        .resolved_models_dir()
        .join(mold_core::manifest::storage_path(manifest, file));
    if !local_control_artifact_is_complete(adapter, &path) {
        mold_core::download::pull_and_configure(
            adapter.download_model,
            &mold_core::download::PullOptions::default(),
        )
        .await
        .map_err(|error| {
            anyhow::anyhow!(
                "failed to download IC-LoRA control '{}': {error}",
                adapter.id
            )
        })?;
    }
    if !local_control_artifact_is_complete(adapter, &path) {
        anyhow::bail!(
            "IC-LoRA control '{}' download completed without a verified {}",
            adapter.id,
            path.display()
        );
    }

    let mut ordered = vec![LoraWeight {
        path: path.to_string_lossy().into_owned(),
        scale: 1.0,

        expert: None,
    }];
    if let Some(legacy) = request.lora.take() {
        ordered.push(legacy);
    }
    ordered.extend(request.loras.take().unwrap_or_default());
    request.loras = Some(ordered);
    Ok(())
}

/// Download any built-in `camera-control:<id>` adapter the request names.
///
/// The server does this in `materialize_builtin_ltx2_camera_controls` before
/// planning. Forced-local has no counterpart, and
/// `execution_plan::materialize_request` rewrites the alias to its manifest
/// path during planning — so by the time the engine's `resolve_loras` runs,
/// the `camera-control:` prefix is gone and its download never fires. The
/// render then failed on a missing file. It only ever *looked* fine because
/// the runtime refused the real path for any plan carrying a LoRA and quietly
/// emitted placeholder frames instead.
async fn materialize_local_builtin_camera_controls(
    request: &mut GenerateRequest,
    config: &Config,
) -> Result<()> {
    require_local_request_model_activation(request, config)?;
    let aliases: Vec<String> = request
        .loras
        .iter()
        .flatten()
        .chain(request.lora.iter())
        .filter_map(|lora| {
            lora.path
                .strip_prefix("camera-control:")
                .map(str::to_string)
        })
        .collect();
    if aliases.is_empty() {
        return Ok(());
    }

    let model_config = config.resolved_model_config(&request.model);
    mold_core::ltx2_camera::camera_profile_for_model(&request.model, &model_config)
        .map_err(anyhow::Error::msg)?;

    for alias in aliases {
        let preset = mold_core::ltx2_camera::resolve_camera_control_preset(&alias)
            .map_err(anyhow::Error::msg)?;
        let manifest = mold_core::manifest::find_manifest(preset.download_model)
            .expect("camera-control registry and hidden manifests must stay in sync");
        let file = manifest
            .files
            .first()
            .expect("camera-control manifests contain one adapter file");
        let path = config
            .resolved_models_dir()
            .join(mold_core::manifest::storage_path(manifest, file));
        if local_camera_control_artifact_is_complete(preset, &path) {
            continue;
        }
        mold_core::download::pull_and_configure(
            preset.download_model,
            &mold_core::download::PullOptions::default(),
        )
        .await
        .map_err(|error| {
            anyhow::anyhow!("failed to download camera-motion preset '{alias}': {error}")
        })?;
        if !local_camera_control_artifact_is_complete(preset, &path) {
            anyhow::bail!(
                "camera-motion preset '{alias}' download completed without a verified {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn local_camera_control_artifact_is_complete(
    preset: &mold_core::ltx2_camera::Ltx2CameraControlPreset,
    path: &std::path::Path,
) -> bool {
    mold_core::download::has_sha256_marker(path)
        || path
            .metadata()
            .is_ok_and(|metadata| metadata.len() == preset.size_bytes)
}

/// Local twin of the server's `control_artifact_is_complete`: every file the
/// adapter needs, each at its own recorded size.
fn local_control_artifact_is_complete(
    adapter: &mold_core::ltx2_control::Ltx2ControlAdapter,
    path: &std::path::Path,
) -> bool {
    let Some(dir) = path.parent() else {
        return false;
    };
    adapter.files().all(|file| {
        let candidate = dir.join(file.hf_filename);
        mold_core::download::has_sha256_marker(&candidate)
            || candidate
                .metadata()
                .is_ok_and(|metadata| metadata.len() == file.size_bytes)
    })
}

async fn bind_remote_reference_uploads(
    client: &MoldClient,
    request: &mut GenerateRequest,
    uploads: Vec<ReferenceUpload>,
) -> Result<Option<ReferenceUploadLease>> {
    let descriptors = request
        .references
        .clone()
        .ok_or_else(|| anyhow::anyhow!("reference uploads require ordered H3 descriptors"))?;
    if descriptors.len() != uploads.len() || descriptors.is_empty() {
        anyhow::bail!(
            "reference descriptor/file count mismatch: {} descriptors for {} files",
            descriptors.len(),
            uploads.len()
        );
    }
    mold_core::minimax_h3::validate_reference_descriptors(&descriptors)
        .map_err(anyhow::Error::new)?;
    let sources = uploads
        .into_iter()
        .map(|upload| ReferenceUploadSource::open_file(upload.file))
        .collect::<Result<Vec<_>>>()?;
    let lease = client.bind_reference_uploads_v2(request, sources).await?;
    *request = lease.request().clone();
    Ok(Some(lease))
}

fn has_remote_reference_handles(request: &GenerateRequest) -> bool {
    request.references.as_ref().is_some_and(|references| {
        references.iter().any(|reference| {
            matches!(
                reference.media(),
                GenerationReferenceAuthority::Upload { .. }
            )
        })
    })
}

/// Gate the SERVER-side auto-pull the remote path triggers when a generate
/// comes back "model missing".
///
/// This is deliberately the ACQUISITION authority, not the activation one:
/// the machine being asked to download and then run the model is the remote
/// server, which enforces its own activation on the resubmitted request. A
/// laptop CLI compiled without the `h3` engine talking to an sm89 host must
/// still be able to trigger that pull — asking the local binary whether IT
/// could run the model is the same category error as reading a Discover
/// badge off the wrong machine (#1276). Unreviewed H3 identities stay refused
/// here exactly as before, because acquisition fails closed for them.
fn require_remote_auto_pull_acquisition(request: &GenerateRequest, config: &Config) -> Result<()> {
    let family = resolve_family(&request.model, config);
    mold_core::require_model_acquisition(&request.model, family.as_deref())
        .map_err(anyhow::Error::new)
}

/// Resolve the container an unset `--format` defaults to.
///
/// Mirrors the server's family policy (`GenerateRequest::normalise_output_format`
/// via `mold_core::family_output_defaults_to_mp4`): every MP4-default video
/// family — wan and ltx-video included, not just LTX-2 — defaults a
/// multi-frame render to MP4 when this build can encode it (#806). `mp4_built`
/// is `cfg!(feature = "mp4")` at the call site: LTX-2 and MiniMax H3 carry
/// their own baseline MP4 media pipelines, but wan and ltx-video encode MP4
/// through the feature-gated `video_enc::encode_mp4`, so a minimal build
/// keeps the truthful APNG fallback instead of erroring after the render.
/// A single-frame wan render stays a PNG still (#798), and `--format` always
/// wins — `format` here is PNG only when the user didn't pick one.
fn default_output_format(
    family: Option<&str>,
    format: OutputFormat,
    effective_frames: Option<u32>,
    audio_only_pipeline: bool,
    is_h3: bool,
    mp4_built: bool,
) -> OutputFormat {
    if is_h3 {
        return OutputFormat::Mp4;
    }
    if audio_only_pipeline && format == OutputFormat::Png {
        return OutputFormat::Wav;
    }
    if format != OutputFormat::Png || effective_frames.is_none() {
        return format;
    }
    match family {
        Some("ltx2") => OutputFormat::Mp4,
        // A single-frame Wan render is a still (#798): keep the raster
        // default instead of wrapping one frame in a video container.
        Some("wan") if effective_frames == Some(1) => OutputFormat::Png,
        Some(family) if mp4_built && mold_core::family_output_defaults_to_mp4(family) => {
            OutputFormat::Mp4
        }
        _ => OutputFormat::Apng,
    }
}

/// The video container an `--output` extension names.
///
/// `png` resolves to APNG deliberately: an animated PNG *is* a PNG, and
/// `OutputFormat::Apng::extension()` is `png`, so a `.png` name on a video
/// render asks for the animated raster container rather than a still.
fn video_container_named_by_extension(extension: &str) -> Option<OutputFormat> {
    match extension {
        "gif" => Some(OutputFormat::Gif),
        "apng" | "png" => Some(OutputFormat::Apng),
        "webp" => Some(OutputFormat::Webp),
        "mp4" => Some(OutputFormat::Mp4),
        _ => None,
    }
}

/// The cargo feature a container needs and this run's encoder does not have.
fn missing_delivery_feature(
    format: OutputFormat,
    delivery: mold_core::GenerationDeliveryCapabilities,
) -> Option<&'static str> {
    match format {
        OutputFormat::Mp4 if !delivery.mp4 => Some("mp4"),
        OutputFormat::Webp if !delivery.webp => Some("webp"),
        _ => None,
    }
}

/// Human-facing container name, for errors that have to name two at once.
fn container_label(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Png => "PNG",
        OutputFormat::Jpeg => "JPEG",
        OutputFormat::Gif => "GIF",
        OutputFormat::Apng => "APNG",
        OutputFormat::Webp => "WebP",
        OutputFormat::Mp4 => "MP4",
        OutputFormat::Wav => "WAV",
    }
}

/// Reconcile the resolved video container with the extension `--output` names.
///
/// The container default is a family (or local-recipe) decision that knows
/// nothing about the filename, and in a build without the `mp4` feature it
/// degrades to a container this binary can actually encode — the truthful
/// minimal-build fallback (#806). Nothing reconciled the two, so `--output
/// out.mp4` happily received GIF bytes: QuickTime refuses that file and VLC
/// content-sniffs it into broken timing that reads as an engine bug (#1050).
///
/// The extension is the authority a caller typed by hand, so it wins whenever
/// this run can honour it, and is refused — before a weight is read, like the
/// MiniMax H3 gate in `run.rs` — when it cannot, using the engine's own
/// wording (`wan/pipeline.rs`). Three cases deliberately change nothing:
/// stdout and an omitted `--output` claim no extension, a still render is not
/// a video (a single-frame wan render stays a PNG, #798), and an extension
/// mold has no container for at all (`.mov`, `.mkv`) carries no claim this
/// function can reason about. An explicit `--format` still outranks the
/// family default, but it may not disagree with the filename either: that is
/// the same silent mismatch chosen by a different authority.
///
/// `delivery` is the encoder that will run this render — this build's linked
/// features for a forced-local run, and everything for a remote one, where
/// the server encodes and refuses at admission what it cannot deliver.
pub(crate) fn reconcile_video_format_with_output_extension(
    resolved: OutputFormat,
    output: Option<&str>,
    format_is_explicit: bool,
    delivery: mold_core::GenerationDeliveryCapabilities,
) -> Result<OutputFormat, String> {
    if !resolved.is_video() {
        return Ok(resolved);
    }
    let Some(path) = output.filter(|path| *path != "-") else {
        return Ok(resolved);
    };
    let Some(extension) = std::path::Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
    else {
        return Ok(resolved);
    };
    let Some(named) = video_container_named_by_extension(&extension) else {
        return match extension.parse::<OutputFormat>() {
            Ok(still) => Err(format!(
                "--output '{path}' names a .{extension} file, but {} is not a supported video \
                 output format — name a .mp4, .webp, .gif, or .png file, or write the bytes to \
                 stdout with --output -",
                container_label(still)
            )),
            Err(_) => Ok(resolved),
        };
    };
    if named == resolved {
        return Ok(resolved);
    }
    if format_is_explicit {
        return Err(format!(
            "--output '{path}' names {}, but --format selected {} — rename the output or drop \
             --format so the saved bytes match the extension",
            container_label(named),
            container_label(resolved)
        ));
    }
    if let Some(feature) = missing_delivery_feature(named, delivery) {
        return Err(format!(
            "--output '{path}' names a .{extension} file, but {} output requires the '{feature}' \
             feature — rebuild with --features {feature}, or name a container this build can \
             encode (.gif or .png)",
            container_label(named)
        ));
    }
    Ok(named)
}

/// Remote generation: try SSE streaming first, fall back to blocking API.
#[allow(clippy::too_many_arguments)]
/// Print the advisories a server attached to an accepted request.
///
/// The server says what it had to adjust or drop — a lip-dub retiming, a
/// filing a host with no metadata database could not apply — on the
/// `x-mold-request-warning` header. Nothing else in the CLI reads it, so
/// without this the feature's own "never a silent drop" promise would hold
/// on the wire and break at the terminal.
///
/// Goes to the `status!` path so it lands on stderr when output is piped,
/// keeping `mold run ... | viu -` byte-clean.
pub(crate) fn report_request_warnings(warnings: &[String]) {
    for warning in warnings {
        status!("{} {warning}", theme::icon_warn());
    }
}

#[allow(clippy::too_many_arguments)]
async fn generate_remote(
    client: &MoldClient,
    req: &GenerateRequest,
    config: &Config,
    model: &str,
    piped: bool,
    effective_width: u32,
    effective_height: u32,
    effective_steps: u32,
    gpus: Option<String>,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Option<String>,
    eager: bool,
    offload: bool,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
) -> Result<GenerateResponse> {
    // One reporting point for every remote path — SSE, the blocking
    // fallback, and the pull-and-retry — so a new branch inside cannot
    // silently skip the advisory.
    let response = generate_remote_inner(
        client,
        req,
        config,
        model,
        piped,
        effective_width,
        effective_height,
        effective_steps,
        gpus,
        t5_variant,
        qwen3_variant,
        qwen2_variant,
        qwen2_text_encoder_mode,
        eager,
        offload,
        cli_width,
        cli_height,
        cli_steps,
        cli_guidance,
    )
    .await?;
    report_request_warnings(&response.request_warnings);
    Ok(response)
}

#[allow(clippy::too_many_arguments)]
async fn generate_remote_inner(
    client: &MoldClient,
    req: &GenerateRequest,
    config: &Config,
    model: &str,
    piped: bool,
    effective_width: u32,
    effective_height: u32,
    effective_steps: u32,
    gpus: Option<String>,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Option<String>,
    eager: bool,
    offload: bool,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
) -> Result<GenerateResponse> {
    // Try SSE streaming first
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let render = tokio::spawn(render_progress(rx));
    let one_use_references = has_remote_reference_handles(req);

    match client.generate_stream(req, tx).await {
        Ok(Some(response)) => {
            let _ = render.await;
            Ok(response)
        }
        Ok(None) if one_use_references => {
            let _ = render.await;
            Err(tag_remote(
                client,
                anyhow::anyhow!(
                    "server lacks secure streaming generation required for one-use MiniMax H3 references; create a fresh upload attempt after updating the server"
                ),
            ))
        }
        Ok(None) => {
            // Server doesn't support SSE — fall back to blocking API with spinner
            let _ = render.await;
            generate_remote_blocking(
                client,
                req,
                config,
                model,
                piped,
                effective_width,
                effective_height,
                effective_steps,
                gpus,
                t5_variant,
                qwen3_variant,
                qwen2_variant,
                qwen2_text_encoder_mode,
                eager,
                offload,
                cli_width,
                cli_height,
                cli_steps,
                cli_guidance,
            )
            .await
        }
        Err(e) => {
            let _ = render.await;
            if one_use_references {
                return Err(tag_remote(
                    client,
                    e.context(
                        "one-use MiniMax H3 reference handles cannot be retried, pulled, or replayed through another generation endpoint",
                    ),
                ));
            }
            match classify_generate_error(&e) {
                GenerateServerAction::PullModelAndRetry => {
                    require_remote_auto_pull_acquisition(req, config)
                        .map_err(|error| tag_remote(client, error))?;
                    print_server_pull_missing_model(model);
                    stream_server_pull(client, model, &[])
                        .await
                        .map_err(|e| tag_remote(client, e))?;

                    status!("{} Generating...", theme::icon_info());

                    let (tx2, rx2) = tokio::sync::mpsc::unbounded_channel();
                    let render2 = tokio::spawn(render_progress(rx2));
                    match client.generate_stream(req, tx2).await {
                        Ok(Some(response)) => {
                            let _ = render2.await;
                            Ok(response)
                        }
                        // Either the server has no SSE endpoint (Ok(None)) or
                        // the SSE stream failed (Err) — some proxies/servers
                        // close the stream before the final `complete` event
                        // even though the blocking `/api/generate` path still
                        // works. Fall back to the blocking endpoint before
                        // giving up.
                        _ => {
                            let _ = render2.await;
                            client
                                .generate(req.clone())
                                .await
                                .map_err(|e| tag_remote(client, e))
                        }
                    }
                }
                GenerateServerAction::FallbackLocal => {
                    if has_remote_reference_handles(req) {
                        return Err(tag_remote(
                            client,
                            anyhow::anyhow!(
                                "server connection failed after H3 references were bound; secure upload handles cannot fall back to local inference"
                            ),
                        ));
                    }
                    print_using_local_inference();
                    let mut local_request = req.clone();
                    materialize_local_builtin_control(&mut local_request, config).await?;
                    materialize_local_builtin_camera_controls(&mut local_request, config).await?;
                    generate_local(
                        &local_request,
                        config,
                        gpus,
                        t5_variant,
                        qwen3_variant,
                        qwen2_variant,
                        qwen2_text_encoder_mode,
                        eager,
                        offload,
                        cli_width,
                        cli_height,
                        cli_steps,
                        cli_guidance,
                    )
                    .await
                }
                GenerateServerAction::SurfaceError => Err(tag_remote(client, e)),
            }
        }
    }
}

/// Blocking remote generation with a simple spinner (fallback for servers without SSE).
#[allow(clippy::too_many_arguments)]
async fn generate_remote_blocking(
    client: &MoldClient,
    req: &GenerateRequest,
    config: &Config,
    model: &str,
    piped: bool,
    effective_width: u32,
    effective_height: u32,
    effective_steps: u32,
    gpus: Option<String>,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Option<String>,
    eager: bool,
    offload: bool,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
) -> Result<GenerateResponse> {
    if has_remote_reference_handles(req) {
        anyhow::bail!(
            "one-use MiniMax H3 reference handles require streaming generation and cannot use the blocking fallback"
        );
    }
    let pb = ProgressBar::new_spinner();
    if piped {
        pb.set_draw_target(indicatif::ProgressDrawTarget::stderr());
    }
    pb.set_style(
        ProgressStyle::default_spinner()
            .template(&format!("{{spinner:.{}}} {{msg}}", theme::SPINNER_STYLE))
            .unwrap(),
    );
    pb.set_message(format!(
        "Generating on server ({}x{}, {} steps)...",
        effective_width, effective_height, effective_steps
    ));
    pb.enable_steady_tick(Duration::from_millis(100));

    match client.generate(req.clone()).await {
        Ok(response) => {
            pb.finish_and_clear();
            Ok(response)
        }
        Err(e) => {
            pb.finish_and_clear();
            match classify_generate_error(&e) {
                GenerateServerAction::PullModelAndRetry => {
                    require_remote_auto_pull_acquisition(req, config)
                        .map_err(|error| tag_remote(client, error))?;
                    print_server_pull_missing_model(model);
                    stream_server_pull(client, model, &[])
                        .await
                        .map_err(|e| tag_remote(client, e))?;
                    status!("{} Generating...", theme::icon_info());
                    client
                        .generate(req.clone())
                        .await
                        .map_err(|e| tag_remote(client, e))
                }
                GenerateServerAction::FallbackLocal => {
                    if has_remote_reference_handles(req) {
                        return Err(tag_remote(
                            client,
                            anyhow::anyhow!(
                                "server connection failed after H3 references were bound; secure upload handles cannot fall back to local inference"
                            ),
                        ));
                    }
                    print_using_local_inference();
                    let mut local_request = req.clone();
                    materialize_local_builtin_control(&mut local_request, config).await?;
                    materialize_local_builtin_camera_controls(&mut local_request, config).await?;
                    generate_local(
                        &local_request,
                        config,
                        gpus,
                        t5_variant,
                        qwen3_variant,
                        qwen2_variant,
                        qwen2_text_encoder_mode,
                        eager,
                        offload,
                        cli_width,
                        cli_height,
                        cli_steps,
                        cli_guidance,
                    )
                    .await
                }
                GenerateServerAction::SurfaceError => Err(tag_remote(client, e)),
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
async fn prepare_local_request(
    req: &GenerateRequest,
    config: &Config,
    gpus: Option<String>,
    t5_variant_override: Option<String>,
    qwen3_variant_override: Option<String>,
    qwen2_variant_override: Option<String>,
    qwen2_text_encoder_mode_override: Option<String>,
    eager: bool,
    offload: bool,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
) -> Result<(
    GenerateRequest,
    mold_core::ModelPaths,
    Config,
    super::local_engine::EngineOverrides,
)> {
    use super::local_engine::{resolve_or_pull_model, EngineOverrides};

    let model_name = req.model.clone();
    let mut req = req.clone();

    require_local_request_model_activation(&req, config)?;

    let (paths, effective_config, pulled) = resolve_or_pull_model(&model_name, config).await?;
    if pulled {
        let model_cfg = effective_config.resolved_model_config(&model_name);
        let family = resolve_family(&model_name, &effective_config);
        refit_request_after_pull(
            &mut req,
            &effective_config,
            &model_cfg,
            family.as_deref(),
            cli_width,
            cli_height,
            cli_steps,
            cli_guidance,
        )?;
        status!(
            "{} Updated defaults: {}x{} ({} steps, guidance {:.1})",
            theme::icon_info(),
            req.width,
            req.height,
            req.steps,
            req.guidance,
        );
    }

    validate_local_request(&req, &effective_config)?;

    let overrides = EngineOverrides {
        gpus,
        t5_variant: t5_variant_override,
        qwen3_variant: qwen3_variant_override,
        qwen2_variant: qwen2_variant_override,
        qwen2_text_encoder_mode: qwen2_text_encoder_mode_override,
        eager,
        offload,
    };
    Ok((req, paths, effective_config, overrides))
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
async fn generate_local(
    req: &GenerateRequest,
    config: &Config,
    gpus: Option<String>,
    t5_variant_override: Option<String>,
    qwen3_variant_override: Option<String>,
    qwen2_variant_override: Option<String>,
    qwen2_text_encoder_mode_override: Option<String>,
    eager: bool,
    offload: bool,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
) -> Result<GenerateResponse> {
    let base_seed = req.seed.unwrap_or_else(|| rand::thread_rng().gen());
    let output_format = req.output_format.unwrap_or(OutputFormat::Png);
    generate_local_batch(
        req,
        config,
        gpus,
        t5_variant_override,
        qwen3_variant_override,
        qwen2_variant_override,
        qwen2_text_encoder_mode_override,
        eager,
        offload,
        cli_width,
        cli_height,
        cli_steps,
        cli_guidance,
        1,
        base_seed,
        None,
        &None,
        output_format,
        false,
    )
    .await
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
struct LocalOwnerPool {
    command_txs: std::collections::BTreeMap<
        usize,
        std::sync::mpsc::SyncSender<Option<(u32, GenerateRequest)>>,
    >,
    workers: tokio::task::JoinSet<()>,
    progress_tasks: Vec<tokio::task::JoinHandle<()>>,
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl LocalOwnerPool {
    async fn shutdown_and_join(&mut self) -> Option<anyhow::Error> {
        // Closing every sender is the shutdown signal. A blocking owner
        // finishes its current request, observes Disconnected, clears the
        // progress callback, and drops the engine on that same thread.
        self.command_txs.clear();
        let mut first_error = None;
        while let Some(result) = self.workers.join_next().await {
            if let Err(error) = result {
                first_error.get_or_insert_with(|| error.into());
            }
        }
        for render in self.progress_tasks.drain(..) {
            if let Err(error) = render.await {
                first_error.get_or_insert_with(|| error.into());
            }
        }
        first_error
    }
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl Drop for LocalOwnerPool {
    fn drop(&mut self) {
        // This is the panic/cancellation safety net. Normal and recoverable
        // error paths call `shutdown_and_join`; Drop still closes the owner
        // channels before aborting async wrappers. Tokio cannot synchronously
        // join from Drop, but spawn_blocking owners retain no sender and
        // therefore exit after their current request.
        self.command_txs.clear();
        self.workers.abort_all();
        for task in &self.progress_tasks {
            task.abort();
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
fn local_batch_requests(
    base: &GenerateRequest,
    batch: u32,
    base_seed: u64,
    prompts: Option<&[String]>,
) -> Vec<GenerateRequest> {
    (0..batch)
        .map(|index| {
            let mut request = base.clone();
            request.seed = Some(base_seed.wrapping_add(index as u64));
            request.batch_size = 1;
            if let Some(prompt) = prompts.and_then(|values| values.get(index as usize)) {
                request.prompt = prompt.clone();
            }
            request
        })
        .collect()
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
#[allow(clippy::too_many_arguments)]
fn finalize_local_batch_outputs(
    req: &GenerateRequest,
    batch_requests: &[GenerateRequest],
    mut completed: Vec<(u32, GenerateResponse)>,
    first_error: Option<anyhow::Error>,
    output: &Option<String>,
    output_format: OutputFormat,
    preview: bool,
    persist_metadata: bool,
    batch: u32,
    base_seed: u64,
) -> Result<GenerateResponse> {
    completed.sort_by_key(|(index, _)| *index);
    let successful_items = completed
        .iter()
        .map(|(index, _)| (index + 1).to_string())
        .collect::<Vec<_>>();

    let save_ctx = BatchSaveContext {
        output,
        model: &req.model,
        output_format,
        preview,
        batch,
    };
    let mut collected = BatchOutputs::new(batch, base_seed);

    for (i, response) in completed {
        let item_request = batch_requests.get(i as usize).ok_or_else(|| {
            anyhow::anyhow!("completed local batch item {} is out of range", i + 1)
        })?;
        let persist = persist_metadata.then_some(PersistArgs {
            request: item_request,
            seed_used: response.seed_used,
            generation_time_ms: response.generation_time_ms,
        });
        collected.absorb(i, response, &save_ctx, persist)?;
    }

    if let Some(error) = first_error {
        return Err(error.context(format!(
            "local batch preserved {} successful output item(s) ({}) with exact per-item prompt/seed provenance",
            successful_items.len(),
            if successful_items.is_empty() {
                "none".to_string()
            } else {
                successful_items.join(", ")
            }
        )));
    }

    Ok(collected.finish())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
async fn generate_local_batch(
    req: &GenerateRequest,
    config: &Config,
    gpus: Option<String>,
    t5_variant_override: Option<String>,
    qwen3_variant_override: Option<String>,
    qwen2_variant_override: Option<String>,
    qwen2_text_encoder_mode_override: Option<String>,
    eager: bool,
    offload: bool,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
    batch: u32,
    base_seed: u64,
    batch_prompts: Option<&[String]>,
    output: &Option<String>,
    output_format: OutputFormat,
    preview: bool,
) -> Result<GenerateResponse> {
    use super::local_engine::{
        apply_local_engine_env_overrides, build_local_engine_from_plan, plan_local_batch,
        LocalBatchAdmission, LocalLease,
    };

    let (base_req, _paths, effective_config, overrides) = prepare_local_request(
        req,
        config,
        gpus,
        t5_variant_override,
        qwen3_variant_override,
        qwen2_variant_override,
        qwen2_text_encoder_mode_override,
        eager,
        offload,
        cli_width,
        cli_height,
        cli_steps,
        cli_guidance,
    )
    .await?;
    apply_local_engine_env_overrides(
        overrides.t5_variant.as_deref(),
        overrides.qwen3_variant.as_deref(),
        overrides.qwen2_variant.as_deref(),
        overrides.qwen2_text_encoder_mode.as_deref(),
    );
    let local_plan = plan_local_batch(&base_req, &effective_config, &overrides).await?;
    let mut admission = LocalBatchAdmission::new(
        &local_plan.candidates,
        batch as usize,
        local_plan.host_headroom_bytes,
    )?;
    let batch_requests = local_batch_requests(&base_req, batch, base_seed, batch_prompts);
    let mut planned_items = batch_requests
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, request)| Some((index as u32, request)))
        .collect::<Vec<_>>();

    enum LocalOwnerEvent {
        Ready(usize),
        Completed(usize, u32, Box<GenerateResponse>),
        Failed(usize, u32, anyhow::Error),
        Crashed(usize, anyhow::Error),
    }
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<LocalOwnerEvent>();
    let mut command_txs = std::collections::BTreeMap::new();
    let mut workers = tokio::task::JoinSet::new();
    let mut progress_tasks = Vec::new();
    for (ordinal, execution_plan) in &local_plan.execution_plans {
        let ordinal = *ordinal;
        let config = effective_config.clone();
        let execution_plan = execution_plan.clone();
        let prepared_execution_inputs = local_plan.prepared_execution_inputs.clone();
        let (command_tx, command_rx) =
            std::sync::mpsc::sync_channel::<Option<(u32, GenerateRequest)>>(1);
        command_txs.insert(ordinal, command_tx);
        let event_tx = event_tx.clone();
        let crash_tx = event_tx.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<mold_core::SseProgressEvent>();
        let render = tokio::spawn(render_progress(rx));
        workers.spawn(async move {
            let joined = tokio::task::spawn_blocking(move || {
                // Engine construction, loading, generation, and drop all remain
                // on this device's one local owner thread.
                mold_inference::device::init_thread_gpu_ordinal(ordinal);
                let mut engine = None;
                let _ = event_tx.send(LocalOwnerEvent::Ready(ordinal));
                while let Ok(Some((index, mut request))) = command_rx.recv() {
                    mold_server::execution_plan::materialize_request(&execution_plan, &mut request);
                    let result = (|| -> Result<GenerateResponse> {
                        if engine.is_none() {
                            let mut created = build_local_engine_from_plan(
                                &request,
                                &config,
                                &execution_plan,
                                &prepared_execution_inputs,
                            )?;
                            created.set_on_progress(Box::new({
                                let tx = tx.clone();
                                move |event| {
                                    let _ = tx.send(event.into());
                                }
                            }));
                            created.load_for_request(&request)?;
                            engine = Some(created);
                        }
                        engine
                            .as_mut()
                            .expect("local engine initialized before generation")
                            .generate(&request)
                    })();
                    match result {
                        Ok(response) => {
                            let _ = event_tx.send(LocalOwnerEvent::Completed(
                                ordinal,
                                index,
                                Box::new(response),
                            ));
                            let _ = event_tx.send(LocalOwnerEvent::Ready(ordinal));
                        }
                        Err(error) => {
                            let _ = event_tx.send(LocalOwnerEvent::Failed(ordinal, index, error));
                            break;
                        }
                    }
                }
                if let Some(engine) = engine.as_mut() {
                    engine.clear_on_progress();
                }
            })
            .await;
            if let Err(error) = joined {
                let _ = crash_tx.send(LocalOwnerEvent::Crashed(ordinal, error.into()));
            }
        });
        progress_tasks.push(render);
    }
    drop(event_tx);
    let mut owners = LocalOwnerPool {
        command_txs,
        workers,
        progress_tasks,
    };

    let mut completed = Vec::with_capacity(batch as usize);
    let mut first_error = None;
    let mut terminal_items = 0_usize;
    let mut initial_ready = std::collections::BTreeSet::new();
    'events: while terminal_items < batch as usize {
        let Some(event) = event_rx.recv().await else {
            if first_error.is_none() {
                first_error = Some(anyhow::anyhow!(
                    "all local GPU owners exited with batch work unfinished"
                ));
            }
            break;
        };
        match event {
            LocalOwnerEvent::Ready(ordinal) => {
                if let Err(error) = admission.owner_ready(ordinal) {
                    first_error = Some(error.context(format!(
                        "local owner {ordinal} published an invalid ready transition"
                    )));
                    break 'events;
                }
                initial_ready.insert(ordinal);
            }
            LocalOwnerEvent::Completed(ordinal, index, response) => {
                if let Err(error) = admission.owner_completed(ordinal) {
                    first_error = Some(error.context(format!(
                        "local owner {ordinal} completed item {} outside its lease",
                        index + 1
                    )));
                    break 'events;
                }
                completed.push((index, *response));
                terminal_items += 1;
            }
            LocalOwnerEvent::Failed(ordinal, index, error) => {
                let _ = admission.owner_failed(ordinal);
                terminal_items += 1;
                if first_error.is_none() {
                    first_error = Some(anyhow::anyhow!(
                        "local item {} failed on GPU {}: {error:#}",
                        index + 1,
                        ordinal
                    ));
                }
            }
            LocalOwnerEvent::Crashed(ordinal, error) => {
                initial_ready.insert(ordinal);
                if admission.owner_failed(ordinal).is_some() {
                    terminal_items += 1;
                }
                if first_error.is_none() {
                    first_error = Some(anyhow::anyhow!(
                        "local GPU owner {ordinal} crashed: {error:#}"
                    ));
                }
            }
        }

        // Wait for every owner to publish its initial Ready before the first
        // admission so batch=1 is selected from the complete device set.
        if initial_ready.len() < owners.command_txs.len() {
            continue;
        }
        loop {
            let leases = match admission.lease_ready() {
                Ok(leases) => leases,
                Err(error) => {
                    first_error = Some(error.context("local batch admission invariant failed"));
                    break 'events;
                }
            };
            if leases.is_empty() {
                break;
            }
            let mut retry_admission = false;
            for lease in leases {
                let Some((index, request)) = planned_items
                    .get_mut(lease.item_index)
                    .and_then(Option::take)
                else {
                    admission.lease_transport_failed(lease);
                    retry_admission = true;
                    if first_error.is_none() {
                        first_error = Some(anyhow::anyhow!(
                            "local scheduler leased an unavailable batch item"
                        ));
                    }
                    continue;
                };
                if batch > 1 {
                    status!(
                        "{} Starting local item {}/{} on GPU {} (seed: {})",
                        theme::icon_info(),
                        index + 1,
                        batch,
                        lease.ordinal,
                        request.seed.unwrap(),
                    );
                }
                let returned = match owners.command_txs.get(&lease.ordinal) {
                    Some(tx) => match tx.send(Some((index, request))) {
                        Ok(()) => None,
                        Err(error) => error.0,
                    },
                    None => Some((index, request)),
                };
                if let Some(returned) = returned {
                    planned_items[lease.item_index] = Some(returned);
                    admission.lease_transport_failed(LocalLease {
                        ordinal: lease.ordinal,
                        item_index: lease.item_index,
                    });
                    retry_admission = true;
                }
            }
            if !retry_admission {
                break;
            }
        }
        if admission.has_pending() && !admission.has_active() && !admission.has_owners() {
            if first_error.is_none() {
                first_error = Some(anyhow::anyhow!(
                    "no healthy local GPU owner remains for queued batch work"
                ));
            }
            break;
        }
    }
    if let Some(error) = owners.shutdown_and_join().await {
        first_error.get_or_insert(error);
    }
    finalize_local_batch_outputs(
        req,
        &batch_requests,
        completed,
        first_error,
        output,
        output_format,
        preview,
        true,
        batch,
        base_seed,
    )
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
#[allow(clippy::too_many_arguments)]
async fn generate_local(
    _req: &GenerateRequest,
    _config: &Config,
    _gpus: Option<String>,
    _t5_variant: Option<String>,
    _qwen3_variant: Option<String>,
    _qwen2_variant: Option<String>,
    _qwen2_text_encoder_mode: Option<String>,
    _eager: bool,
    _offload: bool,
    _cli_width: Option<u32>,
    _cli_height: Option<u32>,
    _cli_steps: Option<u32>,
    _cli_guidance: Option<f64>,
) -> Result<GenerateResponse> {
    anyhow::bail!(
        "No mold server running and this binary was built without GPU support.\n\
         Either start a server with `mold serve` or rebuild with --features cuda"
    )
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
#[allow(clippy::too_many_arguments)]
async fn generate_local_batch(
    _req: &GenerateRequest,
    _config: &Config,
    _gpus: Option<String>,
    _t5_variant: Option<String>,
    _qwen3_variant: Option<String>,
    _qwen2_variant: Option<String>,
    _qwen2_text_encoder_mode: Option<String>,
    _eager: bool,
    _offload: bool,
    _cli_width: Option<u32>,
    _cli_height: Option<u32>,
    _cli_steps: Option<u32>,
    _cli_guidance: Option<f64>,
    _batch: u32,
    _base_seed: u64,
    _batch_prompts: Option<&[String]>,
    _output: &Option<String>,
    _output_format: OutputFormat,
    _preview: bool,
) -> Result<GenerateResponse> {
    anyhow::bail!(
        "No mold server running and this binary was built without GPU support.\n\
         Either start a server with `mold serve` or rebuild with --features cuda"
    )
}

/// Optional metadata used by [`save_and_preview_image`] to persist a row
/// in the SQLite gallery DB after a successful save. Skipped silently when
/// `None` (e.g. tests, stdout output, when the DB is disabled).
#[derive(Clone, Copy)]
struct PersistArgs<'a> {
    request: &'a GenerateRequest,
    seed_used: u64,
    generation_time_ms: u64,
}

/// How a batch names, previews, and formats the artifacts it persists.
struct BatchSaveContext<'a> {
    output: &'a Option<String>,
    model: &'a str,
    output_format: OutputFormat,
    preview: bool,
    batch: u32,
}

/// Outputs collected across one batch, in item order.
///
/// Both batch paths — remote HTTP and local multi-GPU — persist each item as
/// it lands rather than at the end, because the aggregate keeps only the last
/// video or audio artifact and is therefore never the durability boundary. An
/// audio-only pipeline makes that concrete: its WAV is the item's *only*
/// output, so leaving persistence to the aggregate threw away every earlier
/// completed render the moment a later sibling failed.
#[derive(Default)]
struct BatchOutputs {
    images: Vec<ImageData>,
    video: Option<mold_core::VideoData>,
    audio: Option<mold_core::AudioData>,
    total_time_ms: u64,
    last_seed_used: u64,
    last_model: String,
}

impl BatchOutputs {
    fn new(batch: u32, base_seed: u64) -> Self {
        Self {
            images: Vec::with_capacity(batch as usize),
            last_seed_used: base_seed,
            ..Default::default()
        }
    }

    /// Take one completed item's outputs, writing them to disk first when the
    /// batch has siblings that could still fail.
    fn absorb(
        &mut self,
        index: u32,
        mut response: GenerateResponse,
        ctx: &BatchSaveContext<'_>,
        persist: Option<PersistArgs<'_>>,
    ) -> Result<()> {
        self.total_time_ms += response.generation_time_ms;
        self.last_seed_used = response.seed_used;
        self.last_model = response.model.clone();

        // Audio is probed first for the same reason every other surface probes
        // it first: an audio print has no frames and no raster, so a
        // video-shaped or image-shaped probe would miss it entirely.
        if let Some(audio) = response.audio.as_ref() {
            if ctx.batch > 1 {
                save_and_preview_audio(
                    audio,
                    ctx.output,
                    ctx.model,
                    ctx.batch,
                    index,
                    ctx.preview,
                    persist,
                )?;
            }
            self.audio = response.audio.take();
        }

        if let Some(video) = response.video.as_ref() {
            if ctx.batch > 1 {
                save_and_preview_video(
                    video,
                    ctx.output,
                    ctx.model,
                    ctx.batch,
                    index,
                    ctx.preview,
                    persist,
                )?;
            }
            self.video = response.video.take();
        }

        for mut img in response.images {
            img.index = index;
            if ctx.batch > 1 {
                save_and_preview_image(
                    &img,
                    ctx.output,
                    ctx.model,
                    ctx.batch,
                    ctx.output_format,
                    ctx.preview,
                    persist,
                )?;
            }
            self.images.push(img);
        }

        Ok(())
    }

    /// The aggregate handed to the single-output code path. Video and audio
    /// carry the last item's artifact; every image is retained.
    fn finish(self) -> GenerateResponse {
        GenerateResponse {
            request_warnings: Vec::new(),
            audio: self.audio,
            images: self.images,
            video: self.video,
            generation_time_ms: self.total_time_ms,
            model: self.last_model,
            seed_used: self.last_seed_used,
            gpu: None,
        }
    }
}

/// Resolve the on-disk name for one media artifact in a batch.
///
/// `--output` names a single file, so anything past the first item has to be
/// suffixed or the run ends with one file holding the last render. Shared by
/// the video and audio savers, which name their outputs identically. An
/// explicit `--output` is used verbatim; only the default filename folds the
/// request title in as a `~<slug>` (see [`title_slug_for`]).
fn batch_media_filename(
    output: &Option<String>,
    model: &str,
    ext: &str,
    batch: u32,
    index: u32,
    slug: Option<&str>,
) -> String {
    match output {
        Some(path) if batch == 1 => path.clone(),
        Some(path) => {
            let path = std::path::Path::new(path);
            let stem = path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("output");
            let leaf = format!("{stem}-{index}.{ext}");
            path.parent()
                .filter(|directory| !directory.as_os_str().is_empty())
                .map(|directory| directory.join(&leaf))
                .unwrap_or_else(|| std::path::PathBuf::from(&leaf))
                .to_string_lossy()
                .into_owned()
        }
        None => default_filename(
            model,
            mold_core::time::now_epoch_ms_u64(),
            ext,
            batch,
            index,
            slug,
        ),
    }
}

/// The filename slug for a saved artifact: the request's `--title`, reduced
/// by the shared `title_slug` so the CLI's default filename matches the
/// `mold-{model}-{ts}[-{idx}]~{slug}.{ext}` shape the server's gallery
/// writes. `None` when the print is untitled or the title has no ASCII
/// alphanumerics — the legacy filename is then byte-identical.
fn title_slug_for(persist: Option<&PersistArgs<'_>>) -> Option<String> {
    persist
        .and_then(|args| args.request.title.as_deref())
        .and_then(mold_core::title_slug)
}

fn save_and_preview_video(
    video: &mold_core::VideoData,
    output: &Option<String>,
    model: &str,
    batch: u32,
    index: u32,
    preview: bool,
    persist: Option<PersistArgs<'_>>,
) -> anyhow::Result<()> {
    if output.as_deref() == Some("-") {
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(&video.data)?;
        stdout.flush()?;
        return Ok(());
    }
    let slug = title_slug_for(persist.as_ref());
    let filename = batch_media_filename(
        output,
        model,
        video.format.extension(),
        batch,
        index,
        slug.as_deref(),
    );

    if std::path::Path::new(&filename).exists() {
        status!("{} Overwriting: {}", theme::icon_alert(), filename);
    }
    std::fs::write(&filename, &video.data)?;
    status!(
        "{} Saved: {} ({} frames, {}x{}, {} fps)",
        theme::icon_done(),
        filename.bold(),
        video.frames,
        video.width,
        video.height,
        video.fps,
    );
    if let Some(persist) = persist {
        crate::metadata_db::record_local_save(
            std::path::Path::new(&filename),
            persist.request,
            persist.seed_used,
            persist.generation_time_ms,
            video.format,
            Some((video.width, video.height)),
            Some(video),
        );
    }
    if preview {
        // Show first frame preview (viuer doesn't support animation).
        // Fallback to the video data itself for GIF/APNG/WebP when neither
        // precomputed preview is available.
        if !video.gif_preview.is_empty() {
            preview_image(&video.gif_preview);
        } else if !video.thumbnail.is_empty() {
            preview_image(&video.thumbnail);
        } else {
            preview_image(&video.data);
        }
    }
    Ok(())
}

/// Save an audio-only output to disk and cache its waveform thumbnail.
///
/// Audio has no raster frame, so both the TUI gallery cell and the server's
/// on-demand thumbnailer would come up empty. The waveform PNG the engine
/// already rendered is written into the shared thumbnail cache here, at the
/// only moment where it exists.
fn save_and_preview_audio(
    audio: &mold_core::AudioData,
    output: &Option<String>,
    model: &str,
    batch: u32,
    index: u32,
    preview: bool,
    persist: Option<PersistArgs<'_>>,
) -> anyhow::Result<()> {
    if output.as_deref() == Some("-") {
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(&audio.data)?;
        stdout.flush()?;
        return Ok(());
    }
    let slug = title_slug_for(persist.as_ref());
    let filename = batch_media_filename(
        output,
        model,
        audio.format.extension(),
        batch,
        index,
        slug.as_deref(),
    );

    if std::path::Path::new(&filename).exists() {
        status!("{} Overwriting: {}", theme::icon_alert(), filename);
    }
    std::fs::write(&filename, &audio.data)?;
    status!(
        "{} Saved: {} ({:.2}s, {} Hz, {} ch)",
        theme::icon_done(),
        filename.bold(),
        audio.duration_ms as f64 / 1000.0,
        audio.sample_rate,
        audio.channels,
    );
    cache_audio_waveform_thumbnail(std::path::Path::new(&filename), &audio.thumbnail);
    if let Some(persist) = persist {
        crate::metadata_db::record_local_save(
            std::path::Path::new(&filename),
            persist.request,
            persist.seed_used,
            persist.generation_time_ms,
            audio.format,
            Some((audio.thumbnail_width, audio.thumbnail_height)),
            None,
        );
    }
    if preview && !audio.thumbnail.is_empty() {
        preview_image(&audio.thumbnail);
    }
    Ok(())
}

fn cache_audio_waveform_thumbnail(saved: &std::path::Path, png_bytes: &[u8]) {
    if png_bytes.is_empty() {
        return;
    }
    let Some(leaf) = saved.file_name().and_then(|name| name.to_str()) else {
        return;
    };
    let thumb_dir = mold_core::Config::mold_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".mold"))
        .join("cache")
        .join("thumbnails");
    if std::fs::create_dir_all(&thumb_dir).is_err() {
        return;
    }
    for path in mold_core::media_paths::audio_waveform_thumbnail_paths(&thumb_dir, leaf) {
        if let Err(error) = std::fs::write(&path, png_bytes) {
            tracing::warn!(
                "failed to cache waveform thumbnail {}: {error}",
                path.display()
            );
        }
    }
}

/// Save a single image to disk and optionally preview it inline.
///
/// Resolves the output filename from the `--output` flag, batch index,
/// model name, and output format. Used both inside batch loops (for
/// immediate save+preview) and for single-image output. When `persist`
/// is `Some`, also records a metadata row in the SQLite gallery DB.
fn save_and_preview_image(
    img: &ImageData,
    output: &Option<String>,
    model: &str,
    batch: u32,
    output_format: OutputFormat,
    preview: bool,
    persist: Option<PersistArgs<'_>>,
) -> anyhow::Result<()> {
    let filename = match output {
        Some(path) if path == "-" => {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(&img.data)?;
            stdout.flush()?;
            return Ok(());
        }
        Some(path) if batch == 1 => path.clone(),
        Some(path) => {
            let p = std::path::Path::new(path);
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
            let ext = output_format.to_string();
            let leaf = format!("{stem}-{}.{ext}", img.index);
            p.parent()
                .filter(|d| !d.as_os_str().is_empty())
                .map(|d| d.join(&leaf))
                .unwrap_or_else(|| std::path::PathBuf::from(&leaf))
                .to_string_lossy()
                .into_owned()
        }
        None => {
            let ext = output_format.to_string();
            let slug = title_slug_for(persist.as_ref());
            default_filename(
                model,
                mold_core::time::now_epoch_ms_u64(),
                &ext,
                batch,
                img.index,
                slug.as_deref(),
            )
        }
    };

    if std::path::Path::new(&filename).exists() {
        status!("{} Overwriting: {}", theme::icon_alert(), filename);
    }
    std::fs::write(&filename, &img.data)?;
    status!("{} Saved: {}", theme::icon_done(), filename.bold());
    if let Some(p) = persist {
        crate::metadata_db::record_local_save(
            std::path::Path::new(&filename),
            p.request,
            p.seed_used,
            p.generation_time_ms,
            output_format,
            Some((img.width, img.height)),
            None,
        );
    }
    if preview {
        preview_image(&img.data);
    }
    Ok(())
}

/// Display an image inline in the terminal using viuer.
/// Silently skipped if the `preview` feature is not compiled or if decoding fails.
#[cfg(any(feature = "preview", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreviewBackend {
    ViuerAuto,
    GhosttyKitty,
}

#[cfg(any(feature = "preview", test))]
fn preview_backend(term_program: Option<&str>, term: Option<&str>) -> PreviewBackend {
    if matches!(term_program, Some("ghostty")) || matches!(term, Some("xterm-ghostty")) {
        PreviewBackend::GhosttyKitty
    } else {
        PreviewBackend::ViuerAuto
    }
}

#[cfg(any(feature = "preview", test))]
fn fit_preview_cells(img_width: u32, img_height: u32, term_w: u16, term_h: u16) -> (u32, u32) {
    // Kitty is pixel-accurate, but the preview still occupies terminal cells.
    // We intentionally model each character row as roughly 2x taller than a
    // column is wide so Ghostty previews preserve image aspect instead of
    // stretching to the terminal's full row count.
    let bound_width = term_w as u32;
    let bound_height = 2 * term_h as u32;

    if img_width <= bound_width && img_height <= bound_height {
        let rows = std::cmp::max(1, img_height / 2 + img_height % 2);
        let rows = if rows == term_h as u32 && rows > 1 {
            rows - 1
        } else {
            rows
        };
        return (img_width, rows);
    }

    let ratio = img_width * bound_height;
    let nratio = bound_width * img_height;
    let use_width = nratio <= ratio;
    let intermediate = if use_width {
        img_height * bound_width / img_width
    } else {
        img_width * bound_height / img_height
    };

    let (cols, mut rows) = if use_width {
        (bound_width, std::cmp::max(1, intermediate / 2))
    } else {
        (intermediate, std::cmp::max(1, bound_height / 2))
    };

    if rows == term_h as u32 && rows > 1 {
        rows -= 1;
    }

    (cols, rows)
}

#[cfg(any(feature = "preview", test))]
fn ghostty_kitty_preview_chunks(encoded: &str, cell_w: u32, cell_h: u32) -> String {
    let mut out = String::new();
    let mut chunks = encoded.as_bytes().chunks(4096).peekable();
    if let Some(first) = chunks.next() {
        let more = if chunks.peek().is_some() { 1 } else { 0 };
        out.push_str(&format!(
            "\x1b_Gf=100,a=T,t=d,c={},r={},m={};{}\x1b\\",
            cell_w,
            cell_h,
            more,
            std::str::from_utf8(first).expect("base64 payload must be utf-8"),
        ));
    }

    while let Some(chunk) = chunks.next() {
        let more = if chunks.peek().is_some() { 1 } else { 0 };
        out.push_str(&format!(
            "\x1b_Gm={};{}\x1b\\",
            more,
            std::str::from_utf8(chunk).expect("base64 payload must be utf-8"),
        ));
    }

    out
}

#[cfg(any(feature = "preview", test))]
fn build_ghostty_kitty_preview_payload(
    img: &image::DynamicImage,
    term_w: u16,
    term_h: u16,
) -> std::io::Result<(String, u32, u32)> {
    let mut encoded_png = std::io::Cursor::new(Vec::new());
    img.write_to(&mut encoded_png, image::ImageFormat::Png)
        .map_err(std::io::Error::other)?;
    let encoded = general_purpose::STANDARD.encode(encoded_png.into_inner());
    let (cell_w, cell_h) = fit_preview_cells(img.width(), img.height(), term_w, term_h);
    let payload = ghostty_kitty_preview_chunks(&encoded, cell_w, cell_h);
    Ok((payload, cell_w, cell_h))
}

#[cfg(feature = "preview")]
fn print_ghostty_kitty_preview(img: &image::DynamicImage) -> std::io::Result<()> {
    let (term_w, term_h) = viuer::terminal_size();
    let (payload, cell_w, _cell_h) = build_ghostty_kitty_preview_payload(img, term_w, term_h)?;

    let mut stdout = std::io::stdout();
    write!(stdout, "{payload}")?;
    if cell_w < term_w as u32 {
        writeln!(stdout)?;
    }
    stdout.flush()
}

#[cfg(feature = "preview")]
fn preview_config() -> viuer::Config {
    viuer::Config {
        absolute_offset: false,
        ..Default::default()
    }
}

#[cfg(feature = "preview")]
pub(crate) fn preview_image(data: &[u8]) {
    if !std::io::stdout().is_terminal() {
        return;
    }

    let term_program = std::env::var("TERM_PROGRAM").ok();
    let term = std::env::var("TERM").ok();
    let backend = preview_backend(term_program.as_deref(), term.as_deref());

    // Try to decode as an animated container first (GIF/APNG/animated WebP).
    if let Some(frames) = decode_preview_animation(data) {
        if frames.len() >= 2 {
            animate_preview(&frames, backend);
            return;
        }
    }

    let Ok(img) = image::load_from_memory(data) else {
        return;
    };
    render_preview_frame(&img, backend);
}

#[cfg(feature = "preview")]
fn render_preview_frame(img: &image::DynamicImage, backend: PreviewBackend) {
    match backend {
        PreviewBackend::GhosttyKitty => {
            let _ = print_ghostty_kitty_preview(img);
        }
        PreviewBackend::ViuerAuto => {
            let conf = preview_config();
            let _ = viuer::print(img, &conf);
        }
    }
}

/// Decoded frame for CLI preview animation.
#[cfg(feature = "preview")]
struct CliAnimatedFrame {
    image: image::DynamicImage,
    delay: Duration,
}

#[cfg(feature = "preview")]
fn decode_preview_animation(data: &[u8]) -> Option<Vec<CliAnimatedFrame>> {
    use image::AnimationDecoder;
    use std::io::Cursor;

    // Sniff the container before paying for a full decode.
    let is_gif = data.len() >= 6 && (data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a"));
    let is_png_sig =
        data.len() >= 8 && data[..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let is_webp = data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP";

    let frames: Vec<_> = if is_gif {
        image::codecs::gif::GifDecoder::new(Cursor::new(data))
            .ok()?
            .into_frames()
            .collect_frames()
            .ok()?
    } else if is_png_sig {
        let decoder = image::codecs::png::PngDecoder::new(Cursor::new(data)).ok()?;
        decoder.apng().ok()?.into_frames().collect_frames().ok()?
    } else if is_webp {
        image::codecs::webp::WebPDecoder::new(Cursor::new(data))
            .ok()?
            .into_frames()
            .collect_frames()
            .ok()?
    } else {
        return None;
    };

    let mut out = Vec::with_capacity(frames.len());
    for frame in frames {
        let (num, den) = frame.delay().numer_denom_ms();
        // `numer_denom_ms()` is the delay in ms as the ratio num/den.
        let micros = if den == 0 {
            100_000
        } else {
            (u64::from(num) * 1000) / u64::from(den.max(1))
        };
        let delay = Duration::from_micros(micros).max(Duration::from_millis(20));
        let image = image::DynamicImage::ImageRgba8(frame.into_buffer());
        out.push(CliAnimatedFrame { image, delay });
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Decide how many times to loop the animation for the CLI preview. A short
/// clip gets looped a couple of times so the motion is easy to see; longer
/// clips play once. Pure function for testability.
#[cfg(any(feature = "preview", test))]
fn preview_loop_count(total: Duration) -> u32 {
    if total < Duration::from_millis(1500) {
        3
    } else if total < Duration::from_secs(3) {
        2
    } else {
        1
    }
}

#[cfg(feature = "preview")]
fn animate_preview(frames: &[CliAnimatedFrame], backend: PreviewBackend) {
    let total: Duration = frames.iter().map(|f| f.delay).sum();
    let loops = preview_loop_count(total);

    // Save the cursor position once; restore before each frame so every
    // render lands at the same spot and overwrites the previous frame.
    // After the final frame, the backend naturally leaves the cursor
    // below the image — no further restore needed.
    let mut stdout = std::io::stdout();
    let _ = stdout.write_all(b"\x1b[s");
    let _ = stdout.flush();

    for _ in 0..loops {
        for frame in frames {
            let _ = stdout.write_all(b"\x1b[u");
            let _ = stdout.flush();
            render_preview_frame(&frame.image, backend);
            std::thread::sleep(frame.delay);
        }
    }
}

#[cfg(not(feature = "preview"))]
pub(crate) fn preview_image(_data: &[u8]) {
    use std::sync::OnceLock;
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        eprintln!(
            "warning: --preview has no effect — rebuild with `--features preview` to enable inline image display"
        );
    });
}

/// Build a default output filename, sanitizing colons from model names and
/// appending `~<slug>` when the print is titled — the same shape the
/// server's gallery writes, so a remote render and the CLI's local copy of
/// it share a recognizable name.
fn default_filename(
    model: &str,
    timestamp: u64,
    ext: &str,
    batch: u32,
    index: u32,
    slug: Option<&str>,
) -> String {
    mold_core::default_output_filename_titled(model, timestamp, ext, batch, index, slug)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::ModelConfig;

    fn wan_surface_parity_fixture() -> serde_json::Value {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/wan/surface-parity-v1.json"
        )))
        .expect("fixture parses")
    }

    /// The issue's acceptance criterion 2 (#806): in an mp4-enabled build a
    /// multi-frame wan render defaults to MP4 exactly like the server's
    /// `normalise_output_format` would pick, and a minimal build without the
    /// `mp4` feature keeps the truthful APNG fallback instead of promising a
    /// container it cannot encode.
    #[test]
    fn wan_multi_frame_defaults_to_mp4_when_the_build_can_encode_it() {
        let fixture = wan_surface_parity_fixture();
        let expected = fixture["container_default"]["unset_multi_frame"]
            .as_str()
            .unwrap();
        assert_eq!(expected, "mp4");
        assert_eq!(
            default_output_format(Some("wan"), OutputFormat::Png, Some(81), false, false, true),
            OutputFormat::Mp4
        );
        // ltx-video shares the same family policy and the same feature gate.
        assert_eq!(
            default_output_format(
                Some("ltx-video"),
                OutputFormat::Png,
                Some(97),
                false,
                false,
                true
            ),
            OutputFormat::Mp4
        );
    }

    #[test]
    fn wan_multi_frame_falls_back_to_apng_in_minimal_builds() {
        let fixture = wan_surface_parity_fixture();
        let fallback = fixture["container_default"]["cli_minimal_build_fallback"]
            .as_str()
            .unwrap();
        assert_eq!(fallback, "apng");
        assert_eq!(
            default_output_format(
                Some("wan"),
                OutputFormat::Png,
                Some(81),
                false,
                false,
                false
            ),
            OutputFormat::Apng
        );
        assert_eq!(
            default_output_format(
                Some("ltx-video"),
                OutputFormat::Png,
                Some(97),
                false,
                false,
                false
            ),
            OutputFormat::Apng
        );
        // LTX-2's MP4 media pipeline is baseline, not `mp4`-gated: its
        // default must not degrade in a minimal build.
        assert_eq!(
            default_output_format(
                Some("ltx2"),
                OutputFormat::Png,
                Some(97),
                false,
                false,
                false
            ),
            OutputFormat::Mp4
        );
    }

    #[test]
    fn local_profile_default_matches_linked_delivery_encoders() {
        let profile = local_generation_profile(&Config::default(), "ltx-2-19b-dev:fp8")
            .expect("built-in LTX-2 profile");
        let default = mold_core::generation_profile_default_output_format(&profile, None).unwrap();
        if cfg!(feature = "mp4") {
            assert_eq!(default, OutputFormat::Mp4);
        } else {
            assert_eq!(default, OutputFormat::Gif);
            assert!(profile.recipes.iter().all(|recipe| {
                !recipe
                    .capabilities
                    .output
                    .formats
                    .contains(&OutputFormat::Mp4)
            }));
        }
        if !cfg!(feature = "webp") {
            assert!(profile.recipes.iter().all(|recipe| {
                !recipe
                    .capabilities
                    .output
                    .formats
                    .contains(&OutputFormat::Webp)
            }));
        }
        assert!(local_generation_profile(
            &Config::default(),
            mold_core::minimax_h3::FL2VA_OFFICIAL
        )
        .is_none());
    }

    #[test]
    fn local_profile_resolves_config_only_model_from_catalog_contract() {
        let mut config = Config::default();
        config.models.insert(
            "my-local-model".to_string(),
            ModelConfig {
                family: Some("flux".to_string()),
                default_width: Some(768),
                default_height: Some(512),
                default_steps: Some(12),
                default_guidance: Some(2.5),
                ..ModelConfig::default()
            },
        );

        let profile = local_generation_profile(&config, "my-local-model")
            .expect("config-only model resolves through local catalog");
        let recipe = profile.default_recipe().expect("default recipe");
        assert_eq!(recipe.defaults.width, 768);
        assert_eq!(recipe.defaults.height, 512);
        assert_eq!(recipe.defaults.steps, 12);
        assert_eq!(recipe.defaults.guidance, 2.5);
        assert_eq!(recipe.capabilities.output.default_format, OutputFormat::Png);
    }

    /// #798 regression guard: a single-frame wan render is a still and stays
    /// PNG regardless of the build's MP4 capability.
    #[test]
    fn wan_single_frame_stays_png() {
        for mp4_built in [false, true] {
            assert_eq!(
                default_output_format(
                    Some("wan"),
                    OutputFormat::Png,
                    Some(1),
                    false,
                    false,
                    mp4_built
                ),
                OutputFormat::Png
            );
        }
    }

    #[test]
    fn explicit_format_and_image_families_keep_their_container() {
        // An explicit user choice always wins, even for a wan video.
        assert_eq!(
            default_output_format(Some("wan"), OutputFormat::Gif, Some(81), false, false, true),
            OutputFormat::Gif
        );
        // Image families without frames keep the raster default.
        assert_eq!(
            default_output_format(Some("flux"), OutputFormat::Png, None, false, false, true),
            OutputFormat::Png
        );
        // Audio-only pipelines emit WAV, and H3 is always MP4.
        assert_eq!(
            default_output_format(
                Some("ltx2"),
                OutputFormat::Png,
                Some(126),
                true,
                false,
                true
            ),
            OutputFormat::Wav
        );
        assert_eq!(
            default_output_format(
                Some("minimax-h3"),
                OutputFormat::Png,
                Some(90),
                false,
                true,
                true
            ),
            OutputFormat::Mp4
        );
    }

    fn full_build() -> mold_core::GenerationDeliveryCapabilities {
        mold_core::GenerationDeliveryCapabilities::new(true, true)
    }

    fn minimal_build() -> mold_core::GenerationDeliveryCapabilities {
        mold_core::GenerationDeliveryCapabilities::new(false, false)
    }

    /// #1050 acceptance 1: an extension this build can encode outranks the
    /// family's container default, for every video family rather than wan
    /// alone — the reconciliation sits above the family policy.
    #[test]
    fn output_extension_wins_over_the_family_container_default() {
        assert_eq!(
            reconcile_video_format_with_output_extension(
                OutputFormat::Mp4,
                Some("clip.gif"),
                false,
                full_build(),
            ),
            Ok(OutputFormat::Gif)
        );
        assert_eq!(
            reconcile_video_format_with_output_extension(
                OutputFormat::Mp4,
                Some("/tmp/clip.APNG"),
                false,
                full_build(),
            ),
            Ok(OutputFormat::Apng)
        );
        assert_eq!(
            reconcile_video_format_with_output_extension(
                OutputFormat::Gif,
                Some("clip.webp"),
                false,
                full_build(),
            ),
            Ok(OutputFormat::Webp)
        );
        // A `.png` name asks for the animated raster container, not a still:
        // `OutputFormat::Apng::extension()` is `png`.
        assert_eq!(
            reconcile_video_format_with_output_extension(
                OutputFormat::Mp4,
                Some("clip.png"),
                false,
                full_build(),
            ),
            Ok(OutputFormat::Apng)
        );
    }

    #[test]
    fn matching_extensions_pass_through_untouched() {
        for (resolved, path) in [
            (OutputFormat::Mp4, "clip.mp4"),
            (OutputFormat::Gif, "clip.gif"),
            (OutputFormat::Apng, "clip.png"),
            (OutputFormat::Apng, "clip.apng"),
            (OutputFormat::Webp, "clip.webp"),
        ] {
            assert_eq!(
                reconcile_video_format_with_output_extension(
                    resolved,
                    Some(path),
                    false,
                    full_build(),
                ),
                Ok(resolved),
                "{path} should keep {resolved}"
            );
        }
    }

    /// #1050 acceptance 2: the exact repro. The family default degraded to a
    /// container this build can encode, the user named `.mp4`, and the bytes
    /// used to be written anyway. Now it refuses with the engine's own wording
    /// (`wan/pipeline.rs`) before a single weight is read.
    #[test]
    fn mp4_output_extension_is_refused_when_the_build_cannot_encode_it() {
        let error = reconcile_video_format_with_output_extension(
            OutputFormat::Gif,
            Some("out.mp4"),
            false,
            minimal_build(),
        )
        .expect_err("a minimal build cannot honour a .mp4 name");
        assert!(
            error.contains("MP4 output requires the 'mp4' feature — rebuild with --features mp4"),
            "got: {error}"
        );
        assert!(error.contains("out.mp4"), "got: {error}");
    }

    #[test]
    fn webp_output_extension_is_refused_without_the_webp_encoder() {
        let error = reconcile_video_format_with_output_extension(
            OutputFormat::Apng,
            Some("out.webp"),
            false,
            mold_core::GenerationDeliveryCapabilities::new(true, false),
        )
        .expect_err("a build without the webp encoder cannot honour a .webp name");
        assert!(
            error
                .contains("WebP output requires the 'webp' feature — rebuild with --features webp"),
            "got: {error}"
        );
    }

    #[test]
    fn gif_output_extension_is_accepted_by_every_build() {
        for delivery in [minimal_build(), full_build()] {
            assert_eq!(
                reconcile_video_format_with_output_extension(
                    OutputFormat::Mp4,
                    Some("out.gif"),
                    false,
                    delivery,
                ),
                Ok(OutputFormat::Gif)
            );
        }
    }

    /// #1050 acceptance 3: stdout claims no extension, and an omitted
    /// `--output` names no file at all, so both keep the resolved container.
    #[test]
    fn stdout_and_omitted_output_keep_the_resolved_container() {
        for output in [None, Some("-")] {
            assert_eq!(
                reconcile_video_format_with_output_extension(
                    OutputFormat::Gif,
                    output,
                    false,
                    minimal_build(),
                ),
                Ok(OutputFormat::Gif)
            );
        }
    }

    #[test]
    fn stills_and_unnameable_extensions_are_left_alone() {
        // A single-frame render is not a video; `.mp4` there is the caller's
        // business and never re-containers a still (#798).
        assert_eq!(
            reconcile_video_format_with_output_extension(
                OutputFormat::Png,
                Some("still.mp4"),
                false,
                full_build(),
            ),
            Ok(OutputFormat::Png)
        );
        // An extension mold has no container for carries no claim to reconcile.
        for path in ["clip.mov", "clip", "clip.mkv"] {
            assert_eq!(
                reconcile_video_format_with_output_extension(
                    OutputFormat::Mp4,
                    Some(path),
                    false,
                    full_build(),
                ),
                Ok(OutputFormat::Mp4),
                "{path} should keep mp4"
            );
        }
    }

    #[test]
    fn raster_and_audio_extensions_on_a_video_render_are_refused() {
        for path in ["clip.jpg", "clip.wav"] {
            let error = reconcile_video_format_with_output_extension(
                OutputFormat::Mp4,
                Some(path),
                false,
                full_build(),
            )
            .expect_err("no video fits that container");
            assert!(
                error.contains("is not a supported video output format"),
                "got: {error}"
            );
        }
    }

    /// An explicit `--format` still outranks the family default, but it may
    /// not disagree with the filename either — that is the same silent
    /// mismatch, chosen by a different authority.
    #[test]
    fn explicit_format_conflicting_with_the_output_extension_is_refused() {
        let error = reconcile_video_format_with_output_extension(
            OutputFormat::Gif,
            Some("clip.mp4"),
            true,
            full_build(),
        )
        .expect_err("--format gif and a .mp4 name cannot both be honoured");
        assert!(error.contains("--format"), "got: {error}");
        assert!(error.contains("clip.mp4"), "got: {error}");
        // Agreement is still fine.
        assert_eq!(
            reconcile_video_format_with_output_extension(
                OutputFormat::Gif,
                Some("clip.gif"),
                true,
                full_build(),
            ),
            Ok(OutputFormat::Gif)
        );
    }

    /// The end-to-end shape of the #1050 repro through this build's own
    /// linked encoders: the local wan recipe resolves the container, and the
    /// `.mp4` the user typed is reconciled against it.
    #[test]
    fn local_delivery_capabilities_gate_the_named_output_extension() {
        let delivery = local_generation_delivery_capabilities();
        assert_eq!(delivery.mp4, cfg!(feature = "mp4"));
        assert_eq!(delivery.webp, cfg!(feature = "webp"));

        let profile = local_generation_profile(&Config::default(), "wan21-t2v-1.3b:bf16")
            .expect("built-in wan profile");
        let resolved = mold_core::generation_profile_default_output_format(&profile, None).unwrap();
        let reconciled = reconcile_video_format_with_output_extension(
            resolved,
            Some("out.mp4"),
            false,
            delivery,
        );
        #[cfg(feature = "mp4")]
        {
            assert_eq!(resolved, OutputFormat::Mp4);
            assert_eq!(reconciled, Ok(OutputFormat::Mp4));
        }
        #[cfg(not(feature = "mp4"))]
        {
            assert_ne!(resolved, OutputFormat::Mp4);
            let error = reconciled.expect_err("a minimal build must refuse a .mp4 name");
            assert!(
                error.contains(
                    "MP4 output requires the 'mp4' feature — rebuild with --features mp4"
                ),
                "got: {error}"
            );
        }
    }

    fn h3_request(model: &str) -> GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "a synchronized scene",
            "model": model,
            "width": mold_core::minimax_h3::DEFAULT_WIDTH,
            "height": mold_core::minimax_h3::DEFAULT_HEIGHT,
            "steps": mold_core::minimax_h3::DEFAULT_STEPS,
            "guidance": 0.0,
            "seed": 42,
            "batch_size": 1,
            "output_format": "mp4",
            "strength": 1.0,
            "frames": mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES,
            "fps": mold_core::minimax_h3::FIXED_FPS,
            "enable_audio": true
        }))
        .unwrap()
    }

    fn descriptor(name: &str, sha256: &str) -> GenerationReference {
        GenerationReference::Image {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: mold_core::GenerationReferenceProvenance {
                name: Some(name.to_string()),
                sha256: Some(sha256.to_string()),
            },
            mime_type: "image/png".to_string(),
            width: 32,
            height: 32,
        }
    }

    async fn generate_remote_for_test(
        client: &MoldClient,
        request: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        generate_remote(
            client,
            request,
            &Config::default(),
            &request.model,
            false,
            request.width,
            request.height,
            request.steps,
            None,
            None,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await
    }

    /// A request that needs the additive shapes, for the probe tests.
    fn multi_photo_request() -> GenerateRequest {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a portrait",
            "model": "flux-dev:q8",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance": 3.5,
            "batch_size": 1,
        }))
        .expect("the minimal generate-request wire shape");
        request.id_images = Some(vec![vec![0x89, 0x50, 0x4e, 0x47], vec![0xff, 0xd8, 0xff]]);
        request
    }

    /// A REACHABLE server that cannot answer the probe is the dangerous case,
    /// not the safe one: a build predating `/api/capabilities` answers 404 here
    /// and then accepts the generate request while dropping `id_images`. It
    /// must be read as "identity unsupported" and refused by name.
    #[tokio::test]
    async fn a_reachable_server_that_cannot_answer_the_probe_is_refused() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        for response in [
            ResponseTemplate::new(404),
            ResponseTemplate::new(500),
            // Reachable and 200, but not a capabilities document.
            ResponseTemplate::new(200).set_body_string("not json"),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api/capabilities"))
                .respond_with(response)
                .mount(&server)
                .await;

            let client = MoldClient::new(&server.uri());
            let error = require_remote_identity_capabilities(&client, &multi_photo_request())
                .await
                .expect_err("a server that cannot answer must not be sent the request")
                .to_string();
            assert!(error.contains(&server.uri()), "{error}");
            assert!(
                error.contains("more than one identity photograph"),
                "{error}"
            );
        }
    }

    /// An UNREACHABLE server is the safe case: the ordinary remote attempt is
    /// about to fail the same way and fall back to local inference, where both
    /// shapes are honoured in full. Refusing here would take that fallback away
    /// from exactly the requests that need it.
    #[tokio::test]
    async fn an_unreachable_server_leaves_the_local_fallback_alone() {
        // A port nothing is listening on: the probe fails to connect, which is
        // the one failure class `is_connection_error` recognises.
        let client = MoldClient::new("http://127.0.0.1:1");
        require_remote_identity_capabilities(&client, &multi_photo_request())
            .await
            .expect("an unreachable host must not be turned into a hard refusal");
    }

    /// A current server answers, and the request goes through untouched.
    #[tokio::test]
    async fn a_current_server_lets_the_request_through() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(mold_core::ServerCapabilities {
                    identity: mold_core::IdentityCapabilities::advertised(),
                    ..Default::default()
                }),
            )
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        require_remote_identity_capabilities(&client, &multi_photo_request())
            .await
            .expect("a current server understands the plural shape");
    }

    /// And an ordinary request never probes at all — not even against a server
    /// that would refuse it. The mock is mounted to fail loudly if it is asked.
    #[tokio::test]
    async fn an_ordinary_request_never_probes() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(ResponseTemplate::new(404))
            .expect(0)
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let mut single: GenerateRequest = multi_photo_request();
        single.id_images = None;
        single.id_image = Some(vec![0x89, 0x50, 0x4e, 0x47]);
        single.id_weight = Some(0.8);
        require_remote_identity_capabilities(&client, &single)
            .await
            .expect("a single photograph predates the capability block");
        // Dropping the server asserts the `expect(0)`.
    }

    async fn mount_reference_v2_authority(server: &wiremock::MockServer) {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, ResponseTemplate};

        let capabilities = mold_core::ServerCapabilities {
            reference_uploads: mold_core::ReferenceUploadCapabilities {
                available: true,
                protocol_version: 2,
                requires_api_key: true,
                session_path: mold_core::reference_upload::SESSION_PATH.into(),
                upload_path: mold_core::reference_upload::UPLOAD_PATH.into(),
                session_handle_header: mold_core::reference_upload::SESSION_HANDLE_HEADER.into(),
                upload_handle_header: mold_core::reference_upload::UPLOAD_HANDLE_HEADER.into(),
                max_file_bytes: 256 * 1024 * 1024,
                max_session_bytes: 1024 * 1024 * 1024,
                session_ttl_ms: 120_000,
            },
            ..Default::default()
        };
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(ResponseTemplate::new(200).set_body_json(capabilities))
            .expect(1)
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/status"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": "test",
                "models_loaded": [],
                "busy": false,
                "gpu_info": null,
                "uptime_secs": 1,
                "instance_id": "server-1"
            })))
            .expect(1)
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn h3_reference_binding_preserves_order_and_uses_one_use_handles() {
        use sha2::{Digest as _, Sha256};
        use wiremock::matchers::{body_string, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        mount_reference_v2_authority(&server).await;
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.png");
        let second_path = dir.path().join("second.png");
        std::fs::write(&first_path, b"first").unwrap();
        std::fs::write(&second_path, b"second").unwrap();
        let first_sha = format!("{:x}", Sha256::digest(b"first"));
        let second_sha = format!("{:x}", Sha256::digest(b"second"));
        let descriptors = vec![
            descriptor("first.png", &first_sha),
            descriptor("second.png", &second_sha),
        ];
        let first_metadata = descriptors[0].redacted_metadata(0).unwrap();
        let second_metadata = descriptors[1].redacted_metadata(1).unwrap();

        Mock::given(method("POST"))
            .and(path("/api/generate/reference-upload-sessions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                mold_core::ReferenceUploadSessionResponse {
                    instance_id: "server-1".to_string(),
                    expires_at_ms: u64::MAX,
                    request_scope_sha256: "a".repeat(64),
                    session_handle: "mrs_secret".to_string(),
                    uploads: vec![
                        mold_core::ReferenceUploadSlot {
                            reference: 1,
                            handle: "mru_first".to_string(),
                        },
                        mold_core::ReferenceUploadSlot {
                            reference: 2,
                            handle: "mru_second".to_string(),
                        },
                    ],
                },
            ))
            .expect(1)
            .mount(&server)
            .await;
        for (handle, body, metadata, session_complete, scope) in [
            ("mru_first", "first", first_metadata, false, "b".repeat(64)),
            (
                "mru_second",
                "second",
                second_metadata,
                true,
                "c".repeat(64),
            ),
        ] {
            let reference = metadata.index;
            Mock::given(method("PUT"))
                .and(path("/api/generate/reference-upload"))
                .and(header("x-mold-reference-upload", handle))
                .and(body_string(body))
                .respond_with(ResponseTemplate::new(200).set_body_json(
                    mold_core::ReferenceUploadCompleteResponse {
                        instance_id: "server-1".to_string(),
                        reference,
                        metadata,
                        request_scope_sha256: scope,
                        session_complete,
                    },
                ))
                .expect(1)
                .mount(&server)
                .await;
        }

        let mut request = h3_request(mold_core::minimax_h3::REF2VA_COMFY);
        request.references = Some(descriptors);
        let mut lease = bind_remote_reference_uploads(
            &MoldClient::with_api_key(&server.uri(), "sekrit".into()),
            &mut request,
            vec![
                ReferenceUpload {
                    file: mold_core::secure_file::open_regular_file_no_follow(&first_path).unwrap(),
                },
                ReferenceUpload {
                    file: mold_core::secure_file::open_regular_file_no_follow(&second_path)
                        .unwrap(),
                },
            ],
        )
        .await
        .unwrap();
        assert!(lease.is_some());
        let references = request.references.unwrap();
        assert!(matches!(
            references[0].media(),
            GenerationReferenceAuthority::Upload { handle } if handle == "mru_first"
        ));
        assert!(matches!(
            references[1].media(),
            GenerationReferenceAuthority::Upload { handle } if handle == "mru_second"
        ));
        lease.as_mut().unwrap().mark_consumed();
    }

    #[tokio::test]
    async fn h3_reference_http_451_sends_no_upload_or_generation() {
        use sha2::{Digest as _, Sha256};
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        mount_reference_v2_authority(&server).await;
        Mock::given(method("POST"))
            .and(path("/api/generate/reference-upload-sessions"))
            .and(header("x-api-key", "sekrit"))
            .respond_with(ResponseTemplate::new(451).set_body_json(serde_json::json!({
                "error": "MiniMax H3 legal activation is unavailable",
                "code": mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED
            })))
            .expect(1)
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reference.png");
        std::fs::write(&path, b"reference").unwrap();
        let digest = format!("{:x}", Sha256::digest(b"reference"));
        let mut request = h3_request(mold_core::minimax_h3::REF2VA_COMFY);
        request.references = Some(vec![descriptor("reference.png", &digest)]);
        let error = bind_remote_reference_uploads(
            &MoldClient::with_api_key(&server.uri(), "sekrit".into()),
            &mut request,
            vec![ReferenceUpload {
                file: mold_core::secure_file::open_regular_file_no_follow(&path).unwrap(),
            }],
        )
        .await
        .unwrap_err();
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
        let requests = server.received_requests().await.unwrap();
        assert_eq!(
            requests
                .iter()
                .filter(|request| request.method.as_str() == "PUT")
                .count(),
            0
        );
        assert!(!requests.iter().any(|request| {
            matches!(request.url.path(), "/api/generate" | "/api/generate/stream")
        }));
    }

    #[tokio::test]
    async fn one_use_reference_handles_never_use_blocking_fallback() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate/stream"))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;
        let mut request = h3_request(mold_core::minimax_h3::REF2VA_COMFY);
        request.references = Some(vec![GenerationReference::Image {
            media: GenerationReferenceAuthority::Upload {
                handle: "mru_one_use".into(),
            },
            provenance: mold_core::GenerationReferenceProvenance {
                name: Some("reference.png".into()),
                sha256: Some("a".repeat(64)),
            },
            mime_type: "image/png".into(),
            width: 32,
            height: 32,
        }]);
        let error = generate_remote_for_test(&MoldClient::new(&server.uri()), &request)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("secure streaming generation"));
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].url.path(), "/api/generate/stream");
    }

    #[tokio::test]
    async fn one_use_reference_handles_never_pull_and_retry() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate/stream"))
            .respond_with(ResponseTemplate::new(404).set_body_string("model is not downloaded"))
            .expect(1)
            .mount(&server)
            .await;
        let mut request = h3_request(mold_core::minimax_h3::REF2VA_COMFY);
        request.references = Some(vec![GenerationReference::Image {
            media: GenerationReferenceAuthority::Upload {
                handle: "mru_one_use".into(),
            },
            provenance: mold_core::GenerationReferenceProvenance {
                name: Some("reference.png".into()),
                sha256: Some("a".repeat(64)),
            },
            mime_type: "image/png".into(),
            width: 32,
            height: 32,
        }]);
        let error = generate_remote_for_test(&MoldClient::new(&server.uri()), &request)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("cannot be retried"));
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].url.path(), "/api/generate/stream");
    }

    #[tokio::test]
    async fn h3_remote_builder_freezes_exact_av_contract_before_http_451() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate/stream"))
            .respond_with(ResponseTemplate::new(451).set_body_json(serde_json::json!({
                "error": "MiniMax H3 legal activation is unavailable",
                "code": mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED
            })))
            .expect(1)
            .mount(&server)
            .await;

        let error = run(
            "a synchronized scene",
            mold_core::minimax_h3::FL2VA_COMFY,
            None,
            None,
            None,
            None,
            Some(7.5),
            Some(42),
            1,
            Ltx2Options {
                frames: None,
                fps: None,
                clip_frames: None,
                motion_tail: 17,
                enable_audio: None,
                audio_file: None,
                source_video: None,
                extend_video: None,
                extend_overlap_frames: None,
                keyframes: None,
                pipeline: None,
                ic_lora_control: None,
                hdr_exr_dir: None,
                hdr_exr_full_float: false,
                loras: None,
                retake_range: None,
                spatial_upscale: None,
                temporal_upscale: None,
                guidance_overrides: None,
                sample_shift: None,
                distill_strength_high: None,
                distill_strength_low: None,
                source_image_name: None,
                references: None,
                reference_uploads: Vec::new(),
            },
            Some(server.uri()),
            OutputFormat::Png,
            false,
            None,
            FilingOptions::default(),
            false,
            false,
            None,
            None,
            None,
            None,
            None,
            Some(Scheduler::UniPc),
            Some(true),
            false,
            false,
            None,
            None,
            None,
            0.75,
            None,
            None,
            None,
            1.0,
            Some("bad negative".to_string()),
            None,
            None,
            None,
            None,
            crate::commands::identity::IdentityOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["output_format"], "mp4");
        assert_eq!(body["width"], mold_core::minimax_h3::DEFAULT_WIDTH);
        assert_eq!(body["height"], mold_core::minimax_h3::DEFAULT_HEIGHT);
        // The compact model's manifest default (21), never the official BF16
        // constant (50) the client used to submit and the server refused.
        assert_eq!(body["steps"], mold_core::minimax_h3::COMFY_DEFAULT_STEPS);
        assert_eq!(body["guidance"], 0.0);
        assert_eq!(body["strength"], 1.0);
        assert_eq!(
            body["frames"],
            mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES
        );
        assert_eq!(body["fps"], mold_core::minimax_h3::FIXED_FPS);
        assert_eq!(body["enable_audio"], true);
        assert!(body["negative_prompt"].is_null());
        assert!(body["scheduler"].is_null());
        assert!(body["cfg_plus"].is_null());
    }

    #[test]
    fn h3_remote_auto_pull_respects_compiled_acquisition_policy() {
        // The remote server is the machine that will download and run it, so
        // this gate must not consult the local binary's engine (#1276) —
        // otherwise a CLI built without `h3` could not ask an sm89 host to
        // pull the very model that host runs.
        let request = h3_request(mold_core::minimax_h3::FL2VA_COMFY);
        require_remote_auto_pull_acquisition(&request, &Config::default())
            .expect("the reviewed FL2VA model is an ordinary auto-pull identity");
        require_remote_auto_pull_acquisition(
            &h3_request(mold_core::minimax_h3::FL2VA_COMFY_NVFP4),
            &Config::default(),
        )
        .expect("a pinned download-only identity is still acquirable");

        // Acquisition still fails closed for anything unreviewed, which is
        // the refusal this gate has always existed for.
        let mut unreviewed = h3_request(mold_core::minimax_h3::FL2VA_COMFY);
        unreviewed.model = "hf:MiniMaxAI/MiniMax-H3".to_string();
        let error = require_remote_auto_pull_acquisition(&unreviewed, &Config::default())
            .expect_err("an unreviewed H3 identity may not be auto-pulled");
        assert!(
            error
                .to_string()
                .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED),
            "{error}"
        );
    }

    #[tokio::test]
    async fn local_control_materialization_preserves_built_in_first_ordering() {
        let temp = tempfile::tempdir().unwrap();
        let config = Config {
            models_dir: temp.path().display().to_string(),
            ..Config::default()
        };
        let adapter = mold_core::ltx2_control::resolve_control_adapter(
            mold_core::ltx2_control::Ltx2ControlProfile::Ltx2_19bDistilled,
            "union",
        )
        .unwrap();
        let manifest = mold_core::manifest::find_manifest(adapter.download_model).unwrap();
        let adapter_path = temp.path().join(mold_core::manifest::storage_path(
            manifest,
            &manifest.files[0],
        ));
        std::fs::create_dir_all(adapter_path.parent().unwrap()).unwrap();
        std::fs::write(&adapter_path, b"installed").unwrap();
        mold_core::download::write_sha256_marker(&adapter_path, "test").unwrap();
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "test",
            "model": "ltx-2-19b-distilled:fp8",
            "width": 960,
            "height": 576,
            "steps": 8,
            "guidance": 3.0,
            "batch_size": 1,
            "output_format": "mp4",
            "source_video_path": "/guide.mp4",
            "pipeline": "ic-lora",
            "ic_lora_control": "union",
            "lora": { "path": "/loras/legacy.safetensors", "scale": 0.6 },
            "loras": [{ "path": "/loras/style.safetensors", "scale": 0.8 }]
        }))
        .unwrap();

        materialize_local_builtin_control(&mut request, &config)
            .await
            .unwrap();

        assert!(request.lora.is_none());
        let loras = request.loras.unwrap();
        assert_eq!(loras[0].path, adapter_path.to_string_lossy());
        assert_eq!(loras[0].scale, 1.0);
        assert_eq!(loras[1].path, "/loras/legacy.safetensors");
        assert_eq!(loras[2].path, "/loras/style.safetensors");
    }

    #[test]
    fn filename_sanitizes_colon() {
        let name = default_filename("flux-dev:q6", 1773609166, "png", 1, 0, None);
        assert_eq!(name, "mold-flux-dev-q6-1773609166.png");
        assert!(!name.contains(':'));
    }

    #[test]
    fn filename_no_colon_passthrough() {
        let name = default_filename("flux-schnell", 100, "png", 1, 0, None);
        assert_eq!(name, "mold-flux-schnell-100.png");
    }

    #[test]
    fn filename_batch_includes_index() {
        let name = default_filename("flux-dev:q4", 100, "jpeg", 3, 2, None);
        assert_eq!(name, "mold-flux-dev-q4-100-2.jpeg");
    }

    #[test]
    fn filename_single_batch_no_index() {
        let name = default_filename("flux-dev:q4", 100, "png", 1, 0, None);
        assert!(!name.contains("-0."));
    }

    /// `--tag` / `--collection` become the request's filing, and a titled
    /// print picks up its slug as a tag while `generate.auto_tag_title` is
    /// on — reported back so the caller can disclose it.
    #[test]
    fn filing_options_resolve_into_the_request_fields() {
        let filing = FilingOptions {
            tags: vec!["village".into()],
            collection: Some("Smurf Village".into()),
            no_auto_tag: false,
        };
        let resolved = filing.resolve(Some("Smurf Village"), true).unwrap();
        assert_eq!(
            resolved.tags.as_deref(),
            Some(["village".to_string(), "smurf-village".to_string()].as_slice())
        );
        assert_eq!(
            resolved.collection,
            Some(mold_core::CollectionRef::by_name("Smurf Village"))
        );
        assert_eq!(resolved.auto_tagged.as_deref(), Some("smurf-village"));
    }

    /// `--no-auto-tag` wins over the setting, and the setting being off is
    /// enough on its own. Neither drops a tag the user typed.
    #[test]
    fn no_auto_tag_and_the_setting_each_suppress_the_title_tag() {
        let filing = FilingOptions {
            tags: vec!["village".into()],
            collection: None,
            no_auto_tag: true,
        };
        for auto_tag_title in [true, false] {
            let resolved = filing
                .resolve(Some("Smurf Village"), auto_tag_title)
                .unwrap();
            assert_eq!(
                resolved.tags.as_deref(),
                Some(["village".to_string()].as_slice())
            );
            assert_eq!(resolved.auto_tagged, None);
        }

        let opt_in = FilingOptions {
            tags: vec!["village".into()],
            collection: None,
            no_auto_tag: false,
        };
        let resolved = opt_in.resolve(Some("Smurf Village"), false).unwrap();
        assert_eq!(
            resolved.tags.as_deref(),
            Some(["village".to_string()].as_slice())
        );
        assert_eq!(resolved.auto_tagged, None);
    }

    /// An unfiled, untitled run sends neither field, so an older host sees
    /// exactly the request body it saw before.
    #[test]
    fn an_unfiled_run_resolves_to_absent_wire_fields() {
        let resolved = FilingOptions::default().resolve(None, true).unwrap();
        assert_eq!(resolved.tags, None);
        assert_eq!(resolved.collection, None);
        assert_eq!(resolved.auto_tagged, None);
    }

    /// `--title` folds into the default filename as `~<slug>` (the same
    /// grammar the server's gallery writes); an explicit `--output` is never
    /// rewritten, and an untitled request keeps the legacy name byte-for-byte.
    #[test]
    fn titled_requests_fold_the_slug_into_the_default_filename_only() {
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"source","model":"flux-dev:q4","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        let untitled = PersistArgs {
            request: &request,
            seed_used: 1,
            generation_time_ms: 1,
        };
        assert_eq!(title_slug_for(Some(&untitled)), None);
        assert_eq!(title_slug_for(None), None);

        request.title = Some("Smurf Village at Dusk".into());
        let titled = PersistArgs {
            request: &request,
            seed_used: 1,
            generation_time_ms: 1,
        };
        let slug = title_slug_for(Some(&titled));
        assert_eq!(slug.as_deref(), Some("smurf-village-at-dusk"));

        assert_eq!(
            default_filename("flux-dev:q4", 100, "png", 1, 0, slug.as_deref()),
            "mold-flux-dev-q4-100~smurf-village-at-dusk.png"
        );
        assert_eq!(
            default_filename("flux-dev:q4", 100, "png", 3, 2, slug.as_deref()),
            "mold-flux-dev-q4-100-2~smurf-village-at-dusk.png"
        );
        assert_eq!(
            batch_media_filename(
                &Some("clip.mp4".into()),
                "ltx",
                "mp4",
                1,
                0,
                slug.as_deref()
            ),
            "clip.mp4"
        );
        assert!(
            batch_media_filename(&None, "ltx", "mp4", 1, 0, slug.as_deref())
                .ends_with("~smurf-village-at-dusk.mp4")
        );
    }

    #[test]
    fn local_batch_requests_freeze_per_item_prompt_and_seed_provenance() {
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"source","model":"flux-dev:q4","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        let prompts = vec!["first".to_string(), "second".to_string()];
        let requests = local_batch_requests(&request, 2, 41, Some(&prompts));
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].prompt, "first");
        assert_eq!(requests[0].seed, Some(41));
        assert_eq!(requests[1].prompt, "second");
        assert_eq!(requests[1].seed, Some(42));
        assert_eq!(requests[0].batch_size, 1);
        assert_eq!(requests[1].batch_size, 1);
    }

    #[test]
    fn partial_local_video_batch_persists_success_before_returning_failure() {
        let temp = tempfile::TempDir::new().unwrap();
        let output = Some(temp.path().join("clip.mp4").to_string_lossy().into_owned());
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"source","model":"ltx-video:bf16","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        let prompts = vec!["first clip".to_string(), "second clip".to_string()];
        let batch_requests = local_batch_requests(&request, 2, 91, Some(&prompts));
        let response = GenerateResponse {
            request_warnings: Vec::new(),
            audio: None,
            images: Vec::new(),
            video: Some(mold_core::VideoData {
                data: b"successful-video".to_vec(),
                format: OutputFormat::Mp4,
                width: 512,
                height: 512,
                frames: 9,
                fps: 24,
                pipeline: None,
                pipeline_provenance_sha256: None,
                source_preprocessing: None,
                thumbnail: Vec::new(),
                gif_preview: Vec::new(),
                has_audio: false,
                duration_ms: None,
                audio_sample_rate: None,
                audio_channels: None,
            }),
            generation_time_ms: 123,
            model: "ltx-video:bf16".to_string(),
            seed_used: 91,
            gpu: Some(0),
        };

        let error = finalize_local_batch_outputs(
            &request,
            &batch_requests,
            vec![(0, response)],
            Some(anyhow::anyhow!("local item 2 failed")),
            &output,
            OutputFormat::Mp4,
            false,
            false,
            2,
            91,
        )
        .unwrap_err();

        assert_eq!(
            std::fs::read(temp.path().join("clip-0.mp4")).unwrap(),
            b"successful-video"
        );
        let rendered = format!("{error:#}");
        assert!(rendered.contains("local item 2 failed"));
        assert!(rendered.contains("successful output item(s) (1)"));
        assert_eq!(batch_requests[0].prompt, "first clip");
        assert_eq!(batch_requests[0].seed, Some(91));
    }

    #[tokio::test]
    async fn local_owner_pool_shutdown_closes_and_joins_owner_threads() {
        let (command_tx, command_rx) =
            std::sync::mpsc::sync_channel::<Option<(u32, GenerateRequest)>>(1);
        let exited = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let owner_exited = exited.clone();
        let mut workers = tokio::task::JoinSet::new();
        workers.spawn(async move {
            tokio::task::spawn_blocking(move || {
                while command_rx.recv().is_ok() {}
                owner_exited.store(true, std::sync::atomic::Ordering::SeqCst);
            })
            .await
            .unwrap();
        });
        let mut owners = LocalOwnerPool {
            command_txs: std::collections::BTreeMap::from([(0, command_tx)]),
            workers,
            progress_tasks: vec![tokio::spawn(async {})],
        };
        assert!(owners.shutdown_and_join().await.is_none());
        assert!(exited.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn output_dash_is_special() {
        // "--output -" triggers stdout output in both interactive and piped modes
        let path = "-";
        assert_eq!(path, "-");
    }

    #[test]
    fn batch_stdout_rejection_is_media_neutral_for_images_and_videos() {
        for (piped, output) in [(true, None), (false, Some("-".to_string()))] {
            let error = validate_batch_stdout(2, piped, &output).unwrap_err();
            let rendered = format!("{error:#}");
            assert!(rendered.contains("more than 1 output"));
            assert!(rendered.contains("batch outputs"));
            assert!(!rendered.contains("image"));
        }
        validate_batch_stdout(2, true, &Some("clip.mp4".to_string())).unwrap();
        validate_batch_stdout(1, true, &None).unwrap();
    }

    #[test]
    fn pipe_detection_available() {
        // Verify pipe detection is available (used at runtime to route output)
        let _piped = crate::output::is_piped();
    }

    #[test]
    fn test_default_filename_empty_model() {
        let name = default_filename("", 100, "png", 1, 0, None);
        assert_eq!(name, "mold--100.png");
        // Batch variant with empty model
        let batch_name = default_filename("", 100, "png", 2, 1, None);
        assert_eq!(batch_name, "mold--100-1.png");
    }

    #[test]
    fn test_default_filename_special_chars() {
        // Colons are sanitized to dashes
        let name = default_filename("model:tag:extra", 42, "png", 1, 0, None);
        assert_eq!(name, "mold-model-tag-extra-42.png");
        assert!(!name.contains(':'));

        // Other special characters pass through
        let name2 = default_filename("my_model.v2", 42, "png", 1, 0, None);
        assert_eq!(name2, "mold-my_model.v2-42.png");
    }

    #[test]
    fn test_default_filename_jpeg_extension() {
        // "jpeg" extension passes through as-is
        let name = default_filename("flux-dev:q4", 500, "jpeg", 1, 0, None);
        assert_eq!(name, "mold-flux-dev-q4-500.jpeg");
        assert!(name.ends_with(".jpeg"));

        // "jpg" extension also works
        let name2 = default_filename("flux-dev:q4", 500, "jpg", 1, 0, None);
        assert_eq!(name2, "mold-flux-dev-q4-500.jpg");
        assert!(name2.ends_with(".jpg"));
    }

    fn png_with_dimensions(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |_, _| image::Rgb([255, 0, 0]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    /// #4 of the Turbo-manifests work: `mold run` submitted the official
    /// BF16 constant (50 steps) and generic image defaults for compact H3,
    /// which the server refuses only after admission. Every H3 identity must
    /// default to its own manifest tier's envelope: 1344x768 and 21 / 9 / 5
    /// steps.
    #[test]
    fn h3_defaults_come_from_the_manifest_tier_not_generic_image_defaults() {
        let config = Config::default();
        for (model, expected_steps) in [
            (mold_core::minimax_h3::FL2VA_COMFY, 21),
            (mold_core::minimax_h3::FL2VA_COMFY_TURBO_8STEP, 9),
            (mold_core::minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P, 5),
        ] {
            let model_cfg = config.resolved_model_config(model);
            assert_eq!(
                resolve_family(model, &config).as_deref(),
                Some("minimax-h3"),
                "{model}"
            );
            assert_eq!(
                effective_generation_steps(true, model, &model_cfg, &config, None),
                expected_steps,
                "{model}"
            );
            // An explicit --steps always wins.
            assert_eq!(
                effective_generation_steps(true, model, &model_cfg, &config, Some(2)),
                2
            );
            assert_eq!(
                effective_dimensions(
                    &config,
                    &model_cfg,
                    model,
                    Some("minimax-h3"),
                    None,
                    None,
                    None,
                    None
                )
                .unwrap(),
                (
                    mold_core::minimax_h3::DEFAULT_WIDTH,
                    mold_core::minimax_h3::DEFAULT_HEIGHT
                ),
                "{model}"
            );
        }

        // The hidden official BF16 reference keeps its 50-step manifest
        // default through the same path.
        let official = config.resolved_model_config(mold_core::minimax_h3::FL2VA_OFFICIAL);
        assert_eq!(
            effective_generation_steps(
                true,
                mold_core::minimax_h3::FL2VA_OFFICIAL,
                &official,
                &config,
                None
            ),
            mold_core::minimax_h3::DEFAULT_STEPS
        );
    }

    /// `build_model_catalog` launders a user's saved `model_pref` into
    /// `ModelConfig.default_steps`, so a value stored before the reviewed
    /// envelope was pinned can outlive it. The compact tiers run exactly
    /// their own step count, so an omitted `--steps` must resolve from the
    /// tier identity and never from that stored preference — otherwise the
    /// CLI builds a request its own profile then refuses. An explicit
    /// `--steps` stays user input: the profile answers it with the clear
    /// fixed-at message rather than a silent snap. The hidden official BF16
    /// reference keeps the stored value, because its ladder is adjustable.
    #[test]
    fn h3_compact_steps_ignore_a_stale_model_pref_default() {
        let config = Config::default();
        for (model, expected_steps) in [
            (mold_core::minimax_h3::FL2VA_COMFY, 21),
            (mold_core::minimax_h3::REF2VA_COMFY, 21),
            (mold_core::minimax_h3::FL2VA_COMFY_TURBO_8STEP, 9),
            (mold_core::minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P, 5),
        ] {
            let mut stale = config.resolved_model_config(model);
            stale.default_steps = Some(30);
            assert_eq!(
                effective_generation_steps(true, model, &stale, &config, None),
                expected_steps,
                "{model}"
            );
            assert_eq!(
                effective_generation_steps(true, model, &stale, &config, Some(30)),
                30,
                "{model}"
            );

            // End to end: the request the CLI builds from that stale config
            // is admitted by the very profile that pins the envelope. A
            // build without `mp4` withholds H3's MP4-only recipes entirely,
            // so there is nothing to validate against there.
            if let Some(profile) = local_generation_profile(&config, model)
                .filter(|profile| profile.default_recipe().is_some())
            {
                let request: mold_core::GenerateRequest =
                    serde_json::from_value(serde_json::json!({
                        "prompt": "test",
                        "model": model,
                        "width": mold_core::minimax_h3::DEFAULT_WIDTH,
                        "height": mold_core::minimax_h3::DEFAULT_HEIGHT,
                        "steps": effective_generation_steps(true, model, &stale, &config, None),
                        "guidance": 0.0,
                        "frames": mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES,
                        "fps": mold_core::minimax_h3::FIXED_FPS,
                    }))
                    .unwrap();
                mold_core::generation_profile::validate_request_against_generation_profile(
                    &profile, &request,
                )
                .unwrap_or_else(|error| panic!("{model}: {error}"));

                // ...and the stale value the old code submitted is exactly
                // what that profile refuses, with the message a user can act
                // on. This is what an explicit `--steps 30` still earns.
                let mut stale_request = request.clone();
                stale_request.steps = 30;
                assert!(
                    mold_core::generation_profile::validate_request_against_generation_profile(
                        &profile,
                        &stale_request,
                    )
                    .unwrap_err()
                    .contains("steps is fixed at"),
                    "{model}"
                );
            }
        }

        let mut official = config.resolved_model_config(mold_core::minimax_h3::FL2VA_OFFICIAL);
        official.default_steps = Some(30);
        assert_eq!(
            effective_generation_steps(
                true,
                mold_core::minimax_h3::FL2VA_OFFICIAL,
                &official,
                &config,
                None
            ),
            30
        );
    }

    /// A source image without explicit dims must not bend a compact H3
    /// request off the reviewed envelope: admission validates membership in
    /// the reviewed canvas set, and the engine fits the source into whichever
    /// one the request names. Within the set the source's aspect chooses, so
    /// a square source lands on the square canvas rather than being
    /// letterboxed into the 7:4 default. The hidden official BF16 reference
    /// keeps its flexible aspect-derived canvas.
    #[test]
    fn h3_source_image_keeps_the_compact_envelope() {
        let config = Config::default();
        let model_cfg = ModelConfig::default();

        // A square 1024x1024 source (encoded as a real PNG).
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(image::RgbImage::new(1024, 1024))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();

        for model in [
            mold_core::minimax_h3::FL2VA_COMFY,
            mold_core::minimax_h3::FL2VA_COMFY_TURBO_8STEP,
            mold_core::minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P,
        ] {
            assert_eq!(
                effective_dimensions(
                    &config,
                    &model_cfg,
                    model,
                    Some("minimax-h3"),
                    None,
                    None,
                    Some(&png),
                    None
                )
                .unwrap(),
                // A square source now lands on the square reviewed canvas
                // instead of being letterboxed into the 7:4 default.
                (768, 768),
                "{model}"
            );
        }

        // A 16:9 source keeps the historical default canvas.
        let mut landscape = Vec::new();
        image::DynamicImage::ImageRgb8(image::RgbImage::new(1920, 1080))
            .write_to(
                &mut std::io::Cursor::new(&mut landscape),
                image::ImageFormat::Png,
            )
            .unwrap();
        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                mold_core::minimax_h3::FL2VA_COMFY,
                Some("minimax-h3"),
                None,
                None,
                Some(&landscape),
                None
            )
            .unwrap(),
            (
                mold_core::minimax_h3::DEFAULT_WIDTH,
                mold_core::minimax_h3::DEFAULT_HEIGHT
            )
        );

        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                mold_core::minimax_h3::FL2VA_OFFICIAL,
                Some("minimax-h3"),
                None,
                None,
                Some(&png),
                None
            )
            .unwrap(),
            mold_core::minimax_h3::recommended_dimensions(1024, 1024)
        );

        // Explicit dims still win on every layout.
        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                mold_core::minimax_h3::FL2VA_COMFY_TURBO_8STEP,
                Some("minimax-h3"),
                Some(1344),
                Some(768),
                Some(&png),
                None
            )
            .unwrap(),
            (1344, 768)
        );
    }

    #[test]
    fn effective_dimensions_uses_model_defaults_for_txt2img() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            ..ModelConfig::default()
        };

        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                "flux-dev:q4",
                None,
                None,
                None,
                None,
                None
            )
            .unwrap(),
            (1024, 1024)
        );
    }

    #[test]
    fn effective_dimensions_fits_wide_source_to_model_bounds() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            ..ModelConfig::default()
        };
        // 1280x704 is wider than 1:1 model -> width-limited: w=1024, h=1024*(704/1280)=563.2->560
        let source = png_with_dimensions(1280, 704);

        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                "flux-dev:q4",
                None,
                None,
                None,
                Some(&source),
                None,
            )
            .unwrap(),
            (1024, 560)
        );
    }

    #[test]
    fn effective_dimensions_fits_non_aligned_source_to_model_bounds() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            ..ModelConfig::default()
        };
        // 1001x777: src_ratio=1.288 > 1.0, width-limited: w=1024, h=1024/1.288=795.0->784
        let source = png_with_dimensions(1001, 777);

        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                "flux-dev:q4",
                None,
                None,
                None,
                Some(&source),
                None,
            )
            .unwrap(),
            (1024, 784)
        );
    }

    /// The 5B's I2V source fitting must land on its 2.2 VAE's 32px grid —
    /// the flow that used to produce %16-not-%32 canvases the engine rejected
    /// only after loading — while 2.1-VAE checkpoints keep the 16px grid.
    #[test]
    fn effective_dimensions_fits_wan22_ti2v_5b_source_to_its_32px_grid() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1280),
            default_height: Some(704),
            ..ModelConfig::default()
        };
        // 1617x1000 is height-limited: h=704, w=704*1.617=1138.4, which
        // floors differently on the two grids (1120 vs 1136).
        let source = png_with_dimensions(1617, 1000);

        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                "wan22-ti2v-5b",
                Some("wan"),
                None,
                None,
                Some(&source),
                None,
            )
            .unwrap(),
            (1120, 704)
        );
        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                "wan21-t2v-1.3b",
                Some("wan"),
                None,
                None,
                Some(&source),
                None,
            )
            .unwrap(),
            (1136, 704)
        );
    }

    #[test]
    fn effective_dimensions_explicit_overrides_win_for_img2img() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            ..ModelConfig::default()
        };
        let source = png_with_dimensions(1280, 704);

        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                "flux-dev:q4",
                None,
                Some(512),
                Some(768),
                Some(&source),
                None,
            )
            .unwrap(),
            (512, 768)
        );
    }

    #[test]
    fn effective_dimensions_downscales_to_sd15_model() {
        // 1024x1024 FLUX output -> 512x512 SD1.5 model
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(512),
            default_height: Some(512),
            ..ModelConfig::default()
        };
        let source = png_with_dimensions(1024, 1024);

        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                "flux-dev:q4",
                None,
                None,
                None,
                Some(&source),
                None,
            )
            .unwrap(),
            (512, 512)
        );
    }

    #[test]
    fn effective_dimensions_wide_source_to_sd15() {
        // 1920x1080 -> 512x512 SD1.5: width-limited, h=512/1.778=287.9->288
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(512),
            default_height: Some(512),
            ..ModelConfig::default()
        };
        let source = png_with_dimensions(1920, 1080);

        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                "flux-dev:q4",
                None,
                None,
                None,
                Some(&source),
                None,
            )
            .unwrap(),
            (512, 288)
        );
    }

    #[test]
    fn effective_dimensions_upscales_small_source_to_model_native() {
        // 512x512 source -> 1024x1024 FLUX model (upscale to model native)
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            ..ModelConfig::default()
        };
        let source = png_with_dimensions(512, 512);

        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                "flux-dev:q4",
                None,
                None,
                None,
                Some(&source),
                None,
            )
            .unwrap(),
            (1024, 1024)
        );
    }

    #[test]
    fn effective_dimensions_qwen_image_edit_uses_first_edit_image() {
        let config = Config::default();
        let model_cfg = ModelConfig::default();
        let first = png_with_dimensions(1600, 900);
        let second = png_with_dimensions(512, 512);

        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                "qwen-image-edit",
                Some("qwen-image-edit"),
                None,
                None,
                None,
                Some(&[first, second]),
            )
            .unwrap(),
            (1360, 768)
        );
    }

    #[test]
    fn effective_dimensions_qwen_image_edit_requires_input_image() {
        let config = Config::default();
        let model_cfg = ModelConfig::default();
        let err = effective_dimensions(
            &config,
            &model_cfg,
            "qwen-image-edit",
            Some("qwen-image-edit"),
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("requires at least one input image"));
    }

    fn refit_req(width: u32, height: u32, source_image: Option<Vec<u8>>) -> GenerateRequest {
        let mut req: GenerateRequest = serde_json::from_str(
            r#"{
                "prompt":"p",
                "model":"flux-dev:q4",
                "width":0,
                "height":0,
                "steps":50,
                "guidance":1.0
            }"#,
        )
        .unwrap();
        req.width = width;
        req.height = height;
        req.source_image = source_image;
        req
    }

    /// After an auto-pull refreshes the model config, an img2img request
    /// whose dimensions were defaulted must be re-fit to the refreshed
    /// model canvas (this used to be two copy-pasted blocks inside
    /// prepare_local_engine).
    #[test]
    fn refit_request_after_pull_fits_source_after_autopull() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            default_steps: Some(8),
            default_guidance: Some(3.5),
            ..ModelConfig::default()
        };
        let source = png_with_dimensions(1280, 704);
        let mut req = refit_req(512, 512, Some(source));

        refit_request_after_pull(&mut req, &config, &model_cfg, None, None, None, None, None)
            .unwrap();
        // Same fit as effective_dimensions: width-limited 1024, 560.
        assert_eq!((req.width, req.height), (1024, 560));
        assert_eq!(req.steps, 8);
        assert_eq!(req.guidance, 3.5);
    }

    /// Explicit CLI flags always win over refreshed model defaults.
    #[test]
    fn refit_keeps_explicit_cli_dims_and_params() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            default_steps: Some(8),
            default_guidance: Some(3.5),
            ..ModelConfig::default()
        };
        let mut req = refit_req(640, 480, None);
        req.steps = 12;
        req.guidance = 7.0;

        refit_request_after_pull(
            &mut req,
            &config,
            &model_cfg,
            None,
            Some(640),
            Some(480),
            Some(12),
            Some(7.0),
        )
        .unwrap();
        assert_eq!((req.width, req.height), (640, 480));
        assert_eq!(req.steps, 12);
        assert_eq!(req.guidance, 7.0);
    }

    /// txt2img with defaulted dims re-derives from the refreshed model
    /// defaults (previously left at the pre-pull values).
    #[test]
    fn refit_rederives_txt2img_dims_from_refreshed_defaults() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1216),
            default_height: Some(704),
            ..ModelConfig::default()
        };
        let mut req = refit_req(1024, 1024, None);
        refit_request_after_pull(&mut req, &config, &model_cfg, None, None, None, None, None)
            .unwrap();
        assert_eq!((req.width, req.height), (1216, 704));
    }

    #[test]
    fn effective_negative_prompt_injects_space_for_qwen_image_edit_cfg() {
        assert_eq!(
            effective_negative_prompt(Some("qwen-image-edit"), 4.0, None).as_deref(),
            Some(" ")
        );
    }

    #[test]
    fn effective_negative_prompt_preserves_explicit_or_non_edit_values() {
        assert_eq!(
            effective_negative_prompt(Some("qwen-image-edit"), 4.0, Some("keep".to_string()))
                .as_deref(),
            Some("keep")
        );
        assert_eq!(effective_negative_prompt(Some("flux"), 4.0, None), None);
        assert_eq!(
            effective_negative_prompt(Some("qwen-image-edit"), 1.0, None),
            None
        );
    }

    #[test]
    fn qwen_image_edit_rejects_cli_batch_above_one() {
        let err = validate_cli_batch_for_family(Some("qwen-image-edit"), 2).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("qwen-image-edit only supports --batch 1"));
    }

    #[test]
    fn cli_batch_guard_allows_single_edit_and_other_families() {
        validate_cli_batch_for_family(Some("qwen-image-edit"), 1).unwrap();
        validate_cli_batch_for_family(Some("flux"), 4).unwrap();
        validate_cli_batch_for_family(None, 4).unwrap();
    }

    #[test]
    fn ghostty_preview_uses_direct_kitty_backend() {
        assert_eq!(
            preview_backend(Some("ghostty"), Some("xterm-ghostty")),
            PreviewBackend::GhosttyKitty
        );
        assert_eq!(
            preview_backend(Some("ghostty"), Some("xterm-256color")),
            PreviewBackend::GhosttyKitty
        );
        assert_eq!(
            preview_backend(None, Some("xterm-ghostty")),
            PreviewBackend::GhosttyKitty
        );
    }

    #[test]
    fn non_ghostty_preview_uses_viuer_auto() {
        assert_eq!(
            preview_backend(Some("WezTerm"), Some("xterm-256color")),
            PreviewBackend::ViuerAuto
        );
        assert_eq!(
            preview_backend(None, Some("xterm-256color")),
            PreviewBackend::ViuerAuto
        );
    }

    #[test]
    fn fit_preview_cells_keeps_last_row_free_when_using_full_height() {
        assert_eq!(fit_preview_cells(80, 48, 80, 24), (80, 23));
        assert_eq!(fit_preview_cells(40, 20, 80, 24), (40, 10));
    }

    #[test]
    fn ghostty_kitty_preview_payload_uses_png_chunks() {
        let payload = ghostty_kitty_preview_chunks(&"A".repeat(5000), 40, 20);
        assert!(payload.starts_with("\u{1b}_Gf=100,a=T,t=d,c=40,r=20,m=1;"));
        assert!(payload.contains("\u{1b}_Gm=0;"));
    }

    #[test]
    fn ghostty_kitty_preview_payload_uses_single_chunk_when_small() {
        let payload = ghostty_kitty_preview_chunks("AAAA", 12, 6);
        assert!(payload.starts_with("\u{1b}_Gf=100,a=T,t=d,c=12,r=6,m=0;AAAA\u{1b}\\"));
        assert!(!payload.contains("\u{1b}_Gm=1;"));
    }

    #[test]
    fn build_ghostty_kitty_preview_payload_scales_wide_images_to_terminal_width() {
        let img = image::DynamicImage::ImageRgba8(image::RgbaImage::new(200, 100));
        let (payload, cell_w, cell_h) =
            build_ghostty_kitty_preview_payload(&img, 80, 24).expect("payload should build");
        assert_eq!((cell_w, cell_h), (80, 20));
        assert!(payload.starts_with("\u{1b}_Gf=100,a=T,t=d,c=80,r=20,"));
    }

    #[test]
    fn build_ghostty_kitty_preview_payload_leaves_one_row_for_prompt_when_full_height() {
        let img = image::DynamicImage::ImageRgba8(image::RgbaImage::new(80, 48));
        let (payload, cell_w, cell_h) =
            build_ghostty_kitty_preview_payload(&img, 80, 24).expect("payload should build");
        assert_eq!((cell_w, cell_h), (80, 23));
        assert!(payload.starts_with("\u{1b}_Gf=100,a=T,t=d,c=80,r=23,"));
    }

    #[test]
    fn preview_loop_count_replays_short_clips() {
        // ≤1.5s gets 3 plays, ≤3s gets 2, longer plays once.
        assert_eq!(preview_loop_count(Duration::from_millis(500)), 3);
        assert_eq!(preview_loop_count(Duration::from_millis(1499)), 3);
        assert_eq!(preview_loop_count(Duration::from_millis(1500)), 2);
        assert_eq!(preview_loop_count(Duration::from_millis(2999)), 2);
        assert_eq!(preview_loop_count(Duration::from_millis(3000)), 1);
        assert_eq!(preview_loop_count(Duration::from_secs(30)), 1);
    }

    fn ltx2_catalog_id_config() -> Config {
        let mut config = Config::default();
        config.models.insert(
            "cv:2781713".to_string(),
            ModelConfig {
                family: Some("ltx2".to_string()),
                ..ModelConfig::default()
            },
        );
        config
    }

    #[test]
    fn forced_local_validation_passes_family_hint() {
        // `cv:` / `hf:` catalog IDs are absent from the static manifest, so the
        // forced-local path must feed the config-resolved family through or
        // every family-gated LTX-2 feature fails with "unknown model family".
        //
        // The gated field here is `spatial_upscale` and the container is one
        // every build can deliver. This used to be `enable_audio` over `mp4`:
        // since output formats became recipe-qualified, an `mp4` request is
        // invalid in a binary compiled without that feature, so the default
        // feature run failed on the container rather than exercising the family
        // gate this covers. The audio path keeps its own mp4-gated case below.
        let config = ltx2_catalog_id_config();
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a red apple",
            "model": "cv:2781713",
            "width": 960,
            "height": 576,
            "steps": 8,
            "guidance": 3.0,
            "batch_size": 1,
            "output_format": "apng",
            "spatial_upscale": "x2"
        }))
        .unwrap();

        // Sanity: without the hint the family gate rejects it.
        assert!(mold_core::validate_generate_request(&request).is_err());
        validate_local_request(&request, &config).unwrap();
    }

    /// The same hint, carrying the LTX-2-only audio flag in the one container
    /// that can hold a soundtrack — so this case exists only where the build
    /// can actually deliver it.
    #[cfg(feature = "mp4")]
    #[test]
    fn forced_local_validation_passes_family_hint_for_generated_audio() {
        let config = ltx2_catalog_id_config();
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a red apple",
            "model": "cv:2781713",
            "width": 960,
            "height": 576,
            "steps": 8,
            "guidance": 3.0,
            "batch_size": 1,
            "output_format": "mp4",
            "enable_audio": true
        }))
        .unwrap();

        assert!(mold_core::validate_generate_request(&request).is_err());
        validate_local_request(&request, &config).unwrap();
    }

    #[test]
    fn forced_local_validation_still_rejects_invalid_requests() {
        let config = Config::default();
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "",
            "model": "flux-dev:q4",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 0.0,
            "batch_size": 1
        }))
        .unwrap();

        assert!(validate_local_request(&request, &config)
            .unwrap_err()
            .to_string()
            .contains("prompt"));
    }

    #[test]
    fn forced_local_policy_respects_compiled_h3_artifact_activation() {
        let root = "/Volumes/ExternalStorage/mold-uat/minimax-h3/models";
        let mut config = Config {
            models_dir: root.to_string(),
            ..Config::default()
        };
        config.models.insert(
            "private-control".to_string(),
            ModelConfig {
                family: Some("minimax-h3".to_string()),
                ..ModelConfig::default()
            },
        );
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a red apple",
            "model": "flux-dev:q4",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 0.0,
            "batch_size": 1
        }))
        .unwrap();

        request.control_model = Some("private-control".to_string());
        assert!(require_local_request_model_activation(&request, &config).is_err());

        request.control_model = None;
        request.lora = Some(LoraWeight {
            path: format!("{root}/custom/MiniMax-H3/adapter.safetensors"),
            scale: 1.0,

            expert: None,
        });
        let h3_artifact_result = require_local_request_model_activation(&request, &config);
        if cfg!(feature = "h3") {
            h3_artifact_result.expect("public H3 builds accept trusted H3 artifact paths");
        } else {
            assert!(h3_artifact_result.is_err());
        }

        request.lora.as_mut().unwrap().path = format!("{root}/flux/ordinary-adapter.safetensors");
        assert!(require_local_request_model_activation(&request, &config).is_ok());
    }

    /// Metadata must survive a zero-length prompt on every embedded surface.
    ///
    /// `mold-inference`'s `add_metadata_chunks` writes `mold:prompt` as an
    /// UNCONDITIONAL iTXt chunk (unlike the neighbouring `Option`-guarded
    /// fields), and the JPEG COM writer interpolates the prompt unconditionally
    /// too. This pins both encoders' tolerance of `""` so an optional-prompt
    /// image-to-video run can't silently lose its provenance. The JPEG XMP
    /// packet is not covered here — nothing in mold reads it back, and an empty
    /// `<mold:prompt></mold:prompt>` element is well-formed XML.
    #[test]
    fn empty_prompt_metadata_round_trips_through_png_and_jpeg() {
        use std::io::Write;

        // `OutputMetadata` has no `Default`; start from the synthesized shape
        // and fill in the fields the round-trip asserts on.
        let mut metadata = mold_db::metadata_io::synthesize_from_filename("mold-ltx2-1.png", 1);
        metadata.prompt = String::new();
        metadata.model = "ltx-2-19b-distilled:fp8".to_string();
        metadata.seed = 7;
        metadata.steps = 8;
        metadata.guidance = 3.0;
        metadata.width = 64;
        metadata.height = 64;
        metadata.version = "test".to_string();
        let json = serde_json::to_string(&metadata).unwrap();
        let dir = tempfile::tempdir().unwrap();

        // PNG: zero-length iTXt payloads must encode and decode.
        let png_path = dir.path().join("still.png");
        {
            let file = std::fs::File::create(&png_path).unwrap();
            let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .add_itxt_chunk("mold:prompt".to_string(), metadata.prompt.clone())
                .unwrap();
            encoder
                .add_itxt_chunk("mold:parameters".to_string(), json.clone())
                .unwrap();
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[0u8, 0, 0]).unwrap();
            writer.finish().unwrap();
        }
        let recovered =
            mold_db::metadata_io::read_embedded(&png_path, mold_core::OutputFormat::Png)
                .expect("png metadata should decode with an empty prompt");
        assert_eq!(recovered.prompt, "");
        assert_eq!(recovered.model, "ltx-2-19b-distilled:fp8");

        // JPEG: the COM marker carries `mold:parameters {json}` verbatim, so an
        // empty prompt only shortens the JSON.
        let jpeg_path = dir.path().join("still.jpg");
        {
            let comment = format!("mold:parameters {json}");
            let mut out = vec![0xFF, 0xD8];
            out.push(0xFF);
            out.push(0xFE);
            let len = (comment.len() + 2) as u16;
            out.extend_from_slice(&len.to_be_bytes());
            out.extend_from_slice(comment.as_bytes());
            out.extend_from_slice(&minimal_jpeg_body());
            std::fs::File::create(&jpeg_path)
                .unwrap()
                .write_all(&out)
                .unwrap();
        }
        let recovered =
            mold_db::metadata_io::read_embedded(&jpeg_path, mold_core::OutputFormat::Jpeg)
                .expect("jpeg metadata should decode with an empty prompt");
        assert_eq!(recovered.prompt, "");
        assert_eq!(recovered.seed, 7);
    }

    /// A 1x1 baseline JPEG minus its leading SOI marker, produced by the
    /// `image` crate so the COM-marker test has a real scan to append.
    fn minimal_jpeg_body() -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        image::RgbImage::new(1, 1)
            .write_with_encoder(image::codecs::jpeg::JpegEncoder::new(&mut buf))
            .unwrap();
        buf.into_inner()[2..].to_vec()
    }
}

#[cfg(test)]
mod hdr_chain_guard_tests {
    #[test]
    fn remote_hdr_exr_is_rejected_before_request_construction() {
        let path = "/tmp/client-side-exr".to_string();
        let error = super::require_local_hdr_exr_dir(Some(path.clone()), false).unwrap_err();
        assert!(error.to_string().contains("--local"), "got: {error:#}");
        assert!(
            error.to_string().contains("remote server"),
            "got: {error:#}"
        );

        assert_eq!(
            super::require_local_hdr_exr_dir(Some(path.clone()), true).unwrap(),
            Some(path)
        );
        assert_eq!(super::require_local_hdr_exr_dir(None, false).unwrap(), None);

        let source = include_str!("generate.rs");
        let gate = source
            .find("require_local_hdr_exr_dir(hdr_exr_dir, local)?")
            .expect("run must apply the local-only HDR gate");
        let request = source
            .find("let mut req = GenerateRequest")
            .expect("run must construct the generation request");
        assert!(
            gate < request,
            "the local-only gate must run before a normal remote GenerateRequest can be built"
        );
    }

    /// Forced-local chaining must retain the sidecar authority after the
    /// shared local-only gate rather than dropping its export metadata.
    #[test]
    fn auto_chaining_routes_the_exr_sidecar_to_the_local_orchestrator() {
        let source = include_str!("generate.rs");
        let chain_arm = source
            .split("ChainRoutingDecision::Chain {")
            .nth(1)
            .expect("the Chain routing arm must exist");
        assert!(
            chain_arm.contains("ChainHdrInputs"),
            "the local branch must build the ChainHdrInputs authority"
        );
        assert!(
            chain_arm.contains("exact_fit_last_stage_for_sidecar"),
            "an HDR chain must deliver exactly the requested total so the EXR \
             sequence and the reference slices both follow the stitched timeline"
        );
    }

    /// Both stage-request builders keep pinning `hdr_exr_dir: None`: the
    /// sidecar rides `ChainStageRenderer::render_stage`'s dedicated argument
    /// (stamped onto the plan by `render_chain_stage`), so a stage request
    /// built anywhere else can never claim an EXR target by accident.
    #[test]
    fn chain_stage_requests_still_drop_the_exr_directory() {
        for (label, source) in [
            (
                "inference orchestrator",
                include_str!("../../../mold-inference/src/chain/orchestrator.rs"),
            ),
            (
                "server chain runner",
                include_str!("../../../mold-server/src/chain_job_runner.rs"),
            ),
        ] {
            assert!(
                source.contains("hdr_exr_dir: None"),
                "{label} no longer pins hdr_exr_dir to None — the render_stage \
                 sidecar argument must stay the one carrier, or per-stage EXR \
                 numbering can drift from the stitched timeline"
            );
        }
    }
}

#[cfg(test)]
mod audio_batch_passthrough_tests {
    use super::*;

    fn audio_response(seed: u64, bytes: &[u8]) -> GenerateResponse {
        GenerateResponse {
            request_warnings: Vec::new(),
            audio: Some(mold_core::AudioData {
                data: bytes.to_vec(),
                format: OutputFormat::Wav,
                sample_rate: 24_000,
                channels: 2,
                duration_ms: 5_010,
                thumbnail: Vec::new(),
                thumbnail_width: 640,
                thumbnail_height: 360,
            }),
            images: Vec::new(),
            video: None,
            generation_time_ms: 1,
            model: "ltx-2-19b-dev:fp8".to_string(),
            seed_used: seed,
            gpu: None,
        }
    }

    fn ctx<'a>(output: &'a Option<String>, batch: u32) -> BatchSaveContext<'a> {
        BatchSaveContext {
            output,
            model: "ltx-2-19b-dev:fp8",
            output_format: OutputFormat::Wav,
            preview: false,
            batch,
        }
    }

    /// The collector assembles the response the CLI's output branch reads.
    /// It handled `images` and `video` and hard-coded `audio: None`, so an
    /// audio-only pipeline denoised for ten minutes, produced a waveform, and
    /// then reported `✓ Done` having written no file — the audio branch saw
    /// `None` and never ran.
    #[test]
    fn a_single_item_batch_carries_its_audio_out_for_the_caller_to_save() {
        let output = None;
        let mut collected = BatchOutputs::new(1, 42);
        collected
            .absorb(0, audio_response(42, b"RIFF-one"), &ctx(&output, 1), None)
            .unwrap();

        let response = collected.finish();
        let audio = response.audio.expect("the only output must survive");
        assert_eq!(audio.data, b"RIFF-one".to_vec());
        assert_eq!(audio.sample_rate, 24_000);
        assert_eq!(response.seed_used, 42);
    }

    /// A WAV is its item's *only* artifact, so an aggregate that keeps just
    /// the last one loses every earlier render as soon as a later sibling
    /// fails. Each track has to land on disk before the batch moves on, under
    /// its own index — exactly as video and images already do.
    #[test]
    fn every_item_in_a_batch_lands_on_disk_under_its_own_index() {
        let dir = tempfile::tempdir().unwrap();
        let output = Some(dir.path().join("take.wav").to_string_lossy().into_owned());
        let save_ctx = ctx(&output, 3);
        let mut collected = BatchOutputs::new(3, 7);

        for index in 0..3u32 {
            let bytes = format!("RIFF-{index}");
            collected
                .absorb(
                    index,
                    audio_response(7 + u64::from(index), bytes.as_bytes()),
                    &save_ctx,
                    None,
                )
                .unwrap();
        }

        for index in 0..3u32 {
            let path = dir.path().join(format!("take-{index}.wav"));
            assert_eq!(
                std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
                format!("RIFF-{index}").into_bytes(),
                "item {index} must be persisted before a later sibling can fail",
            );
        }

        let response = collected.finish();
        assert_eq!(response.seed_used, 9, "the aggregate reports the last seed");
        assert_eq!(
            response.audio.expect("aggregate keeps the last track").data,
            b"RIFF-2".to_vec(),
        );
    }

    /// A single-item batch must NOT write inside the collector — the caller's
    /// output branch owns that write, and it is the one that honours `-` /
    /// piped stdout. Two writes would also mean two gallery rows.
    #[test]
    fn a_single_item_batch_defers_the_write_to_the_output_branch() {
        let dir = tempfile::tempdir().unwrap();
        let output = Some(dir.path().join("take.wav").to_string_lossy().into_owned());
        let mut collected = BatchOutputs::new(1, 1);
        collected
            .absorb(0, audio_response(1, b"RIFF-one"), &ctx(&output, 1), None)
            .unwrap();

        assert!(
            std::fs::read_dir(dir.path()).unwrap().next().is_none(),
            "the collector must not write for a batch of one",
        );
    }
}
