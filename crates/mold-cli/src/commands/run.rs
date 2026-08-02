use anyhow::Result;
use clap_complete::engine::CompletionCandidate;
use mold_core::manifest::{
    all_generation_model_names, is_known_model, looks_like_model_name, resolve_model_name,
    suggest_similar_models,
};
use mold_core::{
    parse_device_ref_str, AdvancedPlacement, Config, DevicePlacement, KeyframeCondition,
    LoraWeight, Ltx2GuidanceOverrides, Ltx2PipelineMode, Ltx2SpatialUpscale, Ltx2TemporalUpscale,
    OutputFormat, Scheduler, TimeRange,
};
use std::io::{IsTerminal, Read};
use std::path::Path;

use crate::{Ltx2PipelineArg, Ltx2SpatialUpscaleArg, Ltx2TemporalUpscaleArg};

use super::generate;

/// Provide model name completions for shell tab-completion.
pub fn complete_model_name() -> Vec<CompletionCandidate> {
    let config = Config::load_or_default();
    all_generation_model_names(&config)
        .into_iter()
        .map(CompletionCandidate::new)
        .collect()
}

/// Resolve positional args into (model, prompt).
///
/// Rules:
/// - If model_or_prompt matches a known model → (model, prompt_rest joined).
/// - If model_or_prompt is a catalog ID already installed in `config.models`
///   (the async catalog bridge runs ahead of this) → use it as the model.
/// - If model_or_prompt looks like a model name but isn't known → error with suggestions.
/// - Else → (config default_model, all args joined as prompt).
/// - Empty prompt → None. Whether that is fatal is decided later by
///   [`require_prompt`], which lets conditioned video runs proceed unprompted.
fn resolve_run_args(
    model_or_prompt: Option<&str>,
    prompt_rest: &[String],
    config: &mut Config,
) -> Result<(String, Option<String>)> {
    if let Some(first) = model_or_prompt {
        if is_known_model(first, config) {
            let prompt = if prompt_rest.is_empty() {
                None
            } else {
                Some(prompt_rest.join(" "))
            };
            return Ok((resolve_model_name(first), prompt));
        }

        // Catalog ID short-circuit: the async pre-pass (`ensure_catalog_model`)
        // already inserted a synthesized ModelConfig into `config.models` for
        // `cv:<id>` / `hf:<author>/<name>` inputs. Catalog IDs are their own
        // canonical name (not run through `resolve_model_name`), since they
        // don't take tag suffixes.
        if crate::catalog_bridge::looks_like_catalog_id(first) && config.models.contains_key(first)
        {
            let prompt = if prompt_rest.is_empty() {
                None
            } else {
                Some(prompt_rest.join(" "))
            };
            return Ok((first.to_string(), prompt));
        }

        // Check if the first arg looks like it was intended as a model name
        if looks_like_model_name(first, config) {
            let suggestions = suggest_similar_models(first, config, 5);
            let mut msg = format!("unknown model '{first}'");
            if !suggestions.is_empty() {
                msg.push_str("\n\n  Did you mean one of these?");
                for s in &suggestions {
                    msg.push_str(&format!("\n    {s}"));
                }
            }
            msg.push_str("\n\n  hint: Run 'mold list' to see all available models.");
            anyhow::bail!(msg);
        }

        // First arg is part of the prompt, not a model
        let mut parts = vec![first.to_string()];
        parts.extend(prompt_rest.iter().cloned());
        let model = resolve_model_name(&config.resolved_default_model());
        return Ok((model, Some(parts.join(" "))));
    }

    // No args at all
    Ok((resolve_model_name(&config.resolved_default_model()), None))
}

/// Validate file-based CLI arguments early, before expansion or inference.
///
/// Checks:
/// - `--lora`: must exist, be a file (not directory), end in `.safetensors`
/// - `--image`: must exist (unless `-` for stdin)
/// - `--mask`: must exist
/// - `--control`: must exist
/// - `--output`: parent directory must exist; if path is a directory, error with hint
fn resolve_family(model_name: &str, config: &Config) -> String {
    config
        .resolved_model_config(model_name)
        .family
        .or_else(|| mold_core::manifest::find_manifest(model_name).map(|m| m.family.clone()))
        .unwrap_or_else(|| "flux".to_string())
}

/// Resolve the final prompt for a run.
///
/// A video family that already carries visual conditioning (source image,
/// keyframes, source video, or an extend) may run unprompted — LTX-2's text
/// encoder pads to a fixed-width context, so `""` is a trained input rather
/// than a degenerate one. Expect near-static, micro-motion output; it saves no
/// VRAM, since the prompt-context tensor is a fixed size either way. Every
/// other case still needs a prompt.
fn require_prompt(
    prompt: Option<String>,
    family: &str,
    has_visual_conditioning: bool,
) -> Result<String> {
    if let Some(prompt) = prompt {
        return Ok(prompt);
    }
    if !mold_core::prompt_required_with_conditioning(Some(family), has_visual_conditioning) {
        return Ok(String::new());
    }
    Err(anyhow::anyhow!(
        "no prompt provided\n\n\
         Usage: mold run [MODEL] <PROMPT>\n\
         Example: mold run flux-dev:q4 \"a turtle in the desert\"\n\
         Stdin:   echo \"a turtle\" | mold run flux-dev:q4"
    ))
}

#[derive(Default, Clone, Copy)]
struct FileArgRefs<'a> {
    lora: Option<&'a str>,
    image: Option<&'a str>,
    mask: Option<&'a str>,
    control: Option<&'a str>,
    audio: Option<&'a str>,
    video: Option<&'a str>,
    camera_control: Option<&'a str>,
    output: Option<&'a str>,
}

#[cfg(test)]
fn validate_file_args(
    lora: Option<&str>,
    image: Option<&str>,
    mask: Option<&str>,
    control: Option<&str>,
    output: Option<&str>,
) -> Result<()> {
    validate_file_args_full(FileArgRefs {
        lora,
        image,
        mask,
        control,
        output,
        ..FileArgRefs::default()
    })
}

fn validate_file_args_full(args: FileArgRefs<'_>) -> Result<()> {
    // -- --lora validation --
    if let Some(lora_path) = args.lora {
        if !is_virtual_lora_alias(lora_path) {
            let p = Path::new(lora_path);
            if p.is_dir() {
                // List .safetensors files in the directory as suggestions
                let mut suggestions: Vec<String> = Vec::new();
                if let Ok(entries) = std::fs::read_dir(p) {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        if let Some(name_str) = name.to_str() {
                            if name_str.ends_with(".safetensors") {
                                suggestions.push(entry.path().display().to_string());
                            }
                        }
                    }
                }
                suggestions.sort();
                let mut msg = format!("--lora path '{}' is a directory, not a file", lora_path);
                if suggestions.is_empty() {
                    msg.push_str(" (no .safetensors files found inside)");
                } else {
                    msg.push_str(". Did you mean one of these?");
                    for s in &suggestions {
                        msg.push_str(&format!("\n    {s}"));
                    }
                }
                anyhow::bail!(msg);
            }
            if !p.exists() {
                anyhow::bail!("--lora file not found: {lora_path}");
            }
            if !lora_path.ends_with(".safetensors") {
                anyhow::bail!("--lora file must be a .safetensors file, got: {lora_path}");
            }
        }
    }

    // -- --image validation --
    if let Some(img_path) = args.image {
        if img_path != "-" {
            let p = Path::new(img_path);
            if p.is_dir() {
                anyhow::bail!("--image path is a directory, not an image file: {img_path}");
            }
            if !p.exists() {
                anyhow::bail!("--image file not found: {img_path}");
            }
        }
    }

    // -- --mask validation --
    if let Some(mask_path) = args.mask {
        let p = Path::new(mask_path);
        if p.is_dir() {
            anyhow::bail!("--mask path is a directory, not an image file: {mask_path}");
        }
        if !p.exists() {
            anyhow::bail!("--mask file not found: {mask_path}");
        }
    }

    // -- --control validation --
    if let Some(ctrl_path) = args.control {
        let p = Path::new(ctrl_path);
        if p.is_dir() {
            anyhow::bail!("--control path is a directory, not an image file: {ctrl_path}");
        }
        if !p.exists() {
            anyhow::bail!("--control file not found: {ctrl_path}");
        }
    }

    if let Some(audio_path) = args.audio {
        let p = Path::new(audio_path);
        if p.is_dir() {
            anyhow::bail!("--audio-file path is a directory, not a file: {audio_path}");
        }
        if !p.exists() {
            anyhow::bail!("--audio-file file not found: {audio_path}");
        }
    }

    if let Some(video_path) = args.video {
        let p = Path::new(video_path);
        if p.is_dir() {
            anyhow::bail!("--video path is a directory, not a file: {video_path}");
        }
        if !p.exists() {
            anyhow::bail!("--video file not found: {video_path}");
        }
    }

    if let Some(camera_control_value) = args.camera_control {
        if camera_control_value.ends_with(".safetensors") {
            let p = Path::new(camera_control_value);
            if p.is_dir() {
                anyhow::bail!(
                    "--camera-control path is a directory, not a .safetensors file: {camera_control_value}"
                );
            }
            if !p.exists() {
                anyhow::bail!("--camera-control file not found: {camera_control_value}");
            }
        }
        // Compatibility is decided from the *resolved* checkpoint artifacts,
        // by the server on the request path and by the engine locally. A
        // model-name substring test cannot see the architecture behind an
        // opaque `cv:` / `hf:` catalog ID, so it used to reject a perfectly
        // good LTX-2 19B install and wave an LTX-2.3 one straight through.
    }

    // -- --output validation --
    if let Some(out_path) = args.output {
        if out_path != "-" {
            let p = Path::new(out_path);
            if p.is_dir() {
                anyhow::bail!(
                    "--output '{}' is a directory. Provide a filename, e.g.: {}/image.png",
                    out_path,
                    p.display()
                );
            }
            if let Some(parent) = p.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    anyhow::bail!("output directory does not exist: {}", parent.display());
                }
            }
        }
    }

    Ok(())
}

fn is_virtual_lora_alias(value: &str) -> bool {
    value
        .strip_prefix("camera-control:")
        .is_some_and(|preset| !preset.trim().is_empty())
}

fn is_flux2_dev_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("flux2-dev") || model.contains("flux.2-dev")
}

fn validate_image_args_for_model(family: &str, model: &str, image: &[String]) -> Result<()> {
    let flux2_dev = is_flux2_dev_model(model);
    let ordered_images = family == "qwen-image-edit" || flux2_dev;
    if family == "qwen-image-edit" && image.iter().any(|img| img == "-") {
        anyhow::bail!("qwen-image-edit does not support --image -; pass file paths instead");
    }
    if flux2_dev && image.iter().any(|img| img == "-") {
        anyhow::bail!(
            "FLUX.2 Dev ordered references do not support --image -; pass file paths instead"
        );
    }
    if !ordered_images && image.len() > 1 {
        anyhow::bail!(
            "multiple --image values are only supported for Qwen-Image-Edit and FLUX.2 Dev models"
        );
    }
    if flux2_dev && image.len() > mold_core::validation::FLUX2_DEV_MAX_REFERENCE_IMAGES {
        anyhow::bail!(
            "FLUX.2 Dev supports at most {} ordered --image references",
            mold_core::validation::FLUX2_DEV_MAX_REFERENCE_IMAGES
        );
    }
    Ok(())
}

fn parse_pipeline(value: Option<Ltx2PipelineArg>) -> Option<Ltx2PipelineMode> {
    value.map(|value| match value {
        Ltx2PipelineArg::OneStage => Ltx2PipelineMode::OneStage,
        Ltx2PipelineArg::TwoStage => Ltx2PipelineMode::TwoStage,
        Ltx2PipelineArg::TwoStageHq => Ltx2PipelineMode::TwoStageHq,
        Ltx2PipelineArg::Distilled => Ltx2PipelineMode::Distilled,
        Ltx2PipelineArg::IcLora => Ltx2PipelineMode::IcLora,
        Ltx2PipelineArg::Keyframe => Ltx2PipelineMode::Keyframe,
        Ltx2PipelineArg::A2Vid => Ltx2PipelineMode::A2Vid,
        Ltx2PipelineArg::Retake => Ltx2PipelineMode::Retake,
    })
}

fn resolve_ic_lora_pipeline(
    pipeline: Option<Ltx2PipelineMode>,
    control: Option<&str>,
    has_video: bool,
) -> anyhow::Result<Option<Ltx2PipelineMode>> {
    let Some(_) = control else {
        return Ok(pipeline);
    };
    if !has_video {
        anyhow::bail!("--ic-lora-control requires --video");
    }
    match pipeline {
        None | Some(Ltx2PipelineMode::IcLora) => Ok(Some(Ltx2PipelineMode::IcLora)),
        Some(other) => anyhow::bail!(
            "--ic-lora-control conflicts with --pipeline {other}; use --pipeline ic-lora"
        ),
    }
}

fn resolve_effective_loras_for_family(
    family: &str,
    lora: &[String],
    lora_scale: f64,
    default_lora: Option<LoraWeight>,
    camera_control: Option<String>,
) -> Result<(Option<LoraWeight>, Option<Vec<LoraWeight>>)> {
    if lora.len() > 1 && !mold_core::family_supports_lora(family) {
        anyhow::bail!("multiple --lora values are only supported for LoRA-capable models");
    }

    let effective_lora = if let Some(lora_path) = lora.first() {
        Some(LoraWeight {
            path: lora_path.clone(),
            scale: lora_scale,
        })
    } else {
        default_lora
    };

    let mut stack = Vec::new();
    if lora.len() > 1 {
        stack.extend(lora.iter().cloned().map(|path| LoraWeight {
            path,
            scale: lora_scale,
        }));
    } else if family == "ltx2" {
        if !lora.is_empty() {
            stack.extend(lora.iter().cloned().map(|path| LoraWeight {
                path,
                scale: lora_scale,
            }));
        } else if let Some(lora) = effective_lora.clone() {
            stack.push(lora);
        }
    }

    if let Some(camera_control) = camera_control {
        let path = if camera_control.ends_with(".safetensors") {
            camera_control
        } else {
            format!("camera-control:{camera_control}")
        };
        stack.push(LoraWeight { path, scale: 1.0 });
    }

    let lora_stack = if stack.is_empty() { None } else { Some(stack) };
    let effective_lora = if lora_stack.is_some() && lora.len() > 1 {
        None
    } else {
        effective_lora
    };
    Ok((effective_lora, lora_stack))
}

fn parse_spatial_upscale(value: Option<Ltx2SpatialUpscaleArg>) -> Option<Ltx2SpatialUpscale> {
    value.map(|value| match value {
        Ltx2SpatialUpscaleArg::X1_5 => Ltx2SpatialUpscale::X1_5,
        Ltx2SpatialUpscaleArg::X2 => Ltx2SpatialUpscale::X2,
    })
}

fn parse_temporal_upscale(value: Option<Ltx2TemporalUpscaleArg>) -> Option<Ltx2TemporalUpscale> {
    value.map(|Ltx2TemporalUpscaleArg::X2| Ltx2TemporalUpscale::X2)
}

/// Raw `--stg-scale` / `--stg-blocks` / … flags, grouped so `run` keeps one
/// parameter per feature rather than five more positional arguments.
#[derive(Debug, Default, Clone)]
pub struct GuidanceFlags {
    pub stg_scale: Option<f64>,
    pub stg_blocks: Option<String>,
    pub rescale_scale: Option<f64>,
    pub modality_scale: Option<f64>,
    pub skip_step: Option<u32>,
}

impl GuidanceFlags {
    /// Build the wire overrides, or `None` when the user passed no guidance
    /// flag at all — an absent field is what preserves the pipeline default,
    /// so an empty object must never reach the server.
    fn into_overrides(self) -> Result<Option<Ltx2GuidanceOverrides>> {
        let stg_blocks = self
            .stg_blocks
            .as_deref()
            .map(parse_stg_blocks)
            .transpose()?;
        Ok(Ltx2GuidanceOverrides {
            stg_scale: self.stg_scale,
            stg_blocks,
            rescale_scale: self.rescale_scale,
            modality_scale: self.modality_scale,
            skip_step: self.skip_step,
        }
        .into_option())
    }
}

fn parse_stg_blocks(value: &str) -> Result<Vec<u32>> {
    let blocks = value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            entry.parse::<u32>().map_err(|_| {
                anyhow::anyhow!("--stg-blocks takes comma-separated block indices, got '{entry}'")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if blocks.is_empty() {
        anyhow::bail!("--stg-blocks must list at least one transformer block index");
    }
    Ok(blocks)
}

fn parse_retake_range(value: Option<String>) -> Result<Option<TimeRange>> {
    value
        .map(|value| {
            let (start, end) = value
                .split_once(':')
                .ok_or_else(|| anyhow::anyhow!("--retake must be in <start:end> format"))?;
            Ok(TimeRange {
                start_seconds: start.parse()?,
                end_seconds: end.parse()?,
            })
        })
        .transpose()
}

fn parse_keyframes(values: &[String]) -> Result<Option<Vec<KeyframeCondition>>> {
    if values.is_empty() {
        return Ok(None);
    }

    let mut keyframes = Vec::with_capacity(values.len());
    for value in values {
        let (frame, path) = value
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("--keyframe must be in <frame:path> format"))?;
        let path = Path::new(path);
        if !path.exists() {
            anyhow::bail!("--keyframe file not found: {}", path.display());
        }
        keyframes.push(KeyframeCondition {
            frame: frame.parse()?,
            image: std::fs::read(path)?,
        });
    }

    Ok(Some(keyframes))
}

#[allow(clippy::too_many_arguments)]
/// CLI device-placement overrides collected from the seven `--device-*` flags.
/// Each field is the raw user-supplied string (e.g. `"cpu"`, `"gpu:1"`); parsing
/// happens in [`resolve_placement`] so the caller gets a single aggregate error.
#[derive(Debug, Clone, Default)]
pub struct PlacementFlags {
    pub text_encoders: Option<String>,
    pub transformer: Option<String>,
    pub vae: Option<String>,
    pub t5: Option<String>,
    pub clip_l: Option<String>,
    pub clip_g: Option<String>,
    pub qwen: Option<String>,
}

impl PlacementFlags {
    fn any_set(&self) -> bool {
        self.text_encoders.is_some()
            || self.transformer.is_some()
            || self.vae.is_some()
            || self.t5.is_some()
            || self.clip_l.is_some()
            || self.clip_g.is_some()
            || self.qwen.is_some()
    }

    fn any_advanced(&self) -> bool {
        self.transformer.is_some()
            || self.vae.is_some()
            || self.t5.is_some()
            || self.clip_l.is_some()
            || self.clip_g.is_some()
            || self.qwen.is_some()
    }
}

/// Merge CLI placement flags on top of the effective placement from config +
/// env vars (via `config.resolved_placement(model)`). Returns `Ok(None)` when
/// nothing overrides the engine's built-in auto selection. Flag parse errors
/// surface with the flag name so users know which `--device-*` was bad.
fn resolve_placement(
    config: &Config,
    model: &str,
    flags: &PlacementFlags,
) -> Result<Option<DevicePlacement>> {
    let parse = |flag: &str, raw: &Option<String>| -> Result<Option<_>> {
        raw.as_deref()
            .map(|s| parse_device_ref_str(s).map_err(|e| anyhow::anyhow!("--device-{flag}: {e}")))
            .transpose()
    };

    let base = config.resolved_placement(model);
    if !flags.any_set() {
        return Ok(base);
    }

    let mut effective: DevicePlacement = base.unwrap_or_default();
    if let Some(r) = parse("text-encoders", &flags.text_encoders)? {
        effective.text_encoders = r;
    }
    if flags.any_advanced() {
        let mut adv: AdvancedPlacement = effective.advanced.unwrap_or_default();
        if let Some(r) = parse("transformer", &flags.transformer)? {
            adv.transformer = r;
        }
        if let Some(r) = parse("vae", &flags.vae)? {
            adv.vae = r;
        }
        if let Some(r) = parse("t5", &flags.t5)? {
            adv.t5 = Some(r);
        }
        if let Some(r) = parse("clip-l", &flags.clip_l)? {
            adv.clip_l = Some(r);
        }
        if let Some(r) = parse("clip-g", &flags.clip_g)? {
            adv.clip_g = Some(r);
        }
        if let Some(r) = parse("qwen", &flags.qwen)? {
            adv.qwen = Some(r);
        }
        effective.advanced = Some(adv);
    }
    Ok(Some(effective))
}

#[allow(clippy::too_many_arguments)]
pub async fn run(
    model_or_prompt: Option<String>,
    prompt_rest: Vec<String>,
    output: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    guidance: Option<f64>,
    seed: Option<u64>,
    batch: u32,
    frames: Option<u32>,
    fps: Option<u32>,
    clip_frames: Option<u32>,
    motion_tail: u32,
    audio: bool,
    no_audio: bool,
    audio_file: Option<String>,
    video: Option<String>,
    extend: Option<String>,
    extend_overlap: Option<u32>,
    keyframe: Vec<String>,
    pipeline: Option<Ltx2PipelineArg>,
    ic_lora_control: Option<String>,
    retake: Option<String>,
    spatial_upscale: Option<Ltx2SpatialUpscaleArg>,
    temporal_upscale: Option<Ltx2TemporalUpscaleArg>,
    guidance_flags: GuidanceFlags,
    camera_control: Option<String>,
    host: Option<String>,
    format: OutputFormat,
    no_metadata: bool,
    preview: bool,
    local: bool,
    gpus: Option<String>,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Option<String>,
    scheduler: Option<Scheduler>,
    cfg_plus: bool,
    eager: bool,
    offload: bool,
    placement_flags: PlacementFlags,
    lora: Vec<String>,
    lora_scale: f64,
    image: Vec<String>,
    strength: f64,
    mask: Option<String>,
    control: Option<String>,
    control_model: Option<String>,
    control_scale: f64,
    negative_prompt: Option<String>,
    no_negative: bool,
    expand: bool,
    no_expand: bool,
    expand_backend: Option<String>,
    expand_model: Option<String>,
) -> Result<()> {
    let mut config = Config::load_or_default();

    // Async catalog ID pre-pass: if the user typed `cv:<id>` / `hf:<repo>`,
    // resolve it (sidecar-first, then live) and inject the synthesized
    // ModelConfig into config.models before the (sync) resolve_run_args below
    // picks it up. A no-op for non-catalog first positionals (e.g. a prompt).
    if let Some(first) = model_or_prompt.as_deref() {
        crate::catalog_bridge::ensure_catalog_model(&mut config, first).await?;
    }

    let (model, prompt) = resolve_run_args(model_or_prompt.as_deref(), &prompt_rest, &mut config)?;
    let family = resolve_family(&model, &config);

    // Validate file-based arguments early — before expansion or inference.
    validate_file_args_full(FileArgRefs {
        lora: lora.first().map(String::as_str),
        image: image.first().map(String::as_str),
        mask: mask.as_deref(),
        control: control.as_deref(),
        audio: audio_file.as_deref(),
        video: video.as_deref(),
        camera_control: camera_control.as_deref(),
        output: output.as_deref(),
    })?;
    for lora_path in &lora {
        validate_file_args_full(FileArgRefs {
            lora: Some(lora_path.as_str()),
            ..FileArgRefs::default()
        })?;
    }
    for extra_image in image.iter().skip(1) {
        validate_file_args_full(FileArgRefs {
            image: Some(extra_image.as_str()),
            ..FileArgRefs::default()
        })?;
    }

    validate_image_args_for_model(&family, &model, &image)?;

    let loaded_images = image
        .iter()
        .map(|img_path| {
            if img_path == "-" {
                let mut buf = Vec::new();
                std::io::stdin().read_to_end(&mut buf)?;
                Ok(buf)
            } else {
                std::fs::read(img_path)
                    .map_err(|e| anyhow::anyhow!("failed to read image '{}': {e}", img_path))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let ordered_images = family == "qwen-image-edit" || is_flux2_dev_model(&model);
    let source_image = if ordered_images {
        None
    } else {
        loaded_images.first().cloned()
    };
    let edit_images = if ordered_images && !loaded_images.is_empty() {
        Some(loaded_images)
    } else {
        None
    };

    // Read control image if --control specified
    let control_image = if let Some(ref ctrl_path) = control {
        let bytes = std::fs::read(ctrl_path)
            .map_err(|e| anyhow::anyhow!("failed to read control image '{}': {e}", ctrl_path))?;
        Some(bytes)
    } else {
        None
    };

    // Read mask image if --mask specified
    let mask_image = if let Some(ref mask_path) = mask {
        let bytes = std::fs::read(mask_path)
            .map_err(|e| anyhow::anyhow!("failed to read mask '{}': {e}", mask_path))?;
        Some(bytes)
    } else {
        None
    };
    let audio_file_bytes = audio_file.as_deref().map(std::fs::read).transpose()?;
    let source_video_bytes = video.as_deref().map(std::fs::read).transpose()?;
    let extend_video_bytes = extend.as_deref().map(std::fs::read).transpose()?;
    let keyframes = parse_keyframes(&keyframe)?;
    let pipeline = resolve_ic_lora_pipeline(
        parse_pipeline(pipeline),
        ic_lora_control.as_deref(),
        video.is_some(),
    )?;
    let retake_range = parse_retake_range(retake)?;
    let spatial_upscale = parse_spatial_upscale(spatial_upscale);
    let temporal_upscale = parse_temporal_upscale(temporal_upscale);
    let guidance_overrides = guidance_flags.into_overrides()?;

    // If no prompt from args, try reading from stdin (supports piping)
    // When --image - is used, stdin is consumed for the image, so prompt must come from args.
    let prompt = match prompt {
        Some(p) => Some(p),
        None if !image.iter().any(|img| img == "-") && !std::io::stdin().is_terminal() => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            let trimmed = buf.trim().to_string();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        }
        None => None,
    };

    let has_visual_conditioning = source_image.is_some()
        || keyframes.as_ref().is_some_and(|k| !k.is_empty())
        || source_video_bytes.is_some()
        || extend_video_bytes.is_some();
    let prompt = require_prompt(prompt, &family, has_visual_conditioning)?;

    // --- Prompt expansion ---
    // An unprompted conditioned video run has nothing to expand; handing "" to
    // the expander would let it invent the prompt (matches the server's
    // `maybe_expand_prompt` guard). `expand.enabled` must not turn that on
    // behind the user's back either, so the check covers both switches.
    let expand_settings = config.expand.clone().with_env_overrides();
    let should_expand = if no_expand || prompt.trim().is_empty() {
        false
    } else {
        expand || expand_settings.enabled
    };
    if should_expand {
        mold_core::expand::validate_expansion_variation_count(batch.max(1) as usize)?;
    }

    // Expansion strategy:
    // - If --local or server unreachable: expand client-side (existing path)
    // - If remote: delegate to server (single request: expand=true on GenerateRequest;
    //   batch: call /api/expand for all variations upfront)
    let defer_expand_to_server = should_expand && !local;
    let (final_prompt, original_prompt, batch_prompts, server_expand) =
        if should_expand && !defer_expand_to_server {
            // --- Client-side expansion (--local mode or forced local) ---
            use colored::Colorize;

            let mut settings = expand_settings;
            if let Some(ref backend) = expand_backend {
                settings.backend = backend.clone();
            }
            if let Some(ref m) = expand_model {
                if settings.is_local() {
                    settings.model = m.clone();
                } else {
                    settings.api_model = m.clone();
                }
            }

            // Validate custom templates if present
            let template_errors = settings.validate_templates();
            if !template_errors.is_empty() {
                for err in &template_errors {
                    eprintln!("{} {err}", crate::theme::prefix_warning());
                }
            }

            let model_family = super::expand::resolve_family_from_config(&model, &config);
            let expand_config = settings.to_expand_config(&model_family, batch.max(1) as usize);

            let expander = super::expand::create_expander(&settings, &config).await?;

            crate::output::status!("{} Expanding prompt...", crate::theme::icon_info());

            let result = expander.expand(&prompt, &expand_config)?;
            validate_run_expansion(&result.expanded, batch)?;

            if result.expanded.len() == 1 {
                let expanded = &result.expanded[0];
                let display = if expanded.chars().count() > 80 {
                    let truncated: String = expanded.chars().take(77).collect();
                    format!("{truncated}...")
                } else {
                    expanded.clone()
                };
                crate::output::status!(
                    "{} Expanded: \"{}\"",
                    crate::theme::icon_ok(),
                    display.dimmed()
                );
                (expanded.clone(), Some(prompt.clone()), None, None)
            } else {
                // Multiple variations: each batch image gets a different prompt.
                crate::output::status!(
                    "{} Generated {} prompt variations",
                    crate::theme::icon_ok(),
                    result.expanded.len()
                );
                for (i, expanded) in result.expanded.iter().enumerate() {
                    let display = if expanded.chars().count() > 70 {
                        let truncated: String = expanded.chars().take(67).collect();
                        format!("{truncated}...")
                    } else {
                        expanded.clone()
                    };
                    crate::output::status!("  {}: \"{}\"", i + 1, display.dimmed());
                }
                let first = result.expanded[0].clone();
                (first, Some(prompt.clone()), Some(result.expanded), None)
            }
        } else if defer_expand_to_server {
            // --- Server-side expansion via /api/expand ---
            // Always expand upfront so the prompt is ready before generate_remote.
            // This ensures the local fallback path also gets the expanded prompt.
            #[allow(unused_imports)]
            use colored::Colorize;

            let variations = batch.max(1) as usize;
            let model_family = super::expand::resolve_family_from_config(&model, &config);
            let client = match host.as_deref() {
                Some(h) => mold_core::MoldClient::new(h),
                None => mold_core::MoldClient::from_env(),
            };
            let expand_req = mold_core::ExpandRequest {
                prompt: prompt.clone(),
                model_family,
                variations,
                style: None,
            };

            crate::output::status!("{} Expanding prompt (server)...", crate::theme::icon_info());

            match client.expand_prompt(&expand_req).await {
                Ok(result) if result.expanded.len() == 1 => {
                    validate_run_expansion(&result.expanded, batch)?;
                    let expanded = &result.expanded[0];
                    let display = if expanded.chars().count() > 80 {
                        let truncated: String = expanded.chars().take(77).collect();
                        format!("{truncated}...")
                    } else {
                        expanded.clone()
                    };
                    crate::output::status!(
                        "{} Expanded (server): \"{}\"",
                        crate::theme::icon_ok(),
                        display.dimmed()
                    );
                    (expanded.clone(), Some(prompt.clone()), None, None)
                }
                Ok(result) => {
                    validate_run_expansion(&result.expanded, batch)?;
                    crate::output::status!(
                        "{} Generated {} prompt variations (server)",
                        crate::theme::icon_ok(),
                        result.expanded.len()
                    );
                    for (i, expanded) in result.expanded.iter().enumerate() {
                        let display = if expanded.chars().count() > 70 {
                            let truncated: String = expanded.chars().take(67).collect();
                            format!("{truncated}...")
                        } else {
                            expanded.clone()
                        };
                        crate::output::status!("  {}: \"{}\"", i + 1, display.dimmed());
                    }
                    let first = result.expanded[0].clone();
                    (first, Some(prompt.clone()), Some(result.expanded), None)
                }
                Err(e) if mold_core::MoldClient::is_connection_error(&e) => {
                    // Server unreachable — fall back to local expansion so the prompt
                    // is expanded even when generate_remote also falls back to local.
                    crate::output::status!(
                        "{} Server unreachable, expanding locally",
                        crate::theme::prefix_warning()
                    );
                    let mut settings = expand_settings;
                    if let Some(ref backend) = expand_backend {
                        settings.backend = backend.clone();
                    }
                    if let Some(ref m) = expand_model {
                        if settings.is_local() {
                            settings.model = m.clone();
                        } else {
                            settings.api_model = m.clone();
                        }
                    }
                    let family = super::expand::resolve_family_from_config(&model, &config);
                    let expand_config = settings.to_expand_config(&family, batch.max(1) as usize);
                    match super::expand::create_expander(&settings, &config).await {
                        Ok(expander) => match expander.expand(&prompt, &expand_config) {
                            Ok(result) => {
                                validate_run_expansion(&result.expanded, batch)?;
                                let first = result.expanded[0].clone();
                                if result.expanded.len() == 1 {
                                    (first, Some(prompt.clone()), None, None)
                                } else {
                                    (first, Some(prompt.clone()), Some(result.expanded), None)
                                }
                            }
                            Err(_) => (prompt, None, None, None),
                        },
                        Err(_) => (prompt, None, None, None),
                    }
                }
                Err(e) => return Err(e),
            }
        } else {
            (prompt, None, None, None)
        };

    // Resolve effective negative prompt: CLI flag > per-model config > global config > None.
    // --no-negative suppresses all defaults (forces empty unconditional).
    let effective_negative_prompt = if no_negative {
        None
    } else if negative_prompt.is_some() {
        negative_prompt
    } else {
        let model_cfg = config.resolved_model_config(&model);
        model_cfg.effective_negative_prompt(&config)
    };

    // Resolve LoRA: explicit CLI values override config defaults.
    let model_cfg = config.resolved_model_config(&model);
    let default_lora = model_cfg
        .effective_lora()
        .map(|(path, scale)| LoraWeight { path, scale });
    let (effective_lora, loras) = resolve_effective_loras_for_family(
        &family,
        &lora,
        lora_scale,
        default_lora,
        camera_control,
    )?;

    let placement = resolve_placement(&config, &model, &placement_flags)?;

    generate::run(
        &final_prompt,
        &model,
        output,
        width,
        height,
        steps,
        guidance,
        seed,
        batch,
        generate::Ltx2Options {
            frames,
            fps,
            clip_frames,
            motion_tail,
            enable_audio: if audio {
                Some(true)
            } else if no_audio {
                Some(false)
            } else {
                None
            },
            audio_file: audio_file_bytes,
            source_video: source_video_bytes,
            extend_video: extend_video_bytes,
            extend_overlap_frames: extend_overlap,
            keyframes,
            pipeline,
            ic_lora_control,
            loras,
            retake_range,
            spatial_upscale,
            temporal_upscale,
            guidance_overrides,
        },
        host,
        format,
        no_metadata,
        preview,
        local,
        gpus,
        t5_variant,
        qwen3_variant,
        qwen2_variant,
        qwen2_text_encoder_mode,
        scheduler,
        // CLI bool → wire-format Option<bool>: only forward an explicit `true`
        // so the server-side `MOLD_CFG_PLUS` env fallback still wins when the
        // user didn't pass the flag.
        if cfg_plus { Some(true) } else { None },
        eager,
        offload,
        placement,
        source_image,
        edit_images,
        strength,
        mask_image,
        control_image,
        control_model,
        control_scale,
        effective_negative_prompt,
        original_prompt,
        batch_prompts,
        effective_lora,
        server_expand,
    )
    .await
}

fn validate_run_expansion(expanded: &[String], batch: u32) -> Result<()> {
    mold_core::expand::validate_expanded_prompts(expanded, batch.max(1) as usize)
}

#[cfg(test)]
mod placement_flag_tests {
    use super::*;
    use crate::test_support::ENV_LOCK;
    use mold_core::DeviceRef;

    /// Mirrors `tests::test_config()` below — duplicated on purpose to avoid
    /// cross-module visibility games. Keeps the two test modules independent
    /// so adding/removing Config fields is a single local edit.
    fn minimal_config() -> Config {
        Config {
            config_version: 1,
            default_model: "flux2-klein".to_string(),
            models_dir: "/tmp/mold-test-nonexistent-models".to_string(),
            server_port: 7680,
            default_width: 1024,
            default_height: 1024,
            default_steps: 4,
            embed_metadata: true,
            t5_variant: None,
            qwen3_variant: None,
            output_dir: None,
            media_roots: None,
            default_negative_prompt: None,
            expand: mold_core::ExpandSettings::default(),
            scheduler: Default::default(),
            logging: mold_core::LoggingConfig::default(),
            runpod: mold_core::runpod::RunPodSettings::default(),
            lambda: mold_core::lambda::LambdaSettings::default(),
            gpus: None,
            queue_size: None,
            models: std::collections::HashMap::new(),
        }
    }

    fn clear_placement_env() {
        for key in [
            "MOLD_PLACE_TEXT_ENCODERS",
            "MOLD_PLACE_TRANSFORMER",
            "MOLD_PLACE_VAE",
            "MOLD_PLACE_T5",
            "MOLD_PLACE_CLIP_L",
            "MOLD_PLACE_CLIP_G",
            "MOLD_PLACE_QWEN",
        ] {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn resolve_placement_returns_none_when_nothing_set() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_placement_env();
        let cfg = minimal_config();
        let out = resolve_placement(&cfg, "flux-dev:q4", &PlacementFlags::default()).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn resolve_placement_cli_flag_sets_text_encoders() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for key in [
            "MOLD_PLACE_TEXT_ENCODERS",
            "MOLD_PLACE_TRANSFORMER",
            "MOLD_PLACE_VAE",
            "MOLD_PLACE_T5",
            "MOLD_PLACE_CLIP_L",
            "MOLD_PLACE_CLIP_G",
            "MOLD_PLACE_QWEN",
        ] {
            std::env::remove_var(key);
        }
        let cfg = minimal_config();
        let flags = PlacementFlags {
            text_encoders: Some("cpu".into()),
            ..Default::default()
        };
        let out = resolve_placement(&cfg, "flux-dev:q4", &flags)
            .unwrap()
            .expect("placement present");
        assert_eq!(out.text_encoders, DeviceRef::Cpu);
        assert!(out.advanced.is_none());
    }

    #[test]
    fn resolve_placement_cli_overrides_env() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_placement_env();
        std::env::set_var("MOLD_PLACE_TEXT_ENCODERS", "cpu");
        let cfg = minimal_config();
        let flags = PlacementFlags {
            text_encoders: Some("gpu:1".into()),
            ..Default::default()
        };
        let out = resolve_placement(&cfg, "flux-dev:q4", &flags)
            .unwrap()
            .expect("placement present");
        assert_eq!(out.text_encoders, DeviceRef::gpu(1));
        std::env::remove_var("MOLD_PLACE_TEXT_ENCODERS");
    }

    #[test]
    fn resolve_placement_tier2_flags_populate_advanced() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for key in [
            "MOLD_PLACE_TEXT_ENCODERS",
            "MOLD_PLACE_TRANSFORMER",
            "MOLD_PLACE_VAE",
            "MOLD_PLACE_T5",
            "MOLD_PLACE_CLIP_L",
            "MOLD_PLACE_CLIP_G",
            "MOLD_PLACE_QWEN",
        ] {
            std::env::remove_var(key);
        }
        let cfg = minimal_config();
        let flags = PlacementFlags {
            transformer: Some("gpu:0".into()),
            vae: Some("cpu".into()),
            t5: Some("cpu".into()),
            ..Default::default()
        };
        let out = resolve_placement(&cfg, "flux-dev:q4", &flags)
            .unwrap()
            .expect("placement present");
        let adv = out.advanced.expect("advanced populated");
        assert_eq!(adv.transformer, DeviceRef::gpu(0));
        assert_eq!(adv.vae, DeviceRef::Cpu);
        assert_eq!(adv.t5, Some(DeviceRef::Cpu));
        assert_eq!(adv.clip_l, None);
    }

    #[test]
    fn resolve_placement_invalid_flag_value_errors_with_flag_name() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for key in [
            "MOLD_PLACE_TEXT_ENCODERS",
            "MOLD_PLACE_TRANSFORMER",
            "MOLD_PLACE_VAE",
            "MOLD_PLACE_T5",
            "MOLD_PLACE_CLIP_L",
            "MOLD_PLACE_CLIP_G",
            "MOLD_PLACE_QWEN",
        ] {
            std::env::remove_var(key);
        }
        let cfg = minimal_config();
        let flags = PlacementFlags {
            vae: Some("banana".into()),
            ..Default::default()
        };
        let err = resolve_placement(&cfg, "flux-dev:q4", &flags).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("--device-vae"), "got: {msg}");
        assert!(msg.contains("banana"), "got: {msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ENV_LOCK;

    #[test]
    fn official_control_selects_ic_lora_for_every_canonical_id() {
        for id in ["union", "motion-track", "pose", "detailer"] {
            assert_eq!(
                resolve_ic_lora_pipeline(None, Some(id), true).unwrap(),
                Some(Ltx2PipelineMode::IcLora)
            );
        }
    }

    #[test]
    fn official_control_requires_video_and_rejects_conflicting_pipeline() {
        assert!(resolve_ic_lora_pipeline(None, Some("union"), false)
            .unwrap_err()
            .to_string()
            .contains("--video"));
        assert!(
            resolve_ic_lora_pipeline(Some(Ltx2PipelineMode::Retake), Some("union"), true)
                .unwrap_err()
                .to_string()
                .contains("conflicts")
        );
    }

    /// Fully explicit config — does NOT use `..Config::default()` which
    /// triggers `default_models_dir()` → reads `MOLD_HOME` env var and
    /// races with concurrent tests that set it.
    fn test_config() -> Config {
        Config {
            config_version: 1,
            default_model: "flux2-klein".to_string(),
            models_dir: "/tmp/mold-test-nonexistent-models".to_string(),
            server_port: 7680,
            default_width: 1024,
            default_height: 1024,
            default_steps: 4,
            embed_metadata: true,
            t5_variant: None,
            qwen3_variant: None,
            output_dir: None,
            media_roots: None,
            default_negative_prompt: None,
            expand: mold_core::ExpandSettings::default(),
            scheduler: Default::default(),
            logging: mold_core::LoggingConfig::default(),
            runpod: mold_core::runpod::RunPodSettings::default(),
            lambda: mold_core::lambda::LambdaSettings::default(),
            gpus: None,
            queue_size: None,
            models: std::collections::HashMap::new(),
        }
    }

    /// Catalog ID short-circuit: when the async pre-pass has already
    /// inserted a synthesized ModelConfig, `resolve_run_args` must surface
    /// the catalog ID verbatim (no manifest tag mangling).
    #[test]
    fn catalog_id_is_used_when_already_in_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut config = test_config();
        config.models.insert(
            "cv:1759168".into(),
            mold_core::ModelConfig {
                family: Some("sdxl".into()),
                ..Default::default()
            },
        );

        let (model, prompt) = resolve_run_args(
            Some("cv:1759168"),
            &["a".to_string(), "cat".to_string()],
            &mut config,
        )
        .unwrap();

        assert_eq!(model, "cv:1759168");
        assert_eq!(prompt.unwrap(), "a cat");
    }

    #[test]
    fn expanded_run_requires_exact_distinct_batch_prompts() {
        let valid = mold_core::ExpandResult {
            original: "source".into(),
            expanded: (1..=100).map(|index| format!("prompt {index}")).collect(),
        };
        validate_run_expansion(&valid.expanded, 100).unwrap();

        let partial = mold_core::ExpandResult {
            original: "source".into(),
            expanded: vec!["only one".into()],
        };
        let error = validate_run_expansion(&partial.expanded, 8).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Expected exactly 8 distinct non-empty prompts"),
            "{error}"
        );

        let duplicate = mold_core::ExpandResult {
            original: "source".into(),
            expanded: vec!["same prompt".into(), "Same, prompt!".into()],
        };
        assert!(validate_run_expansion(&duplicate.expanded, 2).is_err());
    }

    #[test]
    fn catalog_id_errors_when_not_in_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut config = test_config();
        // No async pre-pass ran (or it didn't insert) → the standard
        // "unknown model" path bails with suggestions.
        let err =
            resolve_run_args(Some("cv:9999999"), &["a cat".to_string()], &mut config).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown model 'cv:9999999'"), "got: {msg}");
    }

    #[test]
    fn first_arg_is_model() {
        let mut config = test_config();
        let (model, prompt) = resolve_run_args(
            Some("flux-dev:q4"),
            &["a".to_string(), "cat".to_string()],
            &mut config,
        )
        .unwrap();
        assert_eq!(model, "flux-dev:q4");
        assert_eq!(prompt.unwrap(), "a cat");
    }

    #[test]
    fn model_only_no_prompt() {
        let mut config = test_config();
        let (model, prompt) = resolve_run_args(Some("flux-dev:q4"), &[], &mut config).unwrap();
        assert_eq!(model, "flux-dev:q4");
        assert!(prompt.is_none());
    }

    #[test]
    fn first_arg_is_prompt() {
        // ENV_LOCK: resolved_default_model() reads MOLD_DEFAULT_MODEL and
        // MOLD_MODELS_DIR env vars, which concurrent tests may mutate.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut config = test_config();
        let (model, prompt) = resolve_run_args(
            Some("a"),
            &[
                "sunset".to_string(),
                "over".to_string(),
                "mountains".to_string(),
            ],
            &mut config,
        )
        .unwrap();
        assert_eq!(model, "flux2-klein:q8");
        assert_eq!(prompt.unwrap(), "a sunset over mountains");
    }

    #[test]
    fn single_prompt_word() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut config = test_config();
        let (model, prompt) = resolve_run_args(Some("sunset"), &[], &mut config).unwrap();
        assert_eq!(model, "flux2-klein:q8");
        assert_eq!(prompt.unwrap(), "sunset");
    }

    #[test]
    fn no_args_returns_none_prompt() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut config = test_config();
        let (model, prompt) = resolve_run_args(None, &[], &mut config).unwrap();
        assert_eq!(model, "flux2-klein:q8");
        assert!(prompt.is_none());
    }

    #[test]
    fn bare_model_name_resolves() {
        let mut config = test_config();
        let (model, prompt) =
            resolve_run_args(Some("flux-dev"), &["a turtle".to_string()], &mut config).unwrap();
        assert_eq!(model, "flux-dev:q8");
        assert_eq!(prompt.unwrap(), "a turtle");
    }

    #[test]
    fn sd15_model_name_is_recognized() {
        let mut config = test_config();
        let (model, prompt) = resolve_run_args(
            Some("sd15"),
            &["a".to_string(), "dog".to_string()],
            &mut config,
        )
        .unwrap();
        assert_eq!(model, "sd15:fp16");
        assert_eq!(prompt.unwrap(), "a dog");
    }

    #[test]
    fn dreamshaper_v8_model_is_recognized() {
        let mut config = test_config();
        let (model, prompt) = resolve_run_args(
            Some("dreamshaper-v8"),
            &["photorealistic".to_string()],
            &mut config,
        )
        .unwrap();
        assert_eq!(model, "dreamshaper-v8:fp16");
        assert_eq!(prompt.unwrap(), "photorealistic");
    }

    #[test]
    fn unknown_model_with_known_family_errors() {
        let mut config = test_config();
        let err = resolve_run_args(Some("ultrareal-v8"), &["a cat".to_string()], &mut config)
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown model 'ultrareal-v8'"), "got: {msg}");
        assert!(
            msg.contains("ultrareal-v4"),
            "should suggest ultrareal-v4, got: {msg}"
        );
    }

    #[test]
    fn unknown_model_with_colon_tag_errors() {
        let mut config = test_config();
        let err = resolve_run_args(Some("flux-dev:q99"), &["a cat".to_string()], &mut config)
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown model 'flux-dev:q99'"), "got: {msg}");
    }

    #[test]
    fn natural_language_not_flagged_as_model() {
        let mut config = test_config();
        for word in &["a", "sunset", "photorealistic", "cat", "beautiful"] {
            let result = resolve_run_args(Some(word), &[], &mut config);
            assert!(
                result.is_ok(),
                "'{word}' should not be flagged as a model name"
            );
        }
    }

    #[test]
    fn completions_return_models() {
        let candidates = complete_model_name();
        assert!(!candidates.is_empty());
    }

    // ── validate_file_args tests ──────────────────────────────────────────

    #[test]
    fn validate_no_file_args_passes() {
        assert!(validate_file_args(None, None, None, None, None).is_ok());
    }

    // -- --lora tests --

    #[test]
    fn validate_lora_nonexistent_file() {
        let err = validate_file_args(
            Some("/tmp/mold-test-nonexistent-lora.safetensors"),
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--lora file not found"), "got: {msg}");
    }

    #[test]
    fn validate_lora_directory_instead_of_file() {
        let dir = std::env::temp_dir().join("mold-test-lora-dir");
        std::fs::create_dir_all(&dir).unwrap();
        // Create a .safetensors file inside so it gets suggested
        let adapter = dir.join("adapter.safetensors");
        std::fs::write(&adapter, b"dummy").unwrap();

        let err =
            validate_file_args(Some(dir.to_str().unwrap()), None, None, None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("is a directory"), "got: {msg}");
        assert!(
            msg.contains("adapter.safetensors"),
            "should suggest files, got: {msg}"
        );

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn validate_lora_directory_empty() {
        let dir = std::env::temp_dir().join("mold-test-lora-empty-dir");
        std::fs::create_dir_all(&dir).unwrap();

        let err =
            validate_file_args(Some(dir.to_str().unwrap()), None, None, None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("is a directory"), "got: {msg}");
        assert!(msg.contains("no .safetensors files found"), "got: {msg}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn validate_lora_wrong_extension() {
        let path = std::env::temp_dir().join("mold-test-lora.bin");
        std::fs::write(&path, b"dummy").unwrap();

        let err =
            validate_file_args(Some(path.to_str().unwrap()), None, None, None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(".safetensors"), "got: {msg}");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn validate_lora_valid_file() {
        let path = std::env::temp_dir().join("mold-test-valid-adapter.safetensors");
        std::fs::write(&path, b"dummy").unwrap();

        assert!(validate_file_args(Some(path.to_str().unwrap()), None, None, None, None,).is_ok());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn validate_lora_camera_control_alias() {
        assert!(validate_file_args(Some("camera-control:static"), None, None, None, None).is_ok());
    }

    /// Arg validation is about *files*, not about model compatibility.
    ///
    /// This used to reject a preset whenever the model name contained
    /// "ltx-2.3", which is unsound in both directions: an opaque `cv:` / `hf:`
    /// catalog ID for an LTX-2.3 checkpoint sails through, while a 19B install
    /// reached by such an ID is refused. Compatibility now comes from the
    /// resolved checkpoint artifacts — the server's 422 on the request path,
    /// `resolve_camera_control_preset_path` locally — so a preset name is
    /// simply not a file argument and arg validation must not opine on it.
    #[test]
    fn camera_control_preset_name_is_not_a_file_argument() {
        for model in [
            "ltx-2.3-22b-distilled:fp8",
            "ltx-2-19b-distilled:fp8",
            "cv:2752735",
        ] {
            assert!(
                validate_file_args_full(FileArgRefs {
                    camera_control: Some("dolly-in"),
                    ..FileArgRefs::default()
                })
                .is_ok(),
                "preset names must pass file-arg validation regardless of model ({model})"
            );
        }
    }

    #[test]
    fn camera_control_preset_accepted_on_ltx_2_19b() {
        assert!(validate_file_args_full(FileArgRefs {
            camera_control: Some("dolly-in"),
            ..FileArgRefs::default()
        })
        .is_ok());
    }

    #[test]
    fn camera_control_explicit_path_accepted_on_ltx_2_3() {
        let path = std::env::temp_dir().join("mold-test-ltx23-camera.safetensors");
        std::fs::write(&path, b"dummy").unwrap();
        let result = validate_file_args_full(FileArgRefs {
            camera_control: Some(path.to_str().unwrap()),
            ..FileArgRefs::default()
        });
        std::fs::remove_file(&path).ok();
        assert!(result.is_ok(), "explicit .safetensors should bypass gate");
    }

    // -- --image tests --

    #[test]
    fn validate_image_nonexistent() {
        let err = validate_file_args(
            None,
            Some("/tmp/mold-test-nonexistent-image.png"),
            None,
            None,
            None,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--image file not found"), "got: {msg}");
    }

    #[test]
    fn validate_image_stdin_skips_check() {
        assert!(validate_file_args(None, Some("-"), None, None, None).is_ok());
    }

    #[test]
    fn validate_image_is_directory() {
        let dir = std::env::temp_dir().join("mold-test-image-dir");
        std::fs::create_dir_all(&dir).unwrap();

        let err =
            validate_file_args(None, Some(dir.to_str().unwrap()), None, None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("is a directory"), "got: {msg}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn validate_image_valid_file() {
        let path = std::env::temp_dir().join("mold-test-valid-image.png");
        std::fs::write(&path, b"dummy png").unwrap();

        assert!(validate_file_args(None, Some(path.to_str().unwrap()), None, None, None,).is_ok());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn qwen_image_edit_rejects_stdin_image_arg() {
        let err = validate_image_args_for_model(
            "qwen-image-edit",
            "qwen-image-edit-2511:q4",
            &[String::from("-")],
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not support --image -"));
    }

    #[test]
    fn non_edit_models_reject_multiple_image_args() {
        let err = validate_image_args_for_model(
            "flux",
            "flux-dev:q4",
            &[String::from("one.png"), String::from("two.png")],
        )
        .unwrap_err();
        assert!(err.to_string().contains("multiple --image values"));
    }

    #[test]
    fn qwen_image_edit_accepts_multiple_image_args() {
        assert!(validate_image_args_for_model(
            "qwen-image-edit",
            "qwen-image-edit-2511:q4",
            &[String::from("one.png"), String::from("two.png")]
        )
        .is_ok());
    }

    #[test]
    fn flux2_dev_accepts_up_to_four_ordered_reference_paths() {
        assert!(validate_image_args_for_model(
            "flux2",
            "flux2-dev:bf16",
            &[
                String::from("one.png"),
                String::from("two.png"),
                String::from("three.jpg"),
                String::from("four.jpg"),
            ],
        )
        .is_ok());
        let error = validate_image_args_for_model(
            "flux2",
            "flux2-dev:bf16",
            &vec![String::from("reference.png"); 5],
        )
        .unwrap_err();
        assert!(error.to_string().contains("at most 4"));
    }

    #[test]
    fn flux2_dev_rejects_stdin_reference_arg() {
        let error = validate_image_args_for_model("flux2", "flux2-dev:bf16", &[String::from("-")])
            .unwrap_err();
        assert!(error.to_string().contains("do not support --image -"));
    }

    #[test]
    fn flux_family_accepts_stacked_lora_args() {
        let loras = vec![
            String::from("/tmp/style-a.safetensors"),
            String::from("/tmp/style-b.safetensors"),
        ];

        let (effective_lora, lora_stack) =
            resolve_effective_loras_for_family("flux", &loras, 0.75, None, None).unwrap();

        assert!(effective_lora.is_none());
        let stack = lora_stack.expect("stacked CLI LoRAs should use loras plural");
        assert_eq!(stack.len(), 2);
        assert_eq!(stack[0].path, "/tmp/style-a.safetensors");
        assert_eq!(stack[0].scale, 0.75);
        assert_eq!(stack[1].path, "/tmp/style-b.safetensors");
        assert_eq!(stack[1].scale, 0.75);
    }

    // -- --mask tests --

    #[test]
    fn validate_mask_nonexistent() {
        let err = validate_file_args(
            None,
            None,
            Some("/tmp/mold-test-nonexistent-mask.png"),
            None,
            None,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--mask file not found"), "got: {msg}");
    }

    #[test]
    fn validate_mask_is_directory() {
        let dir = std::env::temp_dir().join("mold-test-mask-dir");
        std::fs::create_dir_all(&dir).unwrap();

        let err =
            validate_file_args(None, None, Some(dir.to_str().unwrap()), None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("is a directory"), "got: {msg}");

        std::fs::remove_dir_all(&dir).ok();
    }

    // -- --control tests --

    #[test]
    fn validate_control_nonexistent() {
        let err = validate_file_args(
            None,
            None,
            None,
            Some("/tmp/mold-test-nonexistent-control.png"),
            None,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--control file not found"), "got: {msg}");
    }

    #[test]
    fn validate_control_is_directory() {
        let dir = std::env::temp_dir().join("mold-test-control-dir");
        std::fs::create_dir_all(&dir).unwrap();

        let err =
            validate_file_args(None, None, None, Some(dir.to_str().unwrap()), None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("is a directory"), "got: {msg}");

        std::fs::remove_dir_all(&dir).ok();
    }

    // -- --output tests --

    #[test]
    fn validate_output_parent_not_exist() {
        let err = validate_file_args(None, None, None, None, Some("/nonexistent/dir/image.png"))
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("output directory does not exist"),
            "got: {msg}"
        );
    }

    #[test]
    fn validate_output_is_directory() {
        let dir = std::env::temp_dir().join("mold-test-output-dir");
        std::fs::create_dir_all(&dir).unwrap();

        let err =
            validate_file_args(None, None, None, None, Some(dir.to_str().unwrap())).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("is a directory"), "got: {msg}");
        assert!(msg.contains("Provide a filename"), "got: {msg}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn validate_output_stdout_passes() {
        assert!(validate_file_args(None, None, None, None, Some("-")).is_ok());
    }

    #[test]
    fn validate_output_valid_path() {
        let dir = std::env::temp_dir();
        let path = dir.join("mold-test-output.png");
        assert!(validate_file_args(None, None, None, None, Some(path.to_str().unwrap()),).is_ok());
    }

    #[test]
    fn validate_output_relative_filename() {
        // Just a filename like "output.png" — parent is "" which is fine
        assert!(validate_file_args(None, None, None, None, Some("output.png")).is_ok());
    }

    // -- combined tests --

    #[test]
    fn validate_multiple_bad_args_fails_on_first() {
        // --lora is checked first, so it should fail on the lora error
        let err = validate_file_args(
            Some("/tmp/mold-test-nonexistent.safetensors"),
            Some("/tmp/mold-test-nonexistent.png"),
            None,
            None,
            None,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("--lora"),
            "should fail on --lora first, got: {msg}"
        );
    }

    // ── expansion deferral logic tests ──────────────────────────────────

    #[test]
    fn defer_expand_to_server_when_not_local() {
        // should_expand && !local → defer_to_server = true
        let should_expand = true;
        let local = false;
        let defer = should_expand && !local;
        assert!(
            defer,
            "expansion should be deferred to server when not local"
        );
    }

    #[test]
    fn expand_locally_when_local_flag_set() {
        // should_expand && local → defer_to_server = false
        let should_expand = true;
        let local = true;
        let defer = should_expand && !local;
        assert!(
            !defer,
            "expansion should NOT be deferred when --local is set"
        );
    }

    #[test]
    fn no_defer_when_expand_disabled() {
        let should_expand = false;
        let local = false;
        let defer = should_expand && !local;
        assert!(!defer, "should not defer when expansion is disabled");
    }

    #[test]
    fn complete_model_name_excludes_upscalers() {
        let candidates = super::complete_model_name();
        let names: Vec<String> = candidates
            .into_iter()
            .map(|c| c.get_value().to_string_lossy().to_string())
            .collect();
        for name in &names {
            assert!(
                !name.starts_with("real-esrgan"),
                "run model completions should not include upscaler '{name}'"
            );
        }
        // Should still have generation models
        assert!(
            !names.is_empty(),
            "should have generation model completions"
        );
    }

    #[test]
    fn complete_model_name_excludes_utility_models() {
        let candidates = super::complete_model_name();
        let names: Vec<String> = candidates
            .into_iter()
            .map(|c| c.get_value().to_string_lossy().to_string())
            .collect();
        for name in &names {
            assert!(
                !name.starts_with("qwen3-expand"),
                "run model completions should not include utility model '{name}'"
            );
        }
    }

    #[test]
    fn guidance_flags_without_any_value_stay_absent() {
        assert!(super::GuidanceFlags::default()
            .into_overrides()
            .unwrap()
            .is_none());
    }

    #[test]
    fn guidance_flags_carry_only_the_flags_that_were_passed() {
        let overrides = super::GuidanceFlags {
            stg_scale: Some(1.5),
            stg_blocks: Some(" 28, 29 ".to_string()),
            ..super::GuidanceFlags::default()
        }
        .into_overrides()
        .unwrap()
        .expect("a passed flag produces overrides");

        assert_eq!(overrides.stg_scale, Some(1.5));
        assert_eq!(overrides.stg_blocks, Some(vec![28, 29]));
        assert_eq!(overrides.rescale_scale, None);
        assert_eq!(overrides.modality_scale, None);
        assert_eq!(overrides.skip_step, None);
    }

    #[test]
    fn guidance_flags_reject_unparsable_block_lists() {
        let err = super::GuidanceFlags {
            stg_blocks: Some("28,twenty-nine".to_string()),
            ..super::GuidanceFlags::default()
        }
        .into_overrides()
        .unwrap_err();
        assert!(err.to_string().contains("--stg-blocks"), "got: {err}");

        let err = super::GuidanceFlags {
            stg_blocks: Some(" , ".to_string()),
            ..super::GuidanceFlags::default()
        }
        .into_overrides()
        .unwrap_err();
        assert!(err.to_string().contains("at least one"), "got: {err}");
    }

    #[test]
    fn prompt_stays_required_without_visual_conditioning() {
        let err = require_prompt(None, "ltx2", false).unwrap_err().to_string();
        assert!(err.contains("no prompt provided"), "got: {err}");
        assert!(require_prompt(None, "flux", true).is_err());
        assert_eq!(
            require_prompt(Some("a turtle".to_string()), "flux", false).unwrap(),
            "a turtle"
        );
    }

    #[test]
    fn prompt_optional_for_conditioned_video_families() {
        // An LTX-2 / LTX-Video request that already carries a source image,
        // keyframes, a source video, or an extend may run unprompted.
        for family in ["ltx2", "ltx-video"] {
            assert_eq!(require_prompt(None, family, true).unwrap(), "");
        }
        // An explicit prompt still wins.
        assert_eq!(
            require_prompt(Some("a turtle".to_string()), "ltx2", true).unwrap(),
            "a turtle"
        );
    }
}
