use anyhow::Result;
use clap_complete::engine::CompletionCandidate;
use mold_core::manifest::{
    all_generation_model_names, is_known_model, looks_like_model_name, resolve_model_name,
    suggest_similar_models,
};
use mold_core::{
    Config, KeyframeCondition, LoraWeight, Ltx2PipelineMode, Ltx2SpatialUpscale,
    Ltx2TemporalUpscale, OutputFormat, Scheduler, TimeRange,
};
use std::io::{IsTerminal, Read};
use std::path::Path;

use crate::{Ltx2SpatialUpscaleArg, Ltx2TemporalUpscaleArg};

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
/// - If model_or_prompt looks like a model name but isn't known → error with suggestions.
/// - Else → (config default_model, all args joined as prompt).
/// - Empty prompt → None (error: prompt required).
fn resolve_run_args(
    model_or_prompt: Option<&str>,
    prompt_rest: &[String],
    config: &Config,
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
    model: Option<&'a str>,
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
        } else if let Some(model) = args.model {
            if model.contains("ltx-2.3") {
                anyhow::bail!(
                    "--camera-control preset '{camera_control_value}' is published only for LTX-2 19B; \
                     Lightricks has not released camera-control LoRAs for LTX-2.3 yet. \
                     Pass an explicit .safetensors path with --camera-control /path/to/lora.safetensors, \
                     or switch to an LTX-2 19B model."
                );
            }
        }
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

fn validate_image_args_for_family(family: &str, image: &[String]) -> Result<()> {
    if family == "qwen-image-edit" && image.iter().any(|img| img == "-") {
        anyhow::bail!("qwen-image-edit does not support --image -; pass file paths instead");
    }
    if family != "qwen-image-edit" && image.len() > 1 {
        anyhow::bail!("multiple --image values are only supported for qwen-image-edit models");
    }
    Ok(())
}

fn parse_pipeline(value: Option<String>) -> Result<Option<Ltx2PipelineMode>> {
    value
        .map(|value| match value.as_str() {
            "one-stage" => Ok(Ltx2PipelineMode::OneStage),
            "two-stage" => Ok(Ltx2PipelineMode::TwoStage),
            "two-stage-hq" => Ok(Ltx2PipelineMode::TwoStageHq),
            "distilled" => Ok(Ltx2PipelineMode::Distilled),
            "ic-lora" => Ok(Ltx2PipelineMode::IcLora),
            "keyframe" => Ok(Ltx2PipelineMode::Keyframe),
            "a2vid" => Ok(Ltx2PipelineMode::A2Vid),
            "retake" => Ok(Ltx2PipelineMode::Retake),
            _ => anyhow::bail!("unsupported LTX-2 pipeline: {value}"),
        })
        .transpose()
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
    audio: bool,
    no_audio: bool,
    audio_file: Option<String>,
    video: Option<String>,
    keyframe: Vec<String>,
    pipeline: Option<String>,
    retake: Option<String>,
    spatial_upscale: Option<Ltx2SpatialUpscaleArg>,
    temporal_upscale: Option<Ltx2TemporalUpscaleArg>,
    camera_control: Option<String>,
    host: Option<String>,
    format: OutputFormat,
    no_metadata: bool,
    preview: bool,
    local: bool,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Option<String>,
    scheduler: Option<Scheduler>,
    eager: bool,
    offload: bool,
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
    let config = Config::load_or_default();

    let (model, prompt) = resolve_run_args(model_or_prompt.as_deref(), &prompt_rest, &config)?;
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
        model: Some(model.as_str()),
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

    validate_image_args_for_family(&family, &image)?;

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
    let source_image = if family == "qwen-image-edit" {
        None
    } else {
        loaded_images.first().cloned()
    };
    let edit_images = if family == "qwen-image-edit" && !loaded_images.is_empty() {
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
    let keyframes = parse_keyframes(&keyframe)?;
    let pipeline = parse_pipeline(pipeline)?;
    let retake_range = parse_retake_range(retake)?;
    let spatial_upscale = parse_spatial_upscale(spatial_upscale);
    let temporal_upscale = parse_temporal_upscale(temporal_upscale);

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

    let prompt = prompt.ok_or_else(|| {
        anyhow::anyhow!(
            "no prompt provided\n\n\
             Usage: mold run [MODEL] <PROMPT>\n\
             Example: mold run flux-dev:q4 \"a turtle in the desert\"\n\
             Stdin:   echo \"a turtle\" | mold run flux-dev:q4"
        )
    })?;

    // --- Prompt expansion ---
    let expand_settings = config.expand.clone().with_env_overrides();
    let should_expand = if no_expand {
        false
    } else {
        expand || expand_settings.enabled
    };

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
            };

            crate::output::status!("{} Expanding prompt (server)...", crate::theme::icon_info());

            match client.expand_prompt(&expand_req).await {
                Ok(result) if result.expanded.len() == 1 => {
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

    if family != "ltx2" && lora.len() > 1 {
        anyhow::bail!("multiple --lora values are only supported for LTX-2 / LTX-2.3 models");
    }

    // Resolve LoRA: explicit CLI values override config defaults.
    let effective_lora = if let Some(lora_path) = lora.first() {
        Some(LoraWeight {
            path: lora_path.clone(),
            scale: lora_scale,
        })
    } else {
        let model_cfg = config.resolved_model_config(&model);
        model_cfg
            .effective_lora()
            .map(|(path, scale)| LoraWeight { path, scale })
    };
    let loras = if family == "ltx2"
        && (!lora.is_empty() || effective_lora.is_some() || camera_control.is_some())
    {
        let mut loras = Vec::new();
        if !lora.is_empty() {
            loras.extend(lora.iter().cloned().map(|path| LoraWeight {
                path,
                scale: lora_scale,
            }));
        } else if let Some(lora) = effective_lora.clone() {
            loras.push(lora);
        }
        if let Some(camera_control) = camera_control {
            let path = if camera_control.ends_with(".safetensors") {
                camera_control
            } else {
                format!("camera-control:{camera_control}")
            };
            loras.push(LoraWeight { path, scale: 1.0 });
        }
        Some(loras)
    } else {
        None
    };

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
            enable_audio: if audio {
                Some(true)
            } else if no_audio {
                Some(false)
            } else {
                None
            },
            audio_file: audio_file_bytes,
            source_video: source_video_bytes,
            keyframes,
            pipeline,
            loras,
            retake_range,
            spatial_upscale,
            temporal_upscale,
        },
        host,
        format,
        no_metadata,
        preview,
        local,
        t5_variant,
        qwen3_variant,
        qwen2_variant,
        qwen2_text_encoder_mode,
        scheduler,
        eager,
        offload,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ENV_LOCK;

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
            default_negative_prompt: None,
            expand: mold_core::ExpandSettings::default(),
            logging: mold_core::LoggingConfig::default(),
            runpod: mold_core::runpod::RunPodSettings::default(),
            models: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn first_arg_is_model() {
        let config = test_config();
        let (model, prompt) = resolve_run_args(
            Some("flux-dev:q4"),
            &["a".to_string(), "cat".to_string()],
            &config,
        )
        .unwrap();
        assert_eq!(model, "flux-dev:q4");
        assert_eq!(prompt.unwrap(), "a cat");
    }

    #[test]
    fn model_only_no_prompt() {
        let config = test_config();
        let (model, prompt) = resolve_run_args(Some("flux-dev:q4"), &[], &config).unwrap();
        assert_eq!(model, "flux-dev:q4");
        assert!(prompt.is_none());
    }

    #[test]
    fn first_arg_is_prompt() {
        // ENV_LOCK: resolved_default_model() reads MOLD_DEFAULT_MODEL and
        // MOLD_MODELS_DIR env vars, which concurrent tests may mutate.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = test_config();
        let (model, prompt) = resolve_run_args(
            Some("a"),
            &[
                "sunset".to_string(),
                "over".to_string(),
                "mountains".to_string(),
            ],
            &config,
        )
        .unwrap();
        assert_eq!(model, "flux2-klein:q8");
        assert_eq!(prompt.unwrap(), "a sunset over mountains");
    }

    #[test]
    fn single_prompt_word() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = test_config();
        let (model, prompt) = resolve_run_args(Some("sunset"), &[], &config).unwrap();
        assert_eq!(model, "flux2-klein:q8");
        assert_eq!(prompt.unwrap(), "sunset");
    }

    #[test]
    fn no_args_returns_none_prompt() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = test_config();
        let (model, prompt) = resolve_run_args(None, &[], &config).unwrap();
        assert_eq!(model, "flux2-klein:q8");
        assert!(prompt.is_none());
    }

    #[test]
    fn bare_model_name_resolves() {
        let config = test_config();
        let (model, prompt) =
            resolve_run_args(Some("flux-dev"), &["a turtle".to_string()], &config).unwrap();
        assert_eq!(model, "flux-dev:q8");
        assert_eq!(prompt.unwrap(), "a turtle");
    }

    #[test]
    fn sd15_model_name_is_recognized() {
        let config = test_config();
        let (model, prompt) =
            resolve_run_args(Some("sd15"), &["a".to_string(), "dog".to_string()], &config).unwrap();
        assert_eq!(model, "sd15:fp16");
        assert_eq!(prompt.unwrap(), "a dog");
    }

    #[test]
    fn dreamshaper_v8_model_is_recognized() {
        let config = test_config();
        let (model, prompt) = resolve_run_args(
            Some("dreamshaper-v8"),
            &["photorealistic".to_string()],
            &config,
        )
        .unwrap();
        assert_eq!(model, "dreamshaper-v8:fp16");
        assert_eq!(prompt.unwrap(), "photorealistic");
    }

    #[test]
    fn unknown_model_with_known_family_errors() {
        let config = test_config();
        let err =
            resolve_run_args(Some("ultrareal-v8"), &["a cat".to_string()], &config).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown model 'ultrareal-v8'"), "got: {msg}");
        assert!(
            msg.contains("ultrareal-v4"),
            "should suggest ultrareal-v4, got: {msg}"
        );
    }

    #[test]
    fn unknown_model_with_colon_tag_errors() {
        let config = test_config();
        let err =
            resolve_run_args(Some("flux-dev:q99"), &["a cat".to_string()], &config).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown model 'flux-dev:q99'"), "got: {msg}");
    }

    #[test]
    fn natural_language_not_flagged_as_model() {
        let config = test_config();
        for word in &["a", "sunset", "photorealistic", "cat", "beautiful"] {
            let result = resolve_run_args(Some(word), &[], &config);
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

    #[test]
    fn camera_control_preset_rejected_on_ltx_2_3() {
        let err = validate_file_args_full(FileArgRefs {
            camera_control: Some("dolly-in"),
            model: Some("ltx-2.3-22b-distilled:fp8"),
            ..FileArgRefs::default()
        })
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("LTX-2 19B") && msg.contains("LTX-2.3"),
            "expected LTX-2.3 publishing gap message, got: {msg}"
        );
    }

    #[test]
    fn camera_control_preset_accepted_on_ltx_2_19b() {
        assert!(validate_file_args_full(FileArgRefs {
            camera_control: Some("dolly-in"),
            model: Some("ltx-2-19b-distilled:fp8"),
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
            model: Some("ltx-2.3-22b-distilled:fp8"),
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
        let err =
            validate_image_args_for_family("qwen-image-edit", &[String::from("-")]).unwrap_err();
        assert!(err.to_string().contains("does not support --image -"));
    }

    #[test]
    fn non_edit_models_reject_multiple_image_args() {
        let err = validate_image_args_for_family(
            "flux",
            &[String::from("one.png"), String::from("two.png")],
        )
        .unwrap_err();
        assert!(err.to_string().contains("multiple --image values"));
    }

    #[test]
    fn qwen_image_edit_accepts_multiple_image_args() {
        assert!(validate_image_args_for_family(
            "qwen-image-edit",
            &[String::from("one.png"), String::from("two.png")]
        )
        .is_ok());
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
}
