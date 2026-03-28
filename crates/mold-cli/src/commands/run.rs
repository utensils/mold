use anyhow::Result;
use clap_complete::engine::CompletionCandidate;
use mold_core::manifest::{
    all_model_names, is_known_model, looks_like_model_name, resolve_model_name,
    suggest_similar_models,
};
use mold_core::{Config, OutputFormat, Scheduler};
use std::io::{IsTerminal, Read};

use super::generate;

/// Provide model name completions for shell tab-completion.
pub fn complete_model_name() -> Vec<CompletionCandidate> {
    let config = Config::load_or_default();
    all_model_names(&config)
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
    host: Option<String>,
    format: OutputFormat,
    no_metadata: bool,
    preview: bool,
    local: bool,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    scheduler: Option<Scheduler>,
    eager: bool,
    image: Option<String>,
    strength: f64,
    mask: Option<String>,
    control: Option<String>,
    control_model: Option<String>,
    control_scale: f64,
    expand: bool,
    no_expand: bool,
    expand_backend: Option<String>,
    expand_model: Option<String>,
) -> Result<()> {
    let config = Config::load_or_default();
    let (model, prompt) = resolve_run_args(model_or_prompt.as_deref(), &prompt_rest, &config)?;

    // Read source image if --image specified
    let source_image = if let Some(ref img_path) = image {
        let bytes = if img_path == "-" {
            // Read binary image from stdin
            let mut buf = Vec::new();
            std::io::stdin().read_to_end(&mut buf)?;
            buf
        } else {
            std::fs::read(img_path)
                .map_err(|e| anyhow::anyhow!("failed to read image '{}': {e}", img_path))?
        };
        Some(bytes)
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

    // If no prompt from args, try reading from stdin (supports piping)
    // When --image - is used, stdin is consumed for the image, so prompt must come from args.
    let prompt = match prompt {
        Some(p) => Some(p),
        None if image.as_deref() != Some("-") && !std::io::stdin().is_terminal() => {
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

    let (final_prompt, original_prompt) = if should_expand {
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
            (expanded.clone(), Some(prompt.clone()))
        } else {
            // Multiple variations: each batch image gets a different prompt.
            // For now, use the first variation as the main prompt.
            // The batch expansion with per-image prompts will be handled
            // in a follow-up (requires changes to generate loop).
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
            // Use first variation for single generation, will expand per-batch later
            (result.expanded[0].clone(), Some(prompt.clone()))
        }
    } else {
        (prompt, None)
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
        host,
        format,
        no_metadata,
        preview,
        local,
        t5_variant,
        qwen3_variant,
        scheduler,
        eager,
        source_image,
        strength,
        mask_image,
        control_image,
        control_model,
        control_scale,
        original_prompt,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config {
            default_model: "flux-schnell".to_string(),
            // Point models_dir to a non-existent path so the smart default
            // fallback doesn't detect locally downloaded models.
            models_dir: "/tmp/mold-test-nonexistent-models".to_string(),
            ..Config::default()
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
        assert_eq!(model, "flux-schnell:q8");
        assert_eq!(prompt.unwrap(), "a sunset over mountains");
    }

    #[test]
    fn single_prompt_word() {
        let config = test_config();
        let (model, prompt) = resolve_run_args(Some("sunset"), &[], &config).unwrap();
        assert_eq!(model, "flux-schnell:q8");
        assert_eq!(prompt.unwrap(), "sunset");
    }

    #[test]
    fn no_args_returns_none_prompt() {
        let config = test_config();
        let (model, prompt) = resolve_run_args(None, &[], &config).unwrap();
        assert_eq!(model, "flux-schnell:q8");
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
}
