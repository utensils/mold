use anyhow::Result;
use clap_complete::engine::CompletionCandidate;
use mold_core::manifest::{all_model_names, is_known_model, resolve_model_name};
use mold_core::Config;
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
/// - Else → (config default_model, all args joined as prompt).
/// - Empty prompt → None (error: prompt required).
fn resolve_run_args(
    model_or_prompt: Option<&str>,
    prompt_rest: &[String],
    config: &Config,
) -> (String, Option<String>) {
    if let Some(first) = model_or_prompt {
        if is_known_model(first, config) {
            let prompt = if prompt_rest.is_empty() {
                None
            } else {
                Some(prompt_rest.join(" "))
            };
            return (resolve_model_name(first), prompt);
        }
        // First arg is part of the prompt, not a model
        let mut parts = vec![first.to_string()];
        parts.extend(prompt_rest.iter().cloned());
        let model = resolve_model_name(&config.default_model);
        return (model, Some(parts.join(" ")));
    }

    // No args at all
    (resolve_model_name(&config.default_model), None)
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
    format: String,
    local: bool,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    eager: bool,
) -> Result<()> {
    let config = Config::load_or_default();
    let (model, prompt) = resolve_run_args(model_or_prompt.as_deref(), &prompt_rest, &config);

    // If no prompt from args, try reading from stdin (supports piping)
    let prompt = match prompt {
        Some(p) => Some(p),
        None if !std::io::stdin().is_terminal() => {
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

    generate::run(
        &prompt,
        &model,
        output,
        width,
        height,
        steps,
        guidance,
        seed,
        batch,
        host,
        &format,
        local,
        t5_variant,
        qwen3_variant,
        eager,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        let mut config = Config::default();
        config.default_model = "flux-schnell".to_string();
        config
    }

    #[test]
    fn first_arg_is_model() {
        let config = test_config();
        let (model, prompt) = resolve_run_args(
            Some("flux-dev:q4"),
            &["a".to_string(), "cat".to_string()],
            &config,
        );
        assert_eq!(model, "flux-dev:q4");
        assert_eq!(prompt.unwrap(), "a cat");
    }

    #[test]
    fn model_only_no_prompt() {
        let config = test_config();
        let (model, prompt) = resolve_run_args(Some("flux-dev:q4"), &[], &config);
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
        );
        assert_eq!(model, "flux-schnell:q8");
        assert_eq!(prompt.unwrap(), "a sunset over mountains");
    }

    #[test]
    fn single_prompt_word() {
        let config = test_config();
        let (model, prompt) = resolve_run_args(Some("sunset"), &[], &config);
        assert_eq!(model, "flux-schnell:q8");
        assert_eq!(prompt.unwrap(), "sunset");
    }

    #[test]
    fn no_args_returns_none_prompt() {
        let config = test_config();
        let (model, prompt) = resolve_run_args(None, &[], &config);
        assert_eq!(model, "flux-schnell:q8");
        assert!(prompt.is_none());
    }

    #[test]
    fn bare_model_name_resolves() {
        let config = test_config();
        let (model, prompt) =
            resolve_run_args(Some("flux-dev"), &["a turtle".to_string()], &config);
        assert_eq!(model, "flux-dev:q8");
        assert_eq!(prompt.unwrap(), "a turtle");
    }

    #[test]
    fn sd15_model_name_is_recognized() {
        let config = test_config();
        let (model, prompt) =
            resolve_run_args(Some("sd15"), &["a".to_string(), "dog".to_string()], &config);
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
        );
        assert_eq!(model, "dreamshaper-v8:fp16");
        assert_eq!(prompt.unwrap(), "photorealistic");
    }

    #[test]
    fn completions_return_models() {
        let candidates = complete_model_name();
        assert!(!candidates.is_empty());
    }
}
