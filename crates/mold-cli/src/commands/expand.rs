//! `mold expand` — preview prompt expansion without generating images.

use anyhow::Result;
use colored::Colorize;
use mold_core::{Config, ExpandConfig, ExpandSettings, PromptExpander};

use crate::output::status;
use crate::theme;

pub fn run(
    prompt: &str,
    model: Option<&str>,
    variations: usize,
    json_output: bool,
    backend_override: Option<&str>,
    model_override: Option<&str>,
) -> Result<()> {
    let config = Config::load_or_default();
    let mut expand_settings = config.expand.clone().with_env_overrides();

    // Apply overrides
    if let Some(backend) = backend_override {
        expand_settings.backend = backend.to_string();
    }
    if let Some(m) = model_override {
        if expand_settings.is_local() {
            expand_settings.model = m.to_string();
        } else {
            expand_settings.api_model = m.to_string();
        }
    }

    // Determine model family for prompt style
    let model_family = if let Some(model_name) = model {
        resolve_family(model_name, &config)
    } else {
        "flux".to_string() // Default to FLUX-style prompts
    };

    let expand_config = ExpandConfig {
        model_family: model_family.clone(),
        variations,
        temperature: expand_settings.temperature,
        top_p: expand_settings.top_p,
        max_tokens: expand_settings.max_tokens,
        thinking: expand_settings.thinking,
    };

    // Get expander
    let expander = create_expander(&expand_settings, &config)?;

    if !json_output {
        status!(
            "{} Expanding prompt for {} family...",
            theme::icon_info(),
            model_family.bold()
        );
    }

    let result = expander.expand(prompt, &expand_config)?;

    if json_output {
        // Manual JSON output to avoid serde_json dependency in CLI crate
        let mut json = String::from("[\n");
        for (i, expanded) in result.expanded.iter().enumerate() {
            let escaped = expanded.replace('\\', "\\\\").replace('"', "\\\"");
            json.push_str(&format!("  \"{escaped}\""));
            if i < result.expanded.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push(']');
        println!("{json}");
    } else {
        if variations == 1 {
            println!("{}", result.expanded[0]);
        } else {
            for (i, expanded) in result.expanded.iter().enumerate() {
                status!("{} Variation {}:", theme::icon_ok(), i + 1);
                println!("{expanded}");
                println!();
            }
        }
    }

    Ok(())
}

/// Create the appropriate expander based on settings.
pub(crate) fn create_expander(
    settings: &ExpandSettings,
    config: &Config,
) -> Result<Box<dyn PromptExpander>> {
    if let Some(api_expander) = settings.create_api_expander() {
        return Ok(Box::new(api_expander));
    }

    // Local expander
    #[cfg(feature = "expand")]
    {
        if let Some(local) =
            mold_inference::expand::LocalExpander::from_config(config, Some(&settings.model))
        {
            return Ok(Box::new(local));
        }
        anyhow::bail!(
            "local expand model not found. Pull it with: mold pull qwen3-expand\n\
             Or use an API backend: --expand-backend http://localhost:11434"
        );
    }

    #[cfg(not(feature = "expand"))]
    {
        let _ = config; // suppress unused warning
        anyhow::bail!(
            "local prompt expansion not available — this binary was built without the `expand` feature.\n\
             Use an API backend instead: --expand-backend http://localhost:11434"
        );
    }
}

/// Resolve the model family string from a model name (public for use from run.rs).
pub(crate) fn resolve_family_from_config(model_name: &str, config: &Config) -> String {
    resolve_family(model_name, config)
}

/// Resolve the model family string from a model name.
fn resolve_family(model_name: &str, config: &Config) -> String {
    let model_cfg = config.resolved_model_config(model_name);
    if let Some(family) = model_cfg.family {
        return family;
    }
    if let Some(manifest) = mold_core::manifest::find_manifest(model_name) {
        return manifest.family.clone();
    }
    // Default to flux
    "flux".to_string()
}
