//! `mold expand` — preview prompt expansion without generating images.

use anyhow::Result;
use colored::Colorize;
use mold_core::{Config, ExpandConfig, ExpandSettings, PromptExpander};

use crate::output::status;
use crate::theme;

/// Create a progress callback for the local expand model that prints status
/// messages matching the existing engine progress output style.
#[cfg(feature = "expand")]
fn expand_progress_callback() -> mold_inference::progress::ProgressCallback {
    use mold_inference::progress::ProgressEvent;
    Box::new(move |event: ProgressEvent| match event {
        ProgressEvent::StageDone { name, elapsed } => {
            let secs = elapsed.as_secs_f64();
            status!(
                "  {} {} {}",
                theme::icon_done(),
                name,
                format!("[{:.1}s]", secs).dimmed()
            );
        }
        ProgressEvent::Info { message } => {
            status!("  {} {}", theme::icon_bullet(), message.dimmed());
        }
        _ => {}
    })
}

pub async fn run(
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

    // Get expander (auto-pulls expand model if needed)
    let expander = create_expander(&expand_settings, &config).await?;

    if !json_output {
        status!(
            "{} Expanding prompt for {} family...",
            theme::icon_info(),
            model_family.bold()
        );
    }

    let result = expander.expand(prompt, &expand_config)?;

    if json_output {
        let json =
            serde_json::to_string_pretty(&result.expanded).unwrap_or_else(|_| "[]".to_string());
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
///
/// When the local backend is selected and the expand model hasn't been pulled
/// yet, this will auto-pull it (same pattern as diffusion model auto-pull in
/// `generate.rs`).
pub(crate) async fn create_expander(
    settings: &ExpandSettings,
    config: &Config,
) -> Result<Box<dyn PromptExpander>> {
    if let Some(api_expander) = settings.create_api_expander() {
        return Ok(Box::new(api_expander));
    }

    // Local expander
    #[cfg(feature = "expand")]
    {
        if let Some(mut local) =
            mold_inference::expand::LocalExpander::from_config(config, Some(&settings.model))
        {
            local.set_on_progress(expand_progress_callback());
            return Ok(Box::new(local));
        }

        // Auto-pull: if a manifest exists for the expand model, download it
        let expand_model = &settings.model;
        if let Some(manifest) = mold_core::manifest::find_manifest(expand_model) {
            status!(
                "{} Expand model '{}' not found locally, pulling...",
                theme::icon_info(),
                manifest.name.bold(),
            );
            super::pull::pull_and_configure(
                expand_model,
                &mold_core::download::PullOptions::default(),
            )
            .await?;

            // Reload config after pull and retry
            let updated_config = Config::load_or_default();
            if let Some(mut local) = mold_inference::expand::LocalExpander::from_config(
                &updated_config,
                Some(&settings.model),
            ) {
                local.set_on_progress(expand_progress_callback());
                return Ok(Box::new(local));
            }
        }

        anyhow::bail!(
            "local expand model not found and auto-pull failed.\n\
             Try manually: mold pull qwen3-expand\n\
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
