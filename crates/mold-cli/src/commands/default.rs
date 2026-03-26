use anyhow::Result;
use clap_complete::engine::CompletionCandidate;
use colored::Colorize;
use mold_core::config::Config;
use mold_core::manifest::{all_model_names, is_known_model, resolve_model_name};

use crate::theme;
use crate::AlreadyReported;

/// Show or set the default model.
///
/// With no argument: prints the current default and how it was resolved.
/// With a model argument: validates the name and updates the config file.
pub fn run(model: Option<&str>) -> Result<()> {
    let config = Config::load_or_default();

    match model {
        Some(name) => set_default(name, &config),
        None => show_default(&config),
    }
}

fn show_default(config: &Config) -> Result<()> {
    let resolved = config.resolved_default_model();

    // Determine the source of the resolved default.
    let source = if let Ok(env_val) = std::env::var("MOLD_DEFAULT_MODEL") {
        if !env_val.is_empty() {
            "MOLD_DEFAULT_MODEL env var".dimmed()
        } else {
            resolve_source(config)
        }
    } else {
        resolve_source(config)
    };

    println!("{} Default model: {}", theme::icon_ok(), resolved.bold());
    println!("  {} source: {source}", theme::icon_bullet());

    // Warn if not downloaded.
    if !config.manifest_model_is_downloaded(&resolved)
        && config.lookup_model_config(&resolved).is_none()
    {
        println!(
            "  {} not downloaded — will auto-pull on first use",
            theme::icon_warn()
        );
    }

    Ok(())
}

/// Figure out which fallback step resolved the default.
fn resolve_source(config: &Config) -> colored::ColoredString {
    let configured = &config.default_model;

    // Check if config entry has a custom [models] section.
    if config.lookup_model_config(configured).is_some() {
        return "config file (custom model entry)".dimmed();
    }

    // Check if configured manifest model is downloaded.
    if config.manifest_model_is_downloaded(configured) {
        return "config file".dimmed();
    }

    // Check last-used model.
    if let Some(last) = Config::read_last_model() {
        if config.manifest_model_is_downloaded(&last) {
            return "last-used model".dimmed();
        }
    }

    // Check single downloaded model.
    let downloaded: Vec<_> = mold_core::manifest::known_manifests()
        .iter()
        .filter(|m| config.manifest_model_is_downloaded(&m.name))
        .collect();
    if downloaded.len() == 1 {
        return "only downloaded model".dimmed();
    }

    "config file (default)".dimmed()
}

fn set_default(name: &str, config: &Config) -> Result<()> {
    let canonical = resolve_model_name(name);

    // Validate: must be a known model (manifest or config entry).
    if !is_known_model(&canonical, config) {
        eprintln!("{} Unknown model: {}", theme::icon_fail(), name.bold());
        eprintln!();
        eprintln!("Run {} to see available models.", "mold list".bold());
        return Err(AlreadyReported.into());
    }

    // Warn (not error) if not downloaded yet.
    let downloaded = config.manifest_model_is_downloaded(&canonical)
        || config.lookup_model_config(&canonical).is_some();

    // Check if env var would override what we're about to set.
    if let Ok(env_val) = std::env::var("MOLD_DEFAULT_MODEL") {
        if !env_val.is_empty() && env_val != canonical {
            eprintln!(
                "{} MOLD_DEFAULT_MODEL={} will override this config setting at runtime",
                theme::prefix_warning(),
                env_val.bold(),
            );
        }
    }

    // Load fresh config and update.
    let mut config = Config::load_or_default();
    config.default_model = canonical.clone();
    config.save()?;

    println!(
        "{} Default model set to {}",
        theme::icon_done(),
        canonical.bold()
    );

    if !downloaded {
        println!(
            "  {} not yet downloaded — run {} first",
            theme::icon_warn(),
            format!("mold pull {canonical}").bold(),
        );
    }

    Ok(())
}

/// Completion candidates for the model argument (all known models).
pub fn complete_model_name() -> Vec<CompletionCandidate> {
    let config = Config::load_or_default();
    all_model_names(&config)
        .into_iter()
        .map(CompletionCandidate::new)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn empty_config() -> Config {
        Config {
            default_model: "flux-schnell".to_string(),
            models_dir: "/nonexistent".to_string(),
            server_port: 7680,
            default_width: 768,
            default_height: 768,
            default_steps: 4,
            embed_metadata: true,
            t5_variant: None,
            qwen3_variant: None,
            output_dir: None,
            models: HashMap::new(),
        }
    }

    #[test]
    fn set_default_rejects_unknown_model() {
        let config = empty_config();
        let result = set_default("totally-fake-model-xyz", &config);
        assert!(result.is_err());
    }

    #[test]
    fn set_default_accepts_known_manifest_model() {
        // This will try to save, which requires a valid home dir.
        // We test just the validation part by checking it doesn't reject a known model.
        let config = empty_config();
        // Validate manually — the full set_default would write to disk.
        let canonical = resolve_model_name("flux-schnell");
        assert!(is_known_model(&canonical, &config));
    }

    #[test]
    fn set_default_accepts_bare_name_resolution() {
        let config = empty_config();
        let canonical = resolve_model_name("flux-dev");
        assert!(is_known_model(&canonical, &config));
    }

    #[test]
    fn set_default_accepts_custom_config_model() {
        let mut config = empty_config();
        config.models.insert(
            "my-custom-model".to_string(),
            mold_core::config::ModelConfig::default(),
        );
        assert!(is_known_model("my-custom-model", &config));
    }

    #[test]
    fn show_default_returns_ok() {
        // show_default prints to stdout but should not error.
        let config = empty_config();
        let result = show_default(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn resolve_source_default_config() {
        let config = empty_config();
        let source = resolve_source(&config);
        assert!(source.to_string().contains("default"));
    }

    #[test]
    fn resolve_source_custom_model_entry() {
        let mut config = empty_config();
        config.models.insert(
            "flux-schnell".to_string(),
            mold_core::config::ModelConfig::default(),
        );
        let source = resolve_source(&config);
        assert!(source.to_string().contains("custom model entry"));
    }

    #[test]
    fn complete_model_name_returns_candidates() {
        let candidates = complete_model_name();
        assert!(!candidates.is_empty());
    }

    #[test]
    fn set_default_with_tempdir() {
        // Test the full set_default flow by redirecting MOLD_HOME to a temp dir.
        let tmp = std::env::temp_dir().join(format!(
            "mold-default-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        // Point MOLD_HOME to temp dir so Config::save() writes there.
        let prev_home = std::env::var("MOLD_HOME").ok();
        std::env::set_var("MOLD_HOME", &tmp);

        let config = Config::load_or_default();
        let result = set_default("flux-dev:q4", &config);
        assert!(result.is_ok());

        // Verify the config was written.
        let reloaded = Config::load_or_default();
        assert_eq!(reloaded.default_model, "flux-dev:q4");

        // Cleanup.
        match prev_home {
            Some(v) => std::env::set_var("MOLD_HOME", v),
            None => std::env::remove_var("MOLD_HOME"),
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
