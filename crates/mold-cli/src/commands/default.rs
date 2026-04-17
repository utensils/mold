use anyhow::Result;
use clap_complete::engine::CompletionCandidate;
use colored::Colorize;
use mold_core::config::{Config, DefaultModelSource};
use mold_core::manifest::{all_generation_model_names, is_known_model, resolve_model_name};

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
    let resolution = config.resolve_default_model();

    let source_label = match resolution.source {
        DefaultModelSource::EnvVar => "MOLD_DEFAULT_MODEL env var",
        DefaultModelSource::ConfigCustomEntry => "config file (custom model entry)",
        DefaultModelSource::Config => "config file",
        DefaultModelSource::LastUsed => "last-used model",
        DefaultModelSource::OnlyDownloaded => "only downloaded model",
        DefaultModelSource::ConfigDefault => "config file (default)",
    };

    println!(
        "{} Default model: {}",
        theme::icon_ok(),
        resolution.model.bold()
    );
    println!(
        "  {} source: {}",
        theme::icon_bullet(),
        source_label.dimmed()
    );

    // Warn if not downloaded.
    if !config.manifest_model_is_downloaded(&resolution.model)
        && config.lookup_model_config(&resolution.model).is_none()
    {
        println!(
            "  {} not downloaded — will auto-pull on first use",
            theme::icon_warn()
        );
    }

    Ok(())
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

    // Load fresh config, update, and save.
    let mut config = Config::load_or_default();
    config.default_model = canonical.clone();
    config.save()?;

    // Check download status against the freshly loaded config.
    let downloaded = config.manifest_model_is_downloaded(&canonical)
        || config.lookup_model_config(&canonical).is_some();

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

/// Completion candidates for the model argument (generation models only).
pub fn complete_model_name() -> Vec<CompletionCandidate> {
    let config = Config::load_or_default();
    all_generation_model_names(&config)
        .into_iter()
        .map(CompletionCandidate::new)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ENV_LOCK;
    use std::collections::HashMap;

    fn empty_config() -> Config {
        Config {
            config_version: 1,
            default_model: "flux2-klein".to_string(),
            models_dir: "/nonexistent".to_string(),
            server_port: 7680,
            default_width: 768,
            default_height: 768,
            default_steps: 4,
            embed_metadata: true,
            t5_variant: None,
            qwen3_variant: None,
            output_dir: None,
            default_negative_prompt: None,
            expand: mold_core::ExpandSettings::default(),
            logging: mold_core::LoggingConfig::default(),
            runpod: mold_core::runpod::RunPodSettings::default(),
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
        let config = empty_config();
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
        let config = empty_config();
        let result = show_default(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn resolve_source_default_config() {
        let mut config = empty_config();
        // Use a model name that won't be downloaded locally, so resolution
        // falls through to the ConfigDefault step.
        config.default_model = "wuerstchen:bf16".to_string();
        let resolution = config.resolve_default_model();
        assert_eq!(resolution.source, DefaultModelSource::ConfigDefault);
    }

    #[test]
    fn resolve_source_custom_model_entry() {
        let mut config = empty_config();
        config.models.insert(
            "flux2-klein".to_string(),
            mold_core::config::ModelConfig::default(),
        );
        let resolution = config.resolve_default_model();
        assert_eq!(resolution.source, DefaultModelSource::ConfigCustomEntry);
    }

    #[test]
    fn complete_model_name_returns_candidates() {
        let candidates = complete_model_name();
        assert!(!candidates.is_empty());
    }

    #[test]
    fn set_default_with_tempdir() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

        // Note: env mutation is process-wide. Uses a unique temp dir name to
        // minimise interference with concurrent tests that read MOLD_HOME.
        let prev_home = std::env::var("MOLD_HOME").ok();
        std::env::set_var("MOLD_HOME", &tmp);

        let config = Config::load_or_default();
        let result = set_default("flux-dev:q4", &config);
        assert!(result.is_ok());

        // Verify the config file was written to the redirected MOLD_HOME.
        let config_path = tmp.join("config.toml");
        let written = std::fs::read_to_string(&config_path).unwrap();
        assert!(written.contains(r#"default_model = "flux-dev:q4""#));

        // Cleanup.
        match prev_home {
            Some(v) => std::env::set_var("MOLD_HOME", v),
            None => std::env::remove_var("MOLD_HOME"),
        }
        let _ = std::fs::remove_dir_all(&tmp);
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
                "default model completions should not include upscaler '{name}'"
            );
        }
    }
}
