#[cfg(test)]
mod tests {
    use crate::config::{Config, ModelConfig, ModelPaths};
    use crate::manifest::{find_manifest, storage_path};
    use crate::test_support::ENV_LOCK;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn test_models_dir(name: &str) -> PathBuf {
        let unique = format!(
            "mold-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }

    fn populate_manifest_files(root: &std::path::Path, model: &str) {
        let manifest = find_manifest(model).unwrap();
        for file in &manifest.files {
            let path = root.join(storage_path(manifest, file));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, b"test").unwrap();
        }
    }

    // ── Config deserialization ────────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = Config::default();
        assert_eq!(cfg.default_model, "flux2-klein");
        assert_eq!(cfg.server_port, 7680);
        assert_eq!(cfg.default_width, 768);
        assert_eq!(cfg.default_height, 768);
        assert_eq!(cfg.default_steps, 4);
        assert!(cfg.embed_metadata);
        assert!(cfg.models.is_empty());
    }

    #[test]
    fn config_load_or_default_missing_file() {
        let cfg = Config::load_or_default();
        assert!(!cfg.default_model.is_empty());
    }

    #[test]
    fn reload_from_disk_preserving_runtime_keeps_models_dir_override() {
        let cfg = Config {
            models_dir: "/runtime/models".to_string(),
            ..Config::default()
        };

        let reloaded = cfg.reload_from_disk_preserving_runtime();
        assert_eq!(reloaded.models_dir, "/runtime/models");
    }

    #[test]
    fn resolved_model_config_uses_manifest_defaults_and_family() {
        let cfg = Config::default();
        let resolved = cfg.resolved_model_config("flux-schnell:q8");

        assert_eq!(resolved.default_steps, Some(4));
        assert_eq!(resolved.default_guidance, Some(0.0));
        assert_eq!(resolved.default_width, Some(1024));
        assert_eq!(resolved.default_height, Some(1024));
        assert_eq!(resolved.is_schnell, Some(true));
        assert_eq!(resolved.family.as_deref(), Some("flux"));
        assert!(resolved.description.is_some());
    }

    #[test]
    fn config_load_from_toml() {
        let toml = r#"
default_model = "flux-dev"
server_port = 9000
default_width = 896
default_height = 1152
default_steps = 25
embed_metadata = false
"#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.default_model, "flux-dev");
        assert_eq!(cfg.server_port, 9000);
        assert_eq!(cfg.default_width, 896);
        assert_eq!(cfg.default_height, 1152);
        assert_eq!(cfg.default_steps, 25);
        assert!(!cfg.embed_metadata);
    }

    #[test]
    fn config_load_from_toml_partial() {
        let toml = r#"default_model = "flux-dev""#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.default_model, "flux-dev");
        assert_eq!(cfg.server_port, 7680);
        assert_eq!(cfg.default_steps, 4); // default
        assert!(cfg.embed_metadata);
    }

    #[test]
    fn effective_embed_metadata_uses_default_when_unset() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_EMBED_METADATA");
        let cfg = Config::default();
        assert!(cfg.effective_embed_metadata(None));
    }

    #[test]
    fn effective_embed_metadata_uses_config_when_env_missing() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_EMBED_METADATA");
        let cfg = Config {
            embed_metadata: false,
            ..Config::default()
        };
        assert!(!cfg.effective_embed_metadata(None));
    }

    #[test]
    fn effective_embed_metadata_env_overrides_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_EMBED_METADATA", "0");
        let cfg = Config {
            embed_metadata: true,
            ..Config::default()
        };
        assert!(!cfg.effective_embed_metadata(None));
        std::env::remove_var("MOLD_EMBED_METADATA");
    }

    #[test]
    fn effective_embed_metadata_accepts_true_env_values() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_EMBED_METADATA", "true");
        let cfg = Config {
            embed_metadata: false,
            ..Config::default()
        };
        assert!(cfg.effective_embed_metadata(None));
        std::env::remove_var("MOLD_EMBED_METADATA");
    }

    #[test]
    fn effective_embed_metadata_cli_override_wins() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_EMBED_METADATA", "1");
        let cfg = Config {
            embed_metadata: true,
            ..Config::default()
        };
        assert!(!cfg.effective_embed_metadata(Some(false)));
        std::env::remove_var("MOLD_EMBED_METADATA");
    }

    // ── ModelConfig defaults ──────────────────────────────────────────────────

    #[test]
    fn model_config_defaults_from_global() {
        let global = Config::default();
        let mcfg = ModelConfig::default();
        assert_eq!(mcfg.effective_steps(&global), 4);
        assert_eq!(mcfg.effective_width(&global), 768);
        assert_eq!(mcfg.effective_height(&global), 768);
        assert!((mcfg.effective_guidance() - 3.5).abs() < 0.001);
    }

    #[test]
    fn model_config_overrides_global() {
        let global = Config::default();
        let mcfg = ModelConfig {
            default_steps: Some(25),
            default_guidance: Some(0.0),
            default_width: Some(896),
            default_height: Some(1152),
            ..ModelConfig::default()
        };
        assert_eq!(mcfg.effective_steps(&global), 25);
        assert_eq!(mcfg.effective_guidance(), 0.0);
        assert_eq!(mcfg.effective_width(&global), 896);
        assert_eq!(mcfg.effective_height(&global), 1152);
    }

    #[test]
    fn model_config_from_toml() {
        let toml = r#"
[models.ultrareal]
transformer = "/models/ultrareal.safetensors"
vae = "/models/ae.safetensors"
t5_encoder = "/models/t5xxl.safetensors"
clip_encoder = "/models/clip_l.safetensors"
t5_tokenizer = "/models/t5.tokenizer.json"
clip_tokenizer = "/models/clip.tokenizer.json"
default_steps = 25
default_guidance = 3.5
default_width = 896
default_height = 1152
description = "UltraReal - photorealistic"
is_schnell = false
"#;
        let cfg: Config = toml::from_str(toml).unwrap();
        let mcfg = cfg.models.get("ultrareal").unwrap();
        assert_eq!(mcfg.default_steps, Some(25));
        assert_eq!(mcfg.default_width, Some(896));
        assert_eq!(mcfg.is_schnell, Some(false));
        assert_eq!(
            mcfg.description.as_deref(),
            Some("UltraReal - photorealistic")
        );
    }

    // ── ModelPaths via config map (no env vars → deterministic) ──────────────

    fn full_model_config(prefix: &str) -> ModelConfig {
        ModelConfig {
            transformer: Some(format!("{prefix}/transformer.gguf")),
            vae: Some(format!("{prefix}/vae.safetensors")),
            t5_encoder: Some(format!("{prefix}/t5.safetensors")),
            clip_encoder: Some(format!("{prefix}/clip.safetensors")),
            t5_tokenizer: Some(format!("{prefix}/t5.tokenizer.json")),
            clip_tokenizer: Some(format!("{prefix}/clip.tokenizer.json")),
            ..ModelConfig::default()
        }
    }

    #[test]
    fn model_paths_resolve_from_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for var in [
            "MOLD_TRANSFORMER_PATH",
            "MOLD_VAE_PATH",
            "MOLD_T5_PATH",
            "MOLD_CLIP_PATH",
            "MOLD_T5_TOKENIZER_PATH",
            "MOLD_CLIP_TOKENIZER_PATH",
        ] {
            std::env::remove_var(var);
        }
        let mut models = HashMap::new();
        models.insert("test-flux".to_string(), full_model_config("/tmp"));
        let cfg = Config {
            models,
            ..Config::default()
        };
        let paths = ModelPaths::resolve("test-flux", &cfg).unwrap();
        assert_eq!(paths.transformer.to_str().unwrap(), "/tmp/transformer.gguf");
        assert_eq!(paths.vae.to_str().unwrap(), "/tmp/vae.safetensors");
        assert_eq!(
            paths.t5_encoder.as_ref().unwrap().to_str().unwrap(),
            "/tmp/t5.safetensors"
        );
        assert_eq!(
            paths.clip_encoder.as_ref().unwrap().to_str().unwrap(),
            "/tmp/clip.safetensors"
        );
        assert_eq!(
            paths.t5_tokenizer.as_ref().unwrap().to_str().unwrap(),
            "/tmp/t5.tokenizer.json"
        );
        assert_eq!(
            paths.clip_tokenizer.as_ref().unwrap().to_str().unwrap(),
            "/tmp/clip.tokenizer.json"
        );
    }

    #[test]
    fn model_paths_resolve_partial_config_returns_none() {
        // Only transformer is set; other paths are missing → None.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut models = HashMap::new();
        models.insert(
            "partial".to_string(),
            ModelConfig {
                transformer: Some("/tmp/t.gguf".to_string()),
                ..ModelConfig::default()
            },
        );
        let cfg = Config {
            models,
            ..Config::default()
        };
        // Clear all fallback env vars
        for var in [
            "MOLD_VAE_PATH",
            "MOLD_T5_PATH",
            "MOLD_CLIP_PATH",
            "MOLD_T5_TOKENIZER_PATH",
            "MOLD_CLIP_TOKENIZER_PATH",
        ] {
            std::env::remove_var(var);
        }
        assert!(ModelPaths::resolve("partial", &cfg).is_none());
    }

    #[test]
    fn model_paths_resolve_unknown_model_no_env_returns_none() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for var in [
            "MOLD_TRANSFORMER_PATH",
            "MOLD_VAE_PATH",
            "MOLD_T5_PATH",
            "MOLD_CLIP_PATH",
            "MOLD_T5_TOKENIZER_PATH",
            "MOLD_CLIP_TOKENIZER_PATH",
        ] {
            std::env::remove_var(var);
        }
        let paths = ModelPaths::resolve("__no_such_model__", &Config::default());
        assert!(paths.is_none());
    }

    #[test]
    fn model_paths_env_var_fallback() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_TRANSFORMER_PATH", "/env/transformer.gguf");
        std::env::set_var("MOLD_VAE_PATH", "/env/vae.safetensors");
        std::env::set_var("MOLD_T5_PATH", "/env/t5.safetensors");
        std::env::set_var("MOLD_CLIP_PATH", "/env/clip.safetensors");
        std::env::set_var("MOLD_T5_TOKENIZER_PATH", "/env/t5.tokenizer.json");
        std::env::set_var("MOLD_CLIP_TOKENIZER_PATH", "/env/clip.tokenizer.json");

        let paths = ModelPaths::resolve("flux-schnell", &Config::default()).unwrap();
        assert_eq!(paths.transformer.to_str().unwrap(), "/env/transformer.gguf");
        assert_eq!(
            paths.t5_tokenizer.as_ref().unwrap().to_str().unwrap(),
            "/env/t5.tokenizer.json"
        );

        for var in [
            "MOLD_TRANSFORMER_PATH",
            "MOLD_VAE_PATH",
            "MOLD_T5_PATH",
            "MOLD_CLIP_PATH",
            "MOLD_T5_TOKENIZER_PATH",
            "MOLD_CLIP_TOKENIZER_PATH",
        ] {
            std::env::remove_var(var);
        }
    }

    #[test]
    fn model_paths_resolve_manifest_from_models_dir_without_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("resolve-manifest");
        populate_manifest_files(&models_dir, "flux2-klein:q8");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let paths = ModelPaths::resolve("flux2-klein:q8", &Config::default()).unwrap();

        assert!(paths.transformer.starts_with(&models_dir));
        assert!(paths.vae.starts_with(&models_dir));

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
    }

    #[test]
    fn model_paths_models_dir_override_does_not_fall_back_to_stale_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("override-empty");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let mut models = HashMap::new();
        models.insert("flux2-klein:q8".to_string(), full_model_config("/cfg"));
        let cfg = Config {
            models,
            ..Config::default()
        };

        assert!(ModelPaths::resolve("flux2-klein:q8", &cfg).is_none());

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
    }

    #[test]
    fn model_config_prefers_discovered_manifest_paths_from_models_dir() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("model-config-discovery");
        populate_manifest_files(&models_dir, "flux2-klein:q8");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let cfg = Config::default();
        let model_cfg = cfg.model_config("flux2-klein:q8");

        assert!(model_cfg
            .transformer
            .as_deref()
            .is_some_and(|path| path.starts_with(models_dir.to_string_lossy().as_ref())));
        assert!(model_cfg
            .vae
            .as_deref()
            .is_some_and(|path| path.starts_with(models_dir.to_string_lossy().as_ref())));

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
    }

    #[test]
    fn manifest_model_is_downloaded_uses_active_models_dir() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let empty_dir = test_models_dir("manifest-empty");
        std::fs::create_dir_all(&empty_dir).unwrap();
        std::env::set_var("MOLD_MODELS_DIR", &empty_dir);
        let cfg = Config::default();
        assert!(!cfg.manifest_model_is_downloaded("flux-schnell:q8"));
        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(&empty_dir);

        let full_dir = test_models_dir("manifest-full");
        populate_manifest_files(&full_dir, "flux-schnell:q8");
        std::env::set_var("MOLD_MODELS_DIR", &full_dir);
        let cfg = Config::default();
        assert!(cfg.manifest_model_is_downloaded("flux-schnell:q8"));
        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(full_dir);
    }

    #[test]
    fn manifest_model_is_downloaded_respects_component_env_overrides() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_MODELS_DIR");
        std::env::set_var("MOLD_TRANSFORMER_PATH", "/env/transformer.gguf");
        std::env::set_var("MOLD_VAE_PATH", "/env/vae.safetensors");
        std::env::set_var("MOLD_T5_PATH", "/env/t5.safetensors");
        std::env::set_var("MOLD_CLIP_PATH", "/env/clip.safetensors");
        std::env::set_var("MOLD_T5_TOKENIZER_PATH", "/env/t5.tokenizer.json");
        std::env::set_var("MOLD_CLIP_TOKENIZER_PATH", "/env/clip.tokenizer.json");

        let cfg = Config::default();
        assert!(cfg.manifest_model_is_downloaded("flux-schnell:q8"));

        for var in [
            "MOLD_TRANSFORMER_PATH",
            "MOLD_VAE_PATH",
            "MOLD_T5_PATH",
            "MOLD_CLIP_PATH",
            "MOLD_T5_TOKENIZER_PATH",
            "MOLD_CLIP_TOKENIZER_PATH",
        ] {
            std::env::remove_var(var);
        }
    }

    #[test]
    fn model_paths_env_takes_precedence_over_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_TRANSFORMER_PATH", "/env/transformer.gguf");

        let mut models = HashMap::new();
        models.insert("flux-schnell".to_string(), full_model_config("/cfg"));
        let cfg = Config {
            models,
            ..Config::default()
        };
        let paths = ModelPaths::resolve("flux-schnell", &cfg).unwrap();
        assert_eq!(
            paths.transformer.to_str().unwrap(),
            "/env/transformer.gguf",
            "env path should override config path"
        );

        std::env::remove_var("MOLD_TRANSFORMER_PATH");
    }

    #[test]
    fn model_paths_optional_env_takes_precedence_over_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_T5_PATH", "/env/t5.safetensors");
        std::env::set_var("MOLD_CLIP_TOKENIZER_PATH", "/env/clip.tokenizer.json");

        let mut models = HashMap::new();
        models.insert("flux-schnell".to_string(), full_model_config("/cfg"));
        let cfg = Config {
            models,
            ..Config::default()
        };
        let paths = ModelPaths::resolve("flux-schnell", &cfg).unwrap();

        assert_eq!(
            paths.t5_encoder.as_ref().unwrap().to_str().unwrap(),
            "/env/t5.safetensors"
        );
        assert_eq!(
            paths.clip_tokenizer.as_ref().unwrap().to_str().unwrap(),
            "/env/clip.tokenizer.json"
        );

        std::env::remove_var("MOLD_T5_PATH");
        std::env::remove_var("MOLD_CLIP_TOKENIZER_PATH");
    }

    // ── resolved_models_dir ───────────────────────────────────────────────────

    #[test]
    fn resolved_models_dir_from_env() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_MODELS_DIR", "/custom/models");
        let dir = Config::default().resolved_models_dir();
        assert_eq!(dir.to_str().unwrap(), "/custom/models");
        std::env::remove_var("MOLD_MODELS_DIR");
    }

    #[test]
    fn resolved_models_dir_default_expands_tilde() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_MODELS_DIR");
        let dir = Config::default().resolved_models_dir();
        assert!(
            !dir.to_str().unwrap().contains('~'),
            "tilde should be expanded: got {:?}",
            dir
        );
    }

    // ── resolved_output_dir ───────────────────────────────────────────────

    #[test]
    fn resolved_output_dir_none_by_default() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_OUTPUT_DIR");
        let cfg = Config::default();
        assert!(cfg.resolved_output_dir().is_none());
    }

    #[test]
    fn resolved_output_dir_from_env() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_OUTPUT_DIR", "/tmp/mold-output");
        let result = Config::default().resolved_output_dir();
        std::env::remove_var("MOLD_OUTPUT_DIR");
        assert_eq!(result.unwrap(), PathBuf::from("/tmp/mold-output"));
    }

    #[test]
    fn resolved_output_dir_empty_env_returns_none() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_OUTPUT_DIR", "");
        let result = Config::default().resolved_output_dir();
        std::env::remove_var("MOLD_OUTPUT_DIR");
        assert!(result.is_none(), "empty env var should disable output_dir");
    }

    #[test]
    fn resolved_output_dir_from_config_field() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_OUTPUT_DIR");
        let mut cfg = Config::default();
        cfg.output_dir = Some("/srv/images".to_string());
        assert_eq!(
            cfg.resolved_output_dir().unwrap(),
            PathBuf::from("/srv/images")
        );
    }

    #[test]
    fn resolved_output_dir_empty_config_field_returns_none() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_OUTPUT_DIR");
        let mut cfg = Config::default();
        cfg.output_dir = Some(String::new());
        assert!(cfg.resolved_output_dir().is_none());
    }

    #[test]
    fn resolved_output_dir_env_overrides_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_OUTPUT_DIR", "/env/path");
        let mut cfg = Config::default();
        cfg.output_dir = Some("/config/path".to_string());
        let result = cfg.resolved_output_dir();
        std::env::remove_var("MOLD_OUTPUT_DIR");
        assert_eq!(result.unwrap(), PathBuf::from("/env/path"));
    }

    #[test]
    fn resolved_output_dir_expands_tilde() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_OUTPUT_DIR");
        let mut cfg = Config::default();
        cfg.output_dir = Some("~/mold-output".to_string());
        let dir = cfg.resolved_output_dir().unwrap();
        assert!(
            !dir.to_str().unwrap().contains('~'),
            "tilde should be expanded: got {:?}",
            dir
        );
    }

    #[test]
    fn resolved_output_dir_does_not_expand_tilde_in_middle() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_OUTPUT_DIR");
        let mut cfg = Config::default();
        cfg.output_dir = Some("/srv/mold~backup/output".to_string());
        let dir = cfg.resolved_output_dir().unwrap();
        assert_eq!(
            dir,
            PathBuf::from("/srv/mold~backup/output"),
            "tilde in the middle of a path should not be expanded"
        );
    }

    // ── mold_dir / MOLD_HOME ─────────────────────────────────────────────

    #[test]
    fn mold_dir_defaults_to_dot_mold() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_HOME");
        let dir = Config::mold_dir().unwrap();
        assert!(
            dir.ends_with(".mold"),
            "should end with .mold: got {:?}",
            dir
        );
    }

    #[test]
    fn mold_dir_respects_mold_home() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_HOME", "/custom/mold");
        let dir = Config::mold_dir().unwrap();
        std::env::remove_var("MOLD_HOME");
        assert_eq!(dir, PathBuf::from("/custom/mold"));
    }

    #[test]
    fn default_models_dir_respects_mold_home() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_HOME", "/custom/mold");
        std::env::remove_var("MOLD_MODELS_DIR");
        let cfg = Config::default();
        let dir = cfg.resolved_models_dir();
        std::env::remove_var("MOLD_HOME");
        assert_eq!(dir, PathBuf::from("/custom/mold/models"));
    }

    // ── output_dir deserialization ────────────────────────────────────────

    #[test]
    fn config_output_dir_absent_in_toml() {
        let toml = r#"default_model = "flux-schnell""#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert!(cfg.output_dir.is_none());
    }

    #[test]
    fn config_output_dir_present_in_toml() {
        let toml = r#"
            default_model = "flux-schnell"
            output_dir = "/srv/gallery"
        "#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.output_dir.as_deref(), Some("/srv/gallery"));
    }

    // ── resolved_default_model ────────────────────────────────────────────

    /// Config with models_dir pointing to a non-existent path so the smart
    /// default fallback doesn't detect locally downloaded models.
    fn isolated_config() -> Config {
        Config {
            models_dir: "/tmp/mold-test-nonexistent-models".to_string(),
            ..Config::default()
        }
    }

    #[test]
    fn resolved_default_model_returns_config_value() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_DEFAULT_MODEL");
        let cfg = isolated_config();
        assert_eq!(cfg.resolved_default_model(), "flux2-klein");
    }

    #[test]
    fn resolved_default_model_env_overrides_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_DEFAULT_MODEL", "sdxl-turbo:fp16");
        let result = isolated_config().resolved_default_model();
        std::env::remove_var("MOLD_DEFAULT_MODEL");
        assert_eq!(result, "sdxl-turbo:fp16");
    }

    #[test]
    fn resolved_default_model_empty_env_ignored() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_DEFAULT_MODEL", "");
        let result = isolated_config().resolved_default_model();
        std::env::remove_var("MOLD_DEFAULT_MODEL");
        // Empty env should fall through to config value
        assert_eq!(result, "flux2-klein");
    }

    #[test]
    fn resolved_default_model_prefers_config_only_model_over_manifest_fallbacks() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_DEFAULT_MODEL");
        let models_dir = test_models_dir("default-model-config-only");
        populate_manifest_files(&models_dir, "flux2-klein:q8");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let mold_home = std::env::temp_dir().join(format!(
            "mold-home-config-default-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&mold_home).unwrap();
        std::fs::write(mold_home.join("last-model"), "flux-schnell:q8\n").unwrap();
        std::env::set_var("MOLD_HOME", &mold_home);

        let mut models = HashMap::new();
        models.insert(
            "custom-model".to_string(),
            ModelConfig {
                transformer: Some("/models/custom/transformer.safetensors".to_string()),
                vae: Some("/models/custom/vae.safetensors".to_string()),
                ..ModelConfig::default()
            },
        );
        let cfg = Config {
            default_model: "custom-model".to_string(),
            models,
            ..Config::default()
        };

        assert_eq!(cfg.resolved_default_model(), "custom-model");

        std::env::remove_var("MOLD_HOME");
        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(&models_dir);
        let _ = std::fs::remove_dir_all(&mold_home);
    }

    #[test]
    fn resolved_default_model_missing_manifest_falls_back_to_downloaded_last_model() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_DEFAULT_MODEL");
        let models_dir = test_models_dir("default-model-last");
        populate_manifest_files(&models_dir, "sdxl-turbo:fp16");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let mold_home = std::env::temp_dir().join(format!(
            "mold-home-last-model-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&mold_home).unwrap();
        std::fs::write(mold_home.join("last-model"), "sdxl-turbo:fp16\n").unwrap();
        std::env::set_var("MOLD_HOME", &mold_home);

        let cfg = Config {
            default_model: "flux-dev:q4".to_string(),
            ..Config::default()
        };

        assert_eq!(cfg.resolved_default_model(), "sdxl-turbo:fp16");

        std::env::remove_var("MOLD_HOME");
        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(&models_dir);
        let _ = std::fs::remove_dir_all(&mold_home);
    }

    // ── last-model state file ─────────────────────────────────────────────

    #[test]
    fn write_and_read_last_model_roundtrip() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!(
            "mold-last-model-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::env::set_var("MOLD_HOME", &dir);

        Config::write_last_model("flux-dev:q4");
        let result = Config::read_last_model();

        std::env::remove_var("MOLD_HOME");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(result, Some("flux-dev:q4".to_string()));
    }

    #[test]
    fn read_last_model_missing_file_returns_none() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!(
            "mold-no-last-model-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::env::set_var("MOLD_HOME", &dir);

        let result = Config::read_last_model();

        std::env::remove_var("MOLD_HOME");
        assert!(result.is_none());
    }

    #[test]
    fn read_last_model_empty_file_returns_none() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!(
            "mold-empty-last-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("last-model"), "  \n").unwrap();
        std::env::set_var("MOLD_HOME", &dir);

        let result = Config::read_last_model();

        std::env::remove_var("MOLD_HOME");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            result.is_none(),
            "empty/whitespace-only file should be None"
        );
    }

    #[test]
    fn write_last_model_trims_on_read() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!(
            "mold-trim-last-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Simulate a file with trailing newline
        std::fs::write(dir.join("last-model"), "sdxl-turbo:fp16\n").unwrap();
        std::env::set_var("MOLD_HOME", &dir);

        let result = Config::read_last_model();

        std::env::remove_var("MOLD_HOME");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(result, Some("sdxl-turbo:fp16".to_string()));
    }

    // ── effective_output_dir ──────────────────────────────────────────────

    #[test]
    fn effective_output_dir_defaults_to_mold_output() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::remove_var("MOLD_OUTPUT_DIR") };
        unsafe { std::env::remove_var("MOLD_HOME") };
        let cfg = Config::default();
        let dir = cfg.effective_output_dir();
        assert!(
            dir.to_string_lossy().ends_with(".mold/output"),
            "should end with .mold/output: got {:?}",
            dir
        );
    }

    #[test]
    fn effective_output_dir_respects_mold_home() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::remove_var("MOLD_OUTPUT_DIR") };
        unsafe { std::env::set_var("MOLD_HOME", "/custom/mold") };
        let cfg = Config::default();
        let dir = cfg.effective_output_dir();
        unsafe { std::env::remove_var("MOLD_HOME") };
        assert_eq!(dir, PathBuf::from("/custom/mold/output"));
    }

    #[test]
    fn effective_output_dir_env_overrides_default() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("MOLD_OUTPUT_DIR", "/env/output") };
        let cfg = Config::default();
        let dir = cfg.effective_output_dir();
        unsafe { std::env::remove_var("MOLD_OUTPUT_DIR") };
        assert_eq!(dir, PathBuf::from("/env/output"));
    }

    #[test]
    fn effective_output_dir_config_field_overrides_default() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::remove_var("MOLD_OUTPUT_DIR") };
        let mut cfg = Config::default();
        cfg.output_dir = Some("/config/output".to_string());
        assert_eq!(cfg.effective_output_dir(), PathBuf::from("/config/output"));
    }

    #[test]
    fn effective_output_dir_env_beats_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("MOLD_OUTPUT_DIR", "/env/wins") };
        let mut cfg = Config::default();
        cfg.output_dir = Some("/config/loses".to_string());
        let dir = cfg.effective_output_dir();
        unsafe { std::env::remove_var("MOLD_OUTPUT_DIR") };
        assert_eq!(dir, PathBuf::from("/env/wins"));
    }

    // ── resolved_log_dir ─────────────────────────────────────────────────

    #[test]
    fn resolved_log_dir_defaults_to_mold_logs() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::remove_var("MOLD_HOME") };
        let cfg = Config::default();
        let dir = cfg.resolved_log_dir();
        assert!(
            dir.to_string_lossy().ends_with(".mold/logs"),
            "should end with .mold/logs: got {:?}",
            dir
        );
    }

    #[test]
    fn resolved_log_dir_respects_mold_home() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("MOLD_HOME", "/custom/mold") };
        let cfg = Config::default();
        let dir = cfg.resolved_log_dir();
        unsafe { std::env::remove_var("MOLD_HOME") };
        assert_eq!(dir, PathBuf::from("/custom/mold/logs"));
    }

    #[test]
    fn resolved_log_dir_custom_dir_overrides_default() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut cfg = Config::default();
        cfg.logging.dir = Some("/var/log/mold".to_string());
        assert_eq!(cfg.resolved_log_dir(), PathBuf::from("/var/log/mold"));
    }

    #[test]
    fn resolved_log_dir_expands_tilde() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut cfg = Config::default();
        cfg.logging.dir = Some("~/mold-logs".to_string());
        let dir = cfg.resolved_log_dir();
        assert!(
            !dir.to_string_lossy().starts_with('~'),
            "tilde should be expanded: got {:?}",
            dir
        );
        assert!(
            dir.to_string_lossy().ends_with("mold-logs"),
            "should end with mold-logs: got {:?}",
            dir
        );
    }

    // ── logging config deserialization ────────────────────────────────────

    #[test]
    fn logging_config_defaults_when_absent() {
        let toml = r#"default_model = "test""#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.logging.level, "info");
        assert!(!cfg.logging.file);
        assert!(cfg.logging.dir.is_none());
        assert_eq!(cfg.logging.max_days, 7);
    }

    #[test]
    fn logging_config_from_toml() {
        let toml = r#"
            [logging]
            level = "debug"
            file = true
            dir = "/var/log/mold"
            max_days = 30
        "#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.logging.level, "debug");
        assert!(cfg.logging.file);
        assert_eq!(cfg.logging.dir.as_deref(), Some("/var/log/mold"));
        assert_eq!(cfg.logging.max_days, 30);
    }

    #[test]
    fn logging_config_partial_toml() {
        let toml = r#"
            [logging]
            file = true
        "#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.logging.level, "info"); // default
        assert!(cfg.logging.file); // set
        assert!(cfg.logging.dir.is_none()); // default
        assert_eq!(cfg.logging.max_days, 7); // default
    }

    // ── is_output_disabled ───────────────────────────────────────────────

    #[test]
    fn output_not_disabled_by_default() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::remove_var("MOLD_OUTPUT_DIR") };
        let cfg = Config::default();
        assert!(
            !cfg.is_output_disabled(),
            "output should be enabled by default"
        );
    }

    #[test]
    fn output_not_disabled_when_env_set() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("MOLD_OUTPUT_DIR", "/some/path") };
        let cfg = Config::default();
        let disabled = cfg.is_output_disabled();
        unsafe { std::env::remove_var("MOLD_OUTPUT_DIR") };
        assert!(!disabled);
    }

    #[test]
    fn output_disabled_when_env_empty() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("MOLD_OUTPUT_DIR", "") };
        let cfg = Config::default();
        let disabled = cfg.is_output_disabled();
        unsafe { std::env::remove_var("MOLD_OUTPUT_DIR") };
        assert!(disabled, "empty env var should disable output");
    }

    #[test]
    fn output_disabled_when_config_empty_string() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::remove_var("MOLD_OUTPUT_DIR") };
        let mut cfg = Config::default();
        cfg.output_dir = Some(String::new());
        assert!(
            cfg.is_output_disabled(),
            "empty config string should disable output"
        );
    }

    #[test]
    fn output_not_disabled_when_config_has_path() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::remove_var("MOLD_OUTPUT_DIR") };
        let mut cfg = Config::default();
        cfg.output_dir = Some("/srv/images".to_string());
        assert!(!cfg.is_output_disabled());
    }

    #[test]
    fn old_config_without_output_dir_saves_to_default() {
        // Simulate an old config.toml that doesn't mention output_dir at all.
        // Images should still be saved to the default directory.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::remove_var("MOLD_OUTPUT_DIR") };
        let toml = r#"
            default_model = "flux-schnell:q8"
            server_port = 7680
        "#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert!(
            !cfg.is_output_disabled(),
            "old config should not disable output"
        );
        let dir = cfg.effective_output_dir();
        assert!(
            dir.to_string_lossy().ends_with("output"),
            "old config should use default output dir: {:?}",
            dir
        );
    }

    #[test]
    fn old_config_with_metadata_enabled() {
        // Old configs should still embed metadata by default
        let toml = r#"default_model = "flux-schnell:q8""#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert!(
            cfg.effective_embed_metadata(None),
            "metadata should be enabled by default for old configs"
        );
    }

    // ── manifest precedence (#129) ───────────────────────────────────────

    #[test]
    fn manifest_description_always_wins_over_config() {
        // Description and family always come from manifest for known models
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut cfg = Config::default();
        cfg.models.insert(
            "flux-dev:q8".to_string(),
            ModelConfig {
                description: Some("[alpha] old description".to_string()),
                family: Some("wrong-family".to_string()),
                ..ModelConfig::default()
            },
        );
        let resolved = cfg.resolved_model_config("flux-dev:q8");
        let manifest = crate::manifest::find_manifest("flux-dev:q8").unwrap();
        assert_eq!(resolved.description, Some(manifest.description.clone()));
        assert_eq!(resolved.family, Some(manifest.family.clone()));
    }

    #[test]
    fn user_config_overrides_take_precedence_for_defaults() {
        // Explicit user config values override manifest defaults
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut cfg = Config::default();
        cfg.models.insert(
            "flux-schnell:q8".to_string(),
            ModelConfig {
                default_steps: Some(8), // user wants 8 steps
                default_width: Some(512),
                ..ModelConfig::default()
            },
        );
        let resolved = cfg.resolved_model_config("flux-schnell:q8");
        // User values preserved
        assert_eq!(resolved.default_steps, Some(8));
        assert_eq!(resolved.default_width, Some(512));
        // Manifest fills in the rest
        let manifest = crate::manifest::find_manifest("flux-schnell:q8").unwrap();
        assert_eq!(resolved.default_height, Some(manifest.defaults.height));
        assert_eq!(resolved.is_schnell, Some(manifest.defaults.is_schnell));
    }

    #[test]
    fn fresh_pull_gets_manifest_defaults() {
        // After pull, config has no defaults (to_model_config sets None),
        // so manifest fills everything in
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut cfg = Config::default();
        cfg.models.insert(
            "flux-dev:q8".to_string(),
            ModelConfig {
                // Only paths set (simulating a fresh pull)
                transformer: Some("/path/to/transformer.gguf".to_string()),
                vae: Some("/path/to/vae.safetensors".to_string()),
                ..ModelConfig::default()
            },
        );
        let resolved = cfg.resolved_model_config("flux-dev:q8");
        let manifest = crate::manifest::find_manifest("flux-dev:q8").unwrap();
        assert_eq!(resolved.default_steps, Some(manifest.defaults.steps));
        assert_eq!(resolved.default_guidance, Some(manifest.defaults.guidance));
        assert_eq!(resolved.default_width, Some(manifest.defaults.width));
        assert_eq!(resolved.default_height, Some(manifest.defaults.height));
    }

    #[test]
    fn custom_model_keeps_config_values() {
        // Non-manifest models (custom user models) should keep their config values
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut cfg = Config::default();
        cfg.models.insert(
            "my-custom-model".to_string(),
            ModelConfig {
                default_steps: Some(50),
                default_guidance: Some(7.5),
                description: Some("My custom model".to_string()),
                family: Some("flux".to_string()),
                ..ModelConfig::default()
            },
        );
        let resolved = cfg.resolved_model_config("my-custom-model");
        assert_eq!(resolved.default_steps, Some(50));
        assert_eq!(resolved.default_guidance, Some(7.5));
        assert_eq!(resolved.description.as_deref(), Some("My custom model"));
    }

    #[test]
    fn manifest_preserves_user_lora_config() {
        // lora and lora_scale are user overrides — manifest should not touch them
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut cfg = Config::default();
        cfg.models.insert(
            "flux-dev:q8".to_string(),
            ModelConfig {
                lora: Some("/path/to/adapter.safetensors".to_string()),
                lora_scale: Some(0.8),
                ..ModelConfig::default()
            },
        );
        let resolved = cfg.resolved_model_config("flux-dev:q8");
        assert_eq!(
            resolved.lora.as_deref(),
            Some("/path/to/adapter.safetensors")
        );
        assert_eq!(resolved.lora_scale, Some(0.8));
    }

    #[test]
    fn to_model_config_does_not_write_defaults() {
        // to_model_config should only set path fields, not defaults or metadata
        let manifest = crate::manifest::find_manifest("flux-schnell:q8").unwrap();
        let paths = ModelPaths {
            transformer: PathBuf::from("/tmp/transformer.gguf"),
            transformer_shards: Vec::new(),
            vae: PathBuf::from("/tmp/vae.safetensors"),
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };
        let mc = manifest.to_model_config(&paths);
        // Paths should be set
        assert!(mc.transformer.is_some());
        assert!(mc.vae.is_some());
        // Defaults and metadata should NOT be set
        assert!(mc.default_steps.is_none(), "steps should not be in config");
        assert!(
            mc.default_guidance.is_none(),
            "guidance should not be in config"
        );
        assert!(mc.default_width.is_none());
        assert!(mc.default_height.is_none());
        assert!(mc.is_schnell.is_none());
        assert!(mc.scheduler.is_none());
        assert!(
            mc.description.is_none(),
            "description should not be in config"
        );
        assert!(mc.family.is_none(), "family should not be in config");
    }
}
