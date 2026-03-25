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
        assert_eq!(cfg.default_model, "flux-schnell");
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
        populate_manifest_files(&models_dir, "flux-schnell:q8");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let paths = ModelPaths::resolve("flux-schnell:q8", &Config::default()).unwrap();

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
        models.insert("flux-schnell:q8".to_string(), full_model_config("/cfg"));
        let cfg = Config {
            models,
            ..Config::default()
        };

        assert!(ModelPaths::resolve("flux-schnell:q8", &cfg).is_none());

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
    }

    #[test]
    fn model_config_prefers_discovered_manifest_paths_from_models_dir() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("model-config-discovery");
        populate_manifest_files(&models_dir, "flux-schnell:q8");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let cfg = Config::default();
        let model_cfg = cfg.model_config("flux-schnell:q8");

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
}
