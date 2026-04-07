#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]

    use crate::config::{Config, ModelConfig, ModelPaths};
    use crate::manifest::{find_manifest, known_manifests, storage_path};
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

    fn create_pulling_marker(root: &std::path::Path, model: &str) {
        let path = root.join(crate::download::pulling_marker_rel_path(model));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, b"pulling").unwrap();
    }

    // ── Config deserialization ────────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = Config::default();
        assert_eq!(cfg.default_model, "flux2-klein:q8");
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
        // Bare name gets resolved to flux2-klein:q8 via manifest resolution
        assert_eq!(cfg.resolved_default_model(), "flux2-klein:q8");
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
        // Empty env should fall through to config value (resolved bare name)
        assert_eq!(result, "flux2-klein:q8");
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
            spatial_upscaler: None,
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

    // ── config versioning / migrations ───────────────────────────────────

    #[test]
    fn old_config_without_version_defaults_to_zero() {
        let toml = r#"default_model = "flux-schnell:q8""#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.config_version, 0);
    }

    #[test]
    fn new_config_has_current_version() {
        let cfg = Config::default();
        assert!(
            cfg.config_version > 0,
            "default config should have current version"
        );
    }

    #[test]
    fn migrate_v0_strips_stale_manifest_defaults() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut cfg = Config {
            config_version: 0,
            ..Config::default()
        };
        // Simulate old stale config with manifest defaults baked in
        cfg.models.insert(
            "flux-dev:q8".to_string(),
            ModelConfig {
                transformer: Some("/path/to/transformer.gguf".to_string()),
                vae: Some("/path/to/vae.safetensors".to_string()),
                default_steps: Some(999),
                default_guidance: Some(0.0),
                default_width: Some(512),
                default_height: Some(512),
                description: Some("[alpha] stale desc".to_string()),
                family: Some("flux".to_string()),
                is_schnell: Some(false),
                ..ModelConfig::default()
            },
        );
        // Also add a custom model that should NOT be touched
        cfg.models.insert(
            "my-custom-model".to_string(),
            ModelConfig {
                default_steps: Some(50),
                description: Some("My model".to_string()),
                ..ModelConfig::default()
            },
        );

        Config::run_migrations(&mut cfg);

        // Manifest model should have defaults stripped
        let flux = cfg.models.get("flux-dev:q8").unwrap();
        assert!(
            flux.default_steps.is_none(),
            "stale steps should be cleared"
        );
        assert!(
            flux.default_guidance.is_none(),
            "stale guidance should be cleared"
        );
        assert!(
            flux.description.is_none(),
            "stale description should be cleared"
        );
        assert!(flux.family.is_none(), "stale family should be cleared");
        // Paths should be preserved
        assert!(flux.transformer.is_some(), "paths should survive migration");
        assert!(flux.vae.is_some(), "paths should survive migration");

        // Custom model should be untouched
        let custom = cfg.models.get("my-custom-model").unwrap();
        assert_eq!(custom.default_steps, Some(50));
        assert_eq!(custom.description.as_deref(), Some("My model"));
    }

    #[test]
    fn migration_preserves_user_lora() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut cfg = Config {
            config_version: 0,
            ..Config::default()
        };
        cfg.models.insert(
            "flux-dev:q8".to_string(),
            ModelConfig {
                lora: Some("/path/to/adapter.safetensors".to_string()),
                lora_scale: Some(0.8),
                default_steps: Some(999), // stale
                ..ModelConfig::default()
            },
        );

        Config::run_migrations(&mut cfg);

        let flux = cfg.models.get("flux-dev:q8").unwrap();
        assert_eq!(flux.lora.as_deref(), Some("/path/to/adapter.safetensors"));
        assert_eq!(flux.lora_scale, Some(0.8));
        assert!(flux.default_steps.is_none());
    }

    #[test]
    fn config_version_serializes_to_toml() {
        let cfg = Config::default();
        let toml = toml::to_string_pretty(&cfg).unwrap();
        assert!(
            toml.contains("config_version"),
            "config_version should be in serialized TOML"
        );
    }

    // ── resolve_default_model source tracking ───────────────────────────

    #[test]
    fn resolve_default_model_env_returns_env_var_source() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_DEFAULT_MODEL", "sdxl-turbo:fp16");
        let resolution = isolated_config().resolve_default_model();
        std::env::remove_var("MOLD_DEFAULT_MODEL");
        assert_eq!(resolution.model, "sdxl-turbo:fp16");
        assert_eq!(resolution.source, crate::config::DefaultModelSource::EnvVar);
    }

    #[test]
    fn resolve_default_model_custom_entry_returns_config_custom_entry_source() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_DEFAULT_MODEL");
        let mut models = HashMap::new();
        models.insert(
            "custom-model".to_string(),
            ModelConfig {
                transformer: Some("/models/t.safetensors".to_string()),
                vae: Some("/models/v.safetensors".to_string()),
                ..ModelConfig::default()
            },
        );
        let cfg = Config {
            default_model: "custom-model".to_string(),
            models,
            models_dir: "/tmp/mold-test-nonexistent-models".to_string(),
            ..Config::default()
        };
        let resolution = cfg.resolve_default_model();
        assert_eq!(resolution.model, "custom-model");
        assert_eq!(
            resolution.source,
            crate::config::DefaultModelSource::ConfigCustomEntry
        );
    }

    #[test]
    fn resolve_default_model_downloaded_manifest_returns_config_source() {
        // Step 3: config default_model is a manifest model that is downloaded
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_DEFAULT_MODEL");
        let models_dir = test_models_dir("resolve-step3");
        populate_manifest_files(&models_dir, "flux-schnell:q8");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let cfg = Config {
            default_model: "flux-schnell:q8".to_string(),
            models_dir: "/tmp/mold-test-nonexistent-models".to_string(),
            ..Config::default()
        };
        let resolution = cfg.resolve_default_model();

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(&models_dir);
        assert_eq!(resolution.model, "flux-schnell:q8");
        assert_eq!(resolution.source, crate::config::DefaultModelSource::Config);
    }

    #[test]
    fn resolve_default_model_last_used_returns_last_used_source() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_DEFAULT_MODEL");
        let models_dir = test_models_dir("resolve-step4");
        populate_manifest_files(&models_dir, "sdxl-turbo:fp16");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let mold_home = test_models_dir("mold-home-step4");
        std::fs::create_dir_all(&mold_home).unwrap();
        std::fs::write(mold_home.join("last-model"), "sdxl-turbo:fp16").unwrap();
        std::env::set_var("MOLD_HOME", &mold_home);

        // default_model points to something not downloaded
        let cfg = Config {
            default_model: "flux-dev:q4".to_string(),
            models_dir: "/tmp/mold-test-nonexistent-models".to_string(),
            ..Config::default()
        };
        let resolution = cfg.resolve_default_model();

        std::env::remove_var("MOLD_HOME");
        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(&models_dir);
        let _ = std::fs::remove_dir_all(&mold_home);
        assert_eq!(resolution.model, "sdxl-turbo:fp16");
        assert_eq!(
            resolution.source,
            crate::config::DefaultModelSource::LastUsed
        );
    }

    #[test]
    fn resolve_default_model_single_downloaded_returns_only_downloaded_source() {
        // Step 5: exactly one non-utility model is downloaded
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_DEFAULT_MODEL");
        let models_dir = test_models_dir("resolve-step5");
        populate_manifest_files(&models_dir, "sdxl-turbo:fp16");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        // No last-model file
        let mold_home = test_models_dir("mold-home-step5");
        std::fs::create_dir_all(&mold_home).unwrap();
        std::env::set_var("MOLD_HOME", &mold_home);

        // default_model points to something not downloaded and no custom entry
        let cfg = Config {
            default_model: "flux-dev:q4".to_string(),
            models_dir: "/tmp/mold-test-nonexistent-models".to_string(),
            ..Config::default()
        };
        let resolution = cfg.resolve_default_model();

        std::env::remove_var("MOLD_HOME");
        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(&models_dir);
        let _ = std::fs::remove_dir_all(&mold_home);
        assert_eq!(resolution.model, "sdxl-turbo:fp16");
        assert_eq!(
            resolution.source,
            crate::config::DefaultModelSource::OnlyDownloaded
        );
    }

    #[test]
    fn resolve_default_model_fallback_returns_config_default_source() {
        // Step 6: nothing downloaded, no last-model → config default
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_DEFAULT_MODEL");
        let mold_home = test_models_dir("mold-home-step6");
        std::fs::create_dir_all(&mold_home).unwrap();
        std::env::set_var("MOLD_HOME", &mold_home);

        let cfg = isolated_config();
        let resolution = cfg.resolve_default_model();

        std::env::remove_var("MOLD_HOME");
        let _ = std::fs::remove_dir_all(&mold_home);
        // Bare name resolved via manifest
        assert_eq!(resolution.model, "flux2-klein:q8");
        assert_eq!(
            resolution.source,
            crate::config::DefaultModelSource::ConfigDefault
        );
    }

    // ── load_or_default error recovery ──────────────────────────────────

    #[test]
    fn load_or_default_invalid_toml_returns_default() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mold_home = test_models_dir("invalid-toml");
        std::fs::create_dir_all(&mold_home).unwrap();
        std::fs::write(mold_home.join("config.toml"), "{{{{not valid toml!").unwrap();
        std::env::set_var("MOLD_HOME", &mold_home);

        let cfg = Config::load_or_default();

        std::env::remove_var("MOLD_HOME");
        let _ = std::fs::remove_dir_all(&mold_home);
        // Should fall back to defaults
        assert_eq!(cfg.default_model, "flux2-klein:q8");
        assert_eq!(cfg.server_port, 7680);
    }

    #[test]
    fn load_or_default_valid_toml_with_extra_fields_tolerates_unknown_keys() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mold_home = test_models_dir("extra-fields");
        std::fs::create_dir_all(&mold_home).unwrap();
        // TOML with a field that doesn't exist in the Config struct
        let content = r#"
default_model = "flux-dev"
server_port = 8888
some_future_field = "should be ignored"
"#;
        std::fs::write(mold_home.join("config.toml"), content).unwrap();
        std::env::set_var("MOLD_HOME", &mold_home);

        let cfg = Config::load_or_default();

        std::env::remove_var("MOLD_HOME");
        let _ = std::fs::remove_dir_all(&mold_home);
        // serde(deny_unknown_fields) is not set, so unknown keys are silently ignored
        // and the known fields must be parsed correctly (not fall back to defaults).
        assert_eq!(cfg.default_model, "flux-dev");
        assert_eq!(cfg.server_port, 8888);
    }

    #[test]
    fn load_or_default_triggers_migration_on_v0_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mold_home = test_models_dir("migration-trigger");
        std::fs::create_dir_all(&mold_home).unwrap();
        // v0 config (no config_version field)
        let content = r#"
default_model = "flux-schnell:q8"

[models."flux-dev:q8"]
transformer = "/path/to/transformer.gguf"
vae = "/path/to/vae.safetensors"
default_steps = 999
description = "stale"
"#;
        std::fs::write(mold_home.join("config.toml"), content).unwrap();
        std::env::set_var("MOLD_HOME", &mold_home);

        let cfg = Config::load_or_default();

        std::env::remove_var("MOLD_HOME");

        // Migration should have run: version bumped, stale defaults cleared
        assert!(cfg.config_version >= 1);
        if let Some(flux) = cfg.models.get("flux-dev:q8") {
            assert!(
                flux.default_steps.is_none(),
                "migration should clear stale steps"
            );
            assert!(
                flux.description.is_none(),
                "migration should clear stale description"
            );
            // Paths should survive
            assert!(flux.transformer.is_some());
        }
        let _ = std::fs::remove_dir_all(&mold_home);
    }

    // ── ModelPaths SDXL and multi-shard fields ──────────────────────────

    #[test]
    fn model_paths_resolve_sdxl_fields() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for var in [
            "MOLD_TRANSFORMER_PATH",
            "MOLD_VAE_PATH",
            "MOLD_T5_PATH",
            "MOLD_CLIP_PATH",
            "MOLD_T5_TOKENIZER_PATH",
            "MOLD_CLIP_TOKENIZER_PATH",
            "MOLD_CLIP2_PATH",
            "MOLD_CLIP2_TOKENIZER_PATH",
            "MOLD_TEXT_TOKENIZER_PATH",
            "MOLD_DECODER_PATH",
        ] {
            std::env::remove_var(var);
        }
        let mut models = HashMap::new();
        models.insert(
            "sdxl-test".to_string(),
            ModelConfig {
                transformer: Some("/tmp/sdxl/unet.safetensors".to_string()),
                vae: Some("/tmp/sdxl/vae.safetensors".to_string()),
                clip_encoder: Some("/tmp/sdxl/clip_l.safetensors".to_string()),
                clip_tokenizer: Some("/tmp/sdxl/clip_l.tokenizer.json".to_string()),
                clip_encoder_2: Some("/tmp/sdxl/clip_g.safetensors".to_string()),
                clip_tokenizer_2: Some("/tmp/sdxl/clip_g.tokenizer.json".to_string()),
                ..ModelConfig::default()
            },
        );
        let cfg = Config {
            models,
            ..Config::default()
        };
        let paths = ModelPaths::resolve("sdxl-test", &cfg).unwrap();
        assert_eq!(
            paths.clip_encoder_2.as_ref().unwrap().to_str().unwrap(),
            "/tmp/sdxl/clip_g.safetensors"
        );
        assert_eq!(
            paths.clip_tokenizer_2.as_ref().unwrap().to_str().unwrap(),
            "/tmp/sdxl/clip_g.tokenizer.json"
        );
    }

    #[test]
    fn model_paths_resolve_transformer_shards() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for var in [
            "MOLD_TRANSFORMER_PATH",
            "MOLD_VAE_PATH",
            "MOLD_T5_PATH",
            "MOLD_CLIP_PATH",
            "MOLD_T5_TOKENIZER_PATH",
            "MOLD_CLIP_TOKENIZER_PATH",
            "MOLD_CLIP2_PATH",
            "MOLD_CLIP2_TOKENIZER_PATH",
            "MOLD_TEXT_TOKENIZER_PATH",
            "MOLD_DECODER_PATH",
        ] {
            std::env::remove_var(var);
        }
        let mut models = HashMap::new();
        models.insert(
            "sharded-model".to_string(),
            ModelConfig {
                transformer: Some("/tmp/shard/main.safetensors".to_string()),
                transformer_shards: Some(vec![
                    "/tmp/shard/shard-0.safetensors".to_string(),
                    "/tmp/shard/shard-1.safetensors".to_string(),
                ]),
                vae: Some("/tmp/shard/vae.safetensors".to_string()),
                ..ModelConfig::default()
            },
        );
        let cfg = Config {
            models,
            ..Config::default()
        };
        let paths = ModelPaths::resolve("sharded-model", &cfg).unwrap();
        assert_eq!(paths.transformer_shards.len(), 2);
        assert_eq!(
            paths.transformer_shards[0].to_str().unwrap(),
            "/tmp/shard/shard-0.safetensors"
        );
        assert_eq!(
            paths.transformer_shards[1].to_str().unwrap(),
            "/tmp/shard/shard-1.safetensors"
        );
    }

    #[test]
    fn model_paths_resolve_text_encoder_files_and_decoder() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for var in [
            "MOLD_TRANSFORMER_PATH",
            "MOLD_VAE_PATH",
            "MOLD_T5_PATH",
            "MOLD_CLIP_PATH",
            "MOLD_T5_TOKENIZER_PATH",
            "MOLD_CLIP_TOKENIZER_PATH",
            "MOLD_CLIP2_PATH",
            "MOLD_CLIP2_TOKENIZER_PATH",
            "MOLD_TEXT_TOKENIZER_PATH",
            "MOLD_DECODER_PATH",
        ] {
            std::env::remove_var(var);
        }
        let mut models = HashMap::new();
        models.insert(
            "wurst-test".to_string(),
            ModelConfig {
                transformer: Some("/tmp/wurst/prior.safetensors".to_string()),
                vae: Some("/tmp/wurst/vqgan.safetensors".to_string()),
                decoder: Some("/tmp/wurst/decoder.safetensors".to_string()),
                text_encoder_files: Some(vec![
                    "/tmp/wurst/enc-shard-0.safetensors".to_string(),
                    "/tmp/wurst/enc-shard-1.safetensors".to_string(),
                ]),
                text_tokenizer: Some("/tmp/wurst/tokenizer.json".to_string()),
                ..ModelConfig::default()
            },
        );
        let cfg = Config {
            models,
            ..Config::default()
        };
        let paths = ModelPaths::resolve("wurst-test", &cfg).unwrap();
        assert_eq!(
            paths.decoder.as_ref().unwrap().to_str().unwrap(),
            "/tmp/wurst/decoder.safetensors"
        );
        assert_eq!(paths.text_encoder_files.len(), 2);
        assert_eq!(
            paths.text_tokenizer.as_ref().unwrap().to_str().unwrap(),
            "/tmp/wurst/tokenizer.json"
        );
    }

    #[test]
    fn model_paths_env_var_for_decoder() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("MOLD_TRANSFORMER_PATH", "/env/prior.safetensors");
        std::env::set_var("MOLD_VAE_PATH", "/env/vae.safetensors");
        std::env::set_var("MOLD_DECODER_PATH", "/env/decoder.safetensors");

        let paths = ModelPaths::resolve("__no_such__", &Config::default()).unwrap();
        assert_eq!(
            paths.decoder.as_ref().unwrap().to_str().unwrap(),
            "/env/decoder.safetensors"
        );

        std::env::remove_var("MOLD_TRANSFORMER_PATH");
        std::env::remove_var("MOLD_VAE_PATH");
        std::env::remove_var("MOLD_DECODER_PATH");
    }

    // ── all_file_paths ──────────────────────────────────────────────────

    #[test]
    fn all_file_paths_collects_all_fields() {
        let mc = ModelConfig {
            transformer: Some("/a/transformer.gguf".to_string()),
            vae: Some("/a/vae.safetensors".to_string()),
            t5_encoder: Some("/a/t5.safetensors".to_string()),
            clip_encoder: Some("/a/clip.safetensors".to_string()),
            t5_tokenizer: Some("/a/t5.tokenizer.json".to_string()),
            clip_tokenizer: Some("/a/clip.tokenizer.json".to_string()),
            clip_encoder_2: Some("/a/clip_g.safetensors".to_string()),
            clip_tokenizer_2: Some("/a/clip_g.tokenizer.json".to_string()),
            text_tokenizer: Some("/a/text.tokenizer.json".to_string()),
            decoder: Some("/a/decoder.safetensors".to_string()),
            transformer_shards: Some(vec![
                "/a/shard0.safetensors".to_string(),
                "/a/shard1.safetensors".to_string(),
            ]),
            text_encoder_files: Some(vec!["/a/enc0.safetensors".to_string()]),
            ..ModelConfig::default()
        };
        let paths = mc.all_file_paths();
        // 10 single fields + 2 transformer shards + 1 text encoder file = 13
        assert_eq!(paths.len(), 13);
        assert!(paths.contains(&"/a/transformer.gguf".to_string()));
        assert!(paths.contains(&"/a/clip_g.safetensors".to_string()));
        assert!(paths.contains(&"/a/shard0.safetensors".to_string()));
        assert!(paths.contains(&"/a/enc0.safetensors".to_string()));
        assert!(paths.contains(&"/a/decoder.safetensors".to_string()));
    }

    #[test]
    fn all_file_paths_skips_none_fields() {
        let mc = ModelConfig {
            transformer: Some("/a/transformer.gguf".to_string()),
            vae: Some("/a/vae.safetensors".to_string()),
            ..ModelConfig::default()
        };
        let paths = mc.all_file_paths();
        assert_eq!(paths.len(), 2);
    }

    // ── disk_usage ──────────────────────────────────────────────────────

    #[test]
    fn disk_usage_sums_real_files() {
        let dir = test_models_dir("disk-usage");
        std::fs::create_dir_all(&dir).unwrap();
        // Create two files of known sizes
        let file_a = dir.join("a.safetensors");
        let file_b = dir.join("b.safetensors");
        std::fs::write(&file_a, [0u8; 1024]).unwrap(); // 1 KiB
        std::fs::write(&file_b, [0u8; 2048]).unwrap(); // 2 KiB

        let mc = ModelConfig {
            transformer: Some(file_a.to_str().unwrap().to_string()),
            vae: Some(file_b.to_str().unwrap().to_string()),
            ..ModelConfig::default()
        };
        let (bytes, gb) = mc.disk_usage();
        assert_eq!(bytes, 3072);
        assert!((gb - 3072.0 / 1_073_741_824.0).abs() < 1e-12);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_usage_skips_missing_files() {
        let mc = ModelConfig {
            transformer: Some("/nonexistent/path/t.safetensors".to_string()),
            vae: Some("/nonexistent/path/v.safetensors".to_string()),
            ..ModelConfig::default()
        };
        let (bytes, gb) = mc.disk_usage();
        assert_eq!(bytes, 0);
        assert_eq!(gb, 0.0);
    }

    // ── Config::save round-trip ─────────────────────────────────────────

    #[test]
    fn config_save_and_reload_round_trip() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mold_home = test_models_dir("save-roundtrip");
        std::fs::create_dir_all(&mold_home).unwrap();
        std::env::set_var("MOLD_HOME", &mold_home);

        let mut cfg = Config::default();
        cfg.default_model = "sd15:fp16".to_string();
        cfg.server_port = 9999;
        cfg.default_width = 512;
        cfg.default_height = 512;
        cfg.default_steps = 20;
        cfg.default_negative_prompt = Some("ugly, blurry".to_string());
        cfg.models.insert(
            "my-model".to_string(),
            ModelConfig {
                transformer: Some("/models/t.safetensors".to_string()),
                vae: Some("/models/v.safetensors".to_string()),
                default_steps: Some(30),
                lora: Some("/lora/adapter.safetensors".to_string()),
                lora_scale: Some(0.6),
                ..ModelConfig::default()
            },
        );

        cfg.save().unwrap();

        // Re-read the file and parse
        let config_path = mold_home.join("config.toml");
        let contents = std::fs::read_to_string(&config_path).unwrap();
        let reloaded: Config = toml::from_str(&contents).unwrap();

        std::env::remove_var("MOLD_HOME");
        let _ = std::fs::remove_dir_all(&mold_home);

        assert_eq!(reloaded.default_model, "sd15:fp16");
        assert_eq!(reloaded.server_port, 9999);
        assert_eq!(reloaded.default_width, 512);
        assert_eq!(reloaded.default_steps, 20);
        assert_eq!(
            reloaded.default_negative_prompt.as_deref(),
            Some("ugly, blurry")
        );
        let model = reloaded.models.get("my-model").unwrap();
        assert_eq!(model.default_steps, Some(30));
        assert_eq!(model.lora.as_deref(), Some("/lora/adapter.safetensors"));
        assert_eq!(model.lora_scale, Some(0.6));
    }

    // ── effective_negative_prompt ────────────────────────────────────────

    #[test]
    fn effective_negative_prompt_none_by_default() {
        let cfg = Config::default();
        let mc = ModelConfig::default();
        assert!(mc.effective_negative_prompt(&cfg).is_none());
    }

    #[test]
    fn effective_negative_prompt_global_fallback() {
        let cfg = Config {
            default_negative_prompt: Some("ugly, blurry".to_string()),
            ..Config::default()
        };
        let mc = ModelConfig::default();
        assert_eq!(
            mc.effective_negative_prompt(&cfg).as_deref(),
            Some("ugly, blurry")
        );
    }

    #[test]
    fn effective_negative_prompt_model_overrides_global() {
        let cfg = Config {
            default_negative_prompt: Some("ugly, blurry".to_string()),
            ..Config::default()
        };
        let mc = ModelConfig {
            negative_prompt: Some("watermark".to_string()),
            ..ModelConfig::default()
        };
        assert_eq!(
            mc.effective_negative_prompt(&cfg).as_deref(),
            Some("watermark")
        );
    }

    // ── effective_lora ──────────────────────────────────────────────────

    #[test]
    fn effective_lora_none_when_not_configured() {
        let mc = ModelConfig::default();
        assert!(mc.effective_lora().is_none());
    }

    #[test]
    fn effective_lora_returns_path_and_default_scale() {
        let mc = ModelConfig {
            lora: Some("/path/to/adapter.safetensors".to_string()),
            ..ModelConfig::default()
        };
        let (path, scale) = mc.effective_lora().unwrap();
        assert_eq!(path, "/path/to/adapter.safetensors");
        assert_eq!(scale, 1.0); // default scale
    }

    #[test]
    fn effective_lora_uses_configured_scale() {
        let mc = ModelConfig {
            lora: Some("/path/to/adapter.safetensors".to_string()),
            lora_scale: Some(0.5),
            ..ModelConfig::default()
        };
        let (_, scale) = mc.effective_lora().unwrap();
        assert_eq!(scale, 0.5);
    }

    // ── lookup_model_config canonical resolution ────────────────────────

    #[test]
    fn lookup_model_config_exact_match() {
        let mut models = HashMap::new();
        models.insert(
            "flux-dev:q8".to_string(),
            ModelConfig {
                default_steps: Some(25),
                ..ModelConfig::default()
            },
        );
        let cfg = Config {
            models,
            ..Config::default()
        };
        let mc = cfg.lookup_model_config("flux-dev:q8").unwrap();
        assert_eq!(mc.default_steps, Some(25));
    }

    #[test]
    fn lookup_model_config_resolves_bare_name_to_tagged() {
        // "flux-dev" should resolve to "flux-dev:q8" via resolve_model_name
        let mut models = HashMap::new();
        models.insert(
            "flux-dev:q8".to_string(),
            ModelConfig {
                default_steps: Some(25),
                ..ModelConfig::default()
            },
        );
        let cfg = Config {
            models,
            ..Config::default()
        };
        let mc = cfg.lookup_model_config("flux-dev");
        // This depends on resolve_model_name trying :q8 first
        assert!(mc.is_some());
        assert_eq!(mc.unwrap().default_steps, Some(25));
    }

    #[test]
    fn lookup_model_config_returns_none_for_unknown() {
        let cfg = Config::default();
        assert!(cfg.lookup_model_config("__nonexistent__").is_none());
    }

    // ── upsert_model / remove_model ─────────────────────────────────────

    #[test]
    fn upsert_and_remove_model() {
        let mut cfg = Config::default();
        assert!(cfg.models.is_empty());

        cfg.upsert_model(
            "test-model".to_string(),
            ModelConfig {
                transformer: Some("/t.safetensors".to_string()),
                ..ModelConfig::default()
            },
        );
        assert!(cfg.models.contains_key("test-model"));

        let removed = cfg.remove_model("test-model");
        assert!(removed.is_some());
        assert!(cfg.models.is_empty());

        // Removing again returns None
        assert!(cfg.remove_model("test-model").is_none());
    }

    // ── config_path / data_dir ──────────────────────────────────────────

    #[test]
    fn config_path_ends_with_config_toml() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_HOME");
        let path = Config::config_path().unwrap();
        assert!(
            path.ends_with("config.toml"),
            "should end with config.toml: got {:?}",
            path
        );
    }

    #[test]
    fn data_dir_equals_mold_dir() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_HOME");
        assert_eq!(Config::data_dir(), Config::mold_dir());
    }

    #[test]
    fn exists_on_disk_false_for_nonexistent() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mold_home = test_models_dir("no-config-file");
        std::fs::create_dir_all(&mold_home).unwrap();
        std::env::set_var("MOLD_HOME", &mold_home);
        assert!(!Config::exists_on_disk());
        std::env::remove_var("MOLD_HOME");
        let _ = std::fs::remove_dir_all(&mold_home);
    }

    // ── TOML serde edge cases ───────────────────────────────────────────

    #[test]
    fn config_deser_with_expand_section() {
        let toml_str = r#"
default_model = "flux-dev"

[expand]
enabled = true
backend = "local"
model = "qwen3-expand:q8"
temperature = 0.9
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(cfg.expand.enabled);
        assert_eq!(cfg.expand.backend, "local");
        assert_eq!(cfg.expand.model, "qwen3-expand:q8");
        assert!((cfg.expand.temperature - 0.9).abs() < 0.001);
    }

    #[test]
    fn config_deser_negative_prompt_global_and_model() {
        let toml_str = r#"
default_model = "sd15:fp16"
default_negative_prompt = "ugly, blurry, watermark"

[models."sd15:fp16"]
transformer = "/path/t.safetensors"
vae = "/path/v.safetensors"
negative_prompt = "anime, cartoon"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(
            cfg.default_negative_prompt.as_deref(),
            Some("ugly, blurry, watermark")
        );
        let mc = cfg.models.get("sd15:fp16").unwrap();
        assert_eq!(mc.negative_prompt.as_deref(), Some("anime, cartoon"));
    }

    #[test]
    fn config_deser_with_scheduler() {
        let toml_str = r#"
[models."sd15:fp16"]
transformer = "/path/t.safetensors"
vae = "/path/v.safetensors"
scheduler = "euler-ancestral"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        let mc = cfg.models.get("sd15:fp16").unwrap();
        assert!(mc.scheduler.is_some());
    }

    #[test]
    fn config_deser_with_model_lora_and_turbo() {
        let toml_str = r#"
[models."sdxl-turbo:fp16"]
transformer = "/path/t.safetensors"
vae = "/path/v.safetensors"
lora = "/path/adapter.safetensors"
lora_scale = 0.75
is_turbo = true
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        let mc = cfg.models.get("sdxl-turbo:fp16").unwrap();
        assert_eq!(mc.lora.as_deref(), Some("/path/adapter.safetensors"));
        assert_eq!(mc.lora_scale, Some(0.75));
        assert_eq!(mc.is_turbo, Some(true));
    }

    #[test]
    fn config_deser_with_t5_and_qwen3_variant() {
        let toml_str = r#"
default_model = "flux-dev"
t5_variant = "q4"
qwen3_variant = "iq4"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.t5_variant.as_deref(), Some("q4"));
        assert_eq!(cfg.qwen3_variant.as_deref(), Some("iq4"));
    }

    // ── resolved_output_dir bare tilde ──────────────────────────────────

    #[test]
    fn resolved_output_dir_bare_tilde_resolves_to_home() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("MOLD_OUTPUT_DIR");
        let mut cfg = Config::default();
        cfg.output_dir = Some("~".to_string());
        let dir = cfg.resolved_output_dir().unwrap();
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        assert_eq!(dir, home);
    }

    // ── resolved_log_dir bare tilde ─────────────────────────────────────

    #[test]
    fn resolved_log_dir_bare_tilde_resolves_to_home() {
        let mut cfg = Config::default();
        cfg.logging.dir = Some("~".to_string());
        let dir = cfg.resolved_log_dir();
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        assert_eq!(dir, home);
    }

    // ── migration idempotency ───────────────────────────────────────────

    #[test]
    fn run_migrations_idempotent_on_already_migrated() {
        let mut cfg = Config::default(); // config_version = CURRENT
        cfg.models.insert(
            "flux-dev:q8".to_string(),
            ModelConfig {
                transformer: Some("/path/t.gguf".to_string()),
                vae: Some("/path/v.safetensors".to_string()),
                default_steps: Some(25), // user override, not stale
                ..ModelConfig::default()
            },
        );
        let original_steps = cfg.models["flux-dev:q8"].default_steps;

        // Running migrations on a current-version config should be a no-op
        Config::run_migrations(&mut cfg);
        assert_eq!(cfg.models["flux-dev:q8"].default_steps, original_steps);
    }

    // ── DefaultModelResolution Debug/Clone ───────────────────────────────

    #[test]
    fn default_model_resolution_is_clone_and_debug() {
        let resolution = crate::config::DefaultModelResolution {
            model: "test".to_string(),
            source: crate::config::DefaultModelSource::EnvVar,
        };
        let cloned = resolution.clone();
        assert_eq!(cloned.model, "test");
        // Debug should not panic
        let _ = format!("{:?}", cloned);
    }

    #[test]
    fn default_model_source_eq() {
        assert_eq!(
            crate::config::DefaultModelSource::EnvVar,
            crate::config::DefaultModelSource::EnvVar
        );
        assert_ne!(
            crate::config::DefaultModelSource::EnvVar,
            crate::config::DefaultModelSource::Config
        );
    }

    // --- Upscaler/utility manifest_model_is_downloaded (issue #184) ---

    #[test]
    fn upscaler_manifest_model_is_downloaded_when_files_exist() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_models_dir("upscaler-downloaded");
        populate_manifest_files(&dir, "real-esrgan-x4plus:fp16");
        std::env::set_var("MOLD_MODELS_DIR", &dir);
        let cfg = Config::default();
        assert!(
            cfg.manifest_model_is_downloaded("real-esrgan-x4plus:fp16"),
            "upscaler with files on disk should be reported as downloaded"
        );
        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn upscaler_manifest_model_not_downloaded_when_files_missing() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_models_dir("upscaler-missing");
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("MOLD_MODELS_DIR", &dir);
        let cfg = Config::default();
        assert!(
            !cfg.manifest_model_is_downloaded("real-esrgan-x4plus:fp16"),
            "upscaler without files should not be reported as downloaded"
        );
        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn utility_manifest_model_is_downloaded_when_files_exist() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_models_dir("utility-downloaded");
        populate_manifest_files(&dir, "qwen3-expand:q8");
        std::env::set_var("MOLD_MODELS_DIR", &dir);
        let cfg = Config::default();
        assert!(
            cfg.manifest_model_is_downloaded("qwen3-expand:q8"),
            "utility model with files on disk should be reported as downloaded"
        );
        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn utility_manifest_model_not_downloaded_when_files_missing() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_models_dir("utility-missing");
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("MOLD_MODELS_DIR", &dir);
        let cfg = Config::default();
        assert!(
            !cfg.manifest_model_is_downloaded("qwen3-expand:q8"),
            "utility model without files should not be reported as downloaded"
        );
        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn upscaler_discovered_manifest_paths_returns_some() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_models_dir("upscaler-paths");
        populate_manifest_files(&dir, "real-esrgan-x4plus:fp16");
        std::env::set_var("MOLD_MODELS_DIR", &dir);
        let cfg = Config::default();
        let paths = cfg.discovered_manifest_paths("real-esrgan-x4plus:fp16");
        assert!(
            paths.is_some(),
            "upscaler with files on disk should produce ModelPaths"
        );
        let paths = paths.unwrap();
        assert!(
            paths
                .transformer
                .to_string_lossy()
                .contains("diffusion_pytorch_model"),
            "upscaler transformer should point to weights file"
        );
        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn utility_discovered_manifest_paths_returns_some() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_models_dir("utility-paths");
        populate_manifest_files(&dir, "qwen3-expand:q8");
        std::env::set_var("MOLD_MODELS_DIR", &dir);
        let cfg = Config::default();
        let paths = cfg.discovered_manifest_paths("qwen3-expand:q8");
        assert!(
            paths.is_some(),
            "utility model with files on disk should produce ModelPaths"
        );
        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn qwen_image_fp8_stale_marker_does_not_fall_back_to_stale_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_models_dir("qwen-image-fp8-stale-marker");
        populate_manifest_files(&dir, "qwen-image:fp8");
        create_pulling_marker(&dir, "qwen-image:fp8");
        std::env::set_var("MOLD_MODELS_DIR", &dir);

        let mut models = HashMap::new();
        models.insert(
            "qwen-image:fp8".to_string(),
            ModelConfig {
                transformer: Some("/cfg/stale-transformer.safetensors".to_string()),
                vae: Some("/cfg/stale-vae.safetensors".to_string()),
                text_encoder_files: Some(vec!["/cfg/stale-text-encoder.safetensors".to_string()]),
                text_tokenizer: Some("/cfg/stale-tokenizer.json".to_string()),
                ..ModelConfig::default()
            },
        );
        let cfg = Config {
            models,
            ..Config::default()
        };

        let paths = ModelPaths::resolve("qwen-image:fp8", &cfg).unwrap();
        assert!(
            paths.transformer.starts_with(&dir),
            "resolver should prefer manifest-discovered transformer over stale config"
        );
        assert_eq!(
            paths.vae,
            dir.join("shared/qwen-image-base/vae/diffusion_pytorch_model.safetensors")
        );
        assert_eq!(
            paths.text_encoder_files,
            vec![
                dir.join("shared/qwen-image-base/text_encoder/model-00001-of-00004.safetensors"),
                dir.join("shared/qwen-image-base/text_encoder/model-00002-of-00004.safetensors"),
                dir.join("shared/qwen-image-base/text_encoder/model-00003-of-00004.safetensors"),
                dir.join("shared/qwen-image-base/text_encoder/model-00004-of-00004.safetensors"),
            ]
        );
        assert_eq!(
            paths.text_tokenizer,
            Some(dir.join("shared/qwen-image/tokenizer.json"))
        );
        assert!(
            !crate::download::has_pulling_marker("qwen-image:fp8"),
            "stale marker should be self-healed after successful manifest discovery"
        );

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn stale_pull_markers_self_heal_for_all_manifests_with_complete_files() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_models_dir("all-manifests-stale-marker");
        std::env::set_var("MOLD_MODELS_DIR", &dir);
        let cfg = Config::default();

        for manifest in known_manifests() {
            if manifest.is_auxiliary() {
                continue;
            }
            populate_manifest_files(&dir, &manifest.name);
            create_pulling_marker(&dir, &manifest.name);

            assert!(
                cfg.manifest_model_is_downloaded(&manifest.name),
                "{} should be treated as downloaded when all files exist",
                manifest.name
            );
            assert!(
                cfg.discovered_manifest_paths(&manifest.name).is_some(),
                "{} should resolve manifest paths even with a stale marker",
                manifest.name
            );
            assert!(
                !crate::download::has_pulling_marker(&manifest.name),
                "{} stale marker should be removed",
                manifest.name
            );
        }

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(dir);
    }
}
