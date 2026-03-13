#[cfg(test)]
mod tests {
    use crate::config::{Config, ModelConfig, ModelPaths};
    use std::collections::HashMap;
    use std::sync::Mutex;

    // Serialize tests that mutate env vars to avoid race conditions.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // ── Config deserialization ────────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = Config::default();
        assert_eq!(cfg.default_model, "flux-schnell");
        assert_eq!(cfg.server_port, 7680);
        assert_eq!(cfg.default_width, 1024);
        assert_eq!(cfg.default_height, 1024);
        assert!(cfg.models.is_empty());
    }

    #[test]
    fn config_load_or_default_missing_file() {
        // Non-existent path should return defaults without panicking.
        let cfg = Config::load_or_default();
        assert!(!cfg.default_model.is_empty());
    }

    #[test]
    fn config_load_from_toml() {
        let toml = r#"
default_model = "flux-dev"
server_port = 9000
default_width = 768
default_height = 512
"#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.default_model, "flux-dev");
        assert_eq!(cfg.server_port, 9000);
        assert_eq!(cfg.default_width, 768);
        assert_eq!(cfg.default_height, 512);
    }

    #[test]
    fn config_load_from_toml_partial() {
        // Only override some fields; rest should use serde defaults.
        let toml = r#"default_model = "flux-dev""#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.default_model, "flux-dev");
        assert_eq!(cfg.server_port, 7680); // default
    }

    // ── ModelPaths via config map (no env vars → deterministic) ──────────────

    #[test]
    fn model_paths_resolve_from_config() {
        let mut models = HashMap::new();
        models.insert(
            "test-flux".to_string(),
            ModelConfig {
                transformer: Some("/tmp/transformer.gguf".to_string()),
                vae: Some("/tmp/vae.safetensors".to_string()),
                t5_encoder: Some("/tmp/t5.safetensors".to_string()),
                clip_encoder: Some("/tmp/clip.safetensors".to_string()),
            },
        );
        let cfg = Config {
            models,
            ..Config::default()
        };
        let paths = ModelPaths::resolve("test-flux", &cfg).unwrap();
        assert_eq!(paths.transformer.to_str().unwrap(), "/tmp/transformer.gguf");
        assert_eq!(paths.vae.to_str().unwrap(), "/tmp/vae.safetensors");
        assert_eq!(paths.t5_encoder.to_str().unwrap(), "/tmp/t5.safetensors");
        assert_eq!(
            paths.clip_encoder.to_str().unwrap(),
            "/tmp/clip.safetensors"
        );
    }

    #[test]
    fn model_paths_resolve_partial_config_returns_none() {
        // Only transformer is set; other paths are missing → should return None.
        let mut models = HashMap::new();
        models.insert(
            "partial".to_string(),
            ModelConfig {
                transformer: Some("/tmp/t.gguf".to_string()),
                vae: None,
                t5_encoder: None,
                clip_encoder: None,
            },
        );
        let cfg = Config {
            models,
            ..Config::default()
        };

        // Must hold ENV_LOCK to ensure no other test has set the fallback env vars.
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::remove_var("MOLD_VAE_PATH");
        std::env::remove_var("MOLD_T5_PATH");
        std::env::remove_var("MOLD_CLIP_PATH");

        let paths = ModelPaths::resolve("partial", &cfg);
        assert!(paths.is_none(), "expected None for incomplete config");
    }

    #[test]
    fn model_paths_resolve_unknown_model_no_env_returns_none() {
        // Completely unknown model with no env vars → None.
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::remove_var("MOLD_TRANSFORMER_PATH");
        std::env::remove_var("MOLD_VAE_PATH");
        std::env::remove_var("MOLD_T5_PATH");
        std::env::remove_var("MOLD_CLIP_PATH");

        let cfg = Config::default();
        let paths = ModelPaths::resolve("__no_such_model__", &cfg);
        assert!(paths.is_none());
    }

    #[test]
    fn model_paths_env_var_fallback() {
        // Env vars are used when no config entry exists.
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("MOLD_TRANSFORMER_PATH", "/env/transformer.gguf");
        std::env::set_var("MOLD_VAE_PATH", "/env/vae.safetensors");
        std::env::set_var("MOLD_T5_PATH", "/env/t5.safetensors");
        std::env::set_var("MOLD_CLIP_PATH", "/env/clip.safetensors");

        let cfg = Config::default(); // no model entries
        let paths = ModelPaths::resolve("flux-schnell", &cfg).unwrap();
        assert_eq!(paths.transformer.to_str().unwrap(), "/env/transformer.gguf");
        assert_eq!(paths.vae.to_str().unwrap(), "/env/vae.safetensors");

        std::env::remove_var("MOLD_TRANSFORMER_PATH");
        std::env::remove_var("MOLD_VAE_PATH");
        std::env::remove_var("MOLD_T5_PATH");
        std::env::remove_var("MOLD_CLIP_PATH");
    }

    #[test]
    fn model_paths_config_takes_precedence_over_env() {
        // Explicit config paths beat env var fallbacks.
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("MOLD_TRANSFORMER_PATH", "/env/should-not-use.gguf");

        let mut models = HashMap::new();
        models.insert(
            "flux-schnell".to_string(),
            ModelConfig {
                transformer: Some("/cfg/transformer.gguf".to_string()),
                vae: Some("/cfg/vae.safetensors".to_string()),
                t5_encoder: Some("/cfg/t5.safetensors".to_string()),
                clip_encoder: Some("/cfg/clip.safetensors".to_string()),
            },
        );
        let cfg = Config {
            models,
            ..Config::default()
        };
        let paths = ModelPaths::resolve("flux-schnell", &cfg).unwrap();
        assert_eq!(
            paths.transformer.to_str().unwrap(),
            "/cfg/transformer.gguf",
            "config path should override env var"
        );

        std::env::remove_var("MOLD_TRANSFORMER_PATH");
    }

    // ── resolved_models_dir ───────────────────────────────────────────────────

    #[test]
    fn resolved_models_dir_from_env() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("MOLD_MODELS_DIR", "/custom/models");
        let cfg = Config::default();
        let dir = cfg.resolved_models_dir();
        assert_eq!(dir.to_str().unwrap(), "/custom/models");
        std::env::remove_var("MOLD_MODELS_DIR");
    }

    #[test]
    fn resolved_models_dir_default_expands_tilde() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::remove_var("MOLD_MODELS_DIR");
        let cfg = Config::default();
        let dir = cfg.resolved_models_dir();
        // Default is ~/.mold/models; should not contain a literal '~'
        assert!(
            !dir.to_str().unwrap().contains('~'),
            "tilde should be expanded: got {:?}",
            dir
        );
    }
}
