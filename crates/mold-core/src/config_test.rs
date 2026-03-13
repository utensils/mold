#[cfg(test)]
mod tests {
    use crate::config::{Config, ModelPaths};
    use std::collections::HashMap;
    use std::sync::Mutex;

    // Serialize tests that mutate env vars to avoid race conditions.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
        // Non-existent path should silently return defaults
        let cfg = Config::load_or_default();
        // As long as ~/.mold/config.toml doesn't exist, this returns defaults.
        // We can only assert it doesn't panic.
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
        // Only override some fields; rest should use serde defaults
        let toml = r#"default_model = "flux-dev""#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.default_model, "flux-dev");
        assert_eq!(cfg.server_port, 7680); // default
    }

    #[test]
    fn model_paths_resolve_from_env() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("MOLD_TRANSFORMER_PATH", "/tmp/transformer.gguf");
        std::env::set_var("MOLD_VAE_PATH", "/tmp/vae.safetensors");
        std::env::set_var("MOLD_T5_PATH", "/tmp/t5.safetensors");
        std::env::set_var("MOLD_CLIP_PATH", "/tmp/clip.safetensors");

        let cfg = Config {
            models: HashMap::new(),
            ..Config::default()
        };
        let paths = ModelPaths::resolve("flux-schnell", &cfg).unwrap();
        assert_eq!(paths.transformer.to_str().unwrap(), "/tmp/transformer.gguf");
        assert_eq!(paths.vae.to_str().unwrap(), "/tmp/vae.safetensors");

        std::env::remove_var("MOLD_TRANSFORMER_PATH");
        std::env::remove_var("MOLD_VAE_PATH");
        std::env::remove_var("MOLD_T5_PATH");
        std::env::remove_var("MOLD_CLIP_PATH");
    }

    #[test]
    fn model_paths_resolve_missing_returns_none() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::remove_var("MOLD_TRANSFORMER_PATH");
        std::env::remove_var("MOLD_VAE_PATH");
        std::env::remove_var("MOLD_T5_PATH");
        std::env::remove_var("MOLD_CLIP_PATH");

        let cfg = Config::default();
        let paths = ModelPaths::resolve("flux-schnell", &cfg);
        assert!(paths.is_none());
    }

    #[test]
    fn resolved_models_dir_from_env() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("MOLD_MODELS_DIR", "/custom/models");
        let cfg = Config::default();
        let dir = cfg.resolved_models_dir();
        assert_eq!(dir.to_str().unwrap(), "/custom/models");
        std::env::remove_var("MOLD_MODELS_DIR");
    }
}
