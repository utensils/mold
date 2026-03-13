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
        assert_eq!(cfg.default_width, 768);
        assert_eq!(cfg.default_height, 768);
        assert_eq!(cfg.default_steps, 4);
        assert!(cfg.models.is_empty());
    }

    #[test]
    fn config_load_or_default_missing_file() {
        let cfg = Config::load_or_default();
        assert!(!cfg.default_model.is_empty());
    }

    #[test]
    fn config_load_from_toml() {
        let toml = r#"
default_model = "flux-dev"
server_port = 9000
default_width = 896
default_height = 1152
default_steps = 25
"#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.default_model, "flux-dev");
        assert_eq!(cfg.server_port, 9000);
        assert_eq!(cfg.default_width, 896);
        assert_eq!(cfg.default_height, 1152);
        assert_eq!(cfg.default_steps, 25);
    }

    #[test]
    fn config_load_from_toml_partial() {
        let toml = r#"default_model = "flux-dev""#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.default_model, "flux-dev");
        assert_eq!(cfg.server_port, 7680);
        assert_eq!(cfg.default_steps, 4); // default
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
        let mut models = HashMap::new();
        models.insert("test-flux".to_string(), full_model_config("/tmp"));
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
        assert_eq!(
            paths.t5_tokenizer.to_str().unwrap(),
            "/tmp/t5.tokenizer.json"
        );
        assert_eq!(
            paths.clip_tokenizer.to_str().unwrap(),
            "/tmp/clip.tokenizer.json"
        );
    }

    #[test]
    fn model_paths_resolve_partial_config_returns_none() {
        // Only transformer is set; other paths are missing → None.
        let _lock = ENV_LOCK.lock().unwrap();
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
        let _lock = ENV_LOCK.lock().unwrap();
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
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("MOLD_TRANSFORMER_PATH", "/env/transformer.gguf");
        std::env::set_var("MOLD_VAE_PATH", "/env/vae.safetensors");
        std::env::set_var("MOLD_T5_PATH", "/env/t5.safetensors");
        std::env::set_var("MOLD_CLIP_PATH", "/env/clip.safetensors");
        std::env::set_var("MOLD_T5_TOKENIZER_PATH", "/env/t5.tokenizer.json");
        std::env::set_var("MOLD_CLIP_TOKENIZER_PATH", "/env/clip.tokenizer.json");

        let paths = ModelPaths::resolve("flux-schnell", &Config::default()).unwrap();
        assert_eq!(paths.transformer.to_str().unwrap(), "/env/transformer.gguf");
        assert_eq!(
            paths.t5_tokenizer.to_str().unwrap(),
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
    fn model_paths_config_takes_precedence_over_env() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("MOLD_TRANSFORMER_PATH", "/env/should-not-use.gguf");

        let mut models = HashMap::new();
        models.insert("flux-schnell".to_string(), full_model_config("/cfg"));
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
        let dir = Config::default().resolved_models_dir();
        assert_eq!(dir.to_str().unwrap(), "/custom/models");
        std::env::remove_var("MOLD_MODELS_DIR");
    }

    #[test]
    fn resolved_models_dir_default_expands_tilde() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::remove_var("MOLD_MODELS_DIR");
        let dir = Config::default().resolved_models_dir();
        assert!(
            !dir.to_str().unwrap().contains('~'),
            "tilde should be expanded: got {:?}",
            dir
        );
    }
}
