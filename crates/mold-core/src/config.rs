use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

/// Per-model file path configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub transformer: Option<String>,
    pub vae: Option<String>,
    pub t5_encoder: Option<String>,
    pub clip_encoder: Option<String>,
}

/// Resolved model file paths (all required).
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub transformer: PathBuf,
    pub vae: PathBuf,
    pub t5_encoder: PathBuf,
    pub clip_encoder: PathBuf,
}

impl ModelPaths {
    /// Resolve paths for a model. Checks config, then env vars, then defaults.
    pub fn resolve(model_name: &str, config: &Config) -> Option<Self> {
        let model_cfg = config.models.get(model_name);

        let transformer = Self::resolve_path(
            model_cfg.and_then(|m| m.transformer.as_deref()),
            "MOLD_TRANSFORMER_PATH",
        )?;
        let vae = Self::resolve_path(model_cfg.and_then(|m| m.vae.as_deref()), "MOLD_VAE_PATH")?;
        let t5_encoder = Self::resolve_path(
            model_cfg.and_then(|m| m.t5_encoder.as_deref()),
            "MOLD_T5_PATH",
        )?;
        let clip_encoder = Self::resolve_path(
            model_cfg.and_then(|m| m.clip_encoder.as_deref()),
            "MOLD_CLIP_PATH",
        )?;

        Some(Self {
            transformer,
            vae,
            t5_encoder,
            clip_encoder,
        })
    }

    fn resolve_path(config_val: Option<&str>, env_var: &str) -> Option<PathBuf> {
        if let Some(path) = config_val {
            return Some(PathBuf::from(path));
        }
        if let Ok(path) = std::env::var(env_var) {
            return Some(PathBuf::from(path));
        }
        None
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_model")]
    pub default_model: String,

    #[serde(default = "default_models_dir")]
    pub models_dir: String,

    #[serde(default = "default_port")]
    pub server_port: u16,

    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    #[serde(default = "default_dimension")]
    pub default_width: u32,

    #[serde(default = "default_dimension")]
    pub default_height: u32,

    /// Per-model path overrides, keyed by model name.
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
}

fn default_model() -> String {
    "flux-schnell".to_string()
}

fn default_models_dir() -> String {
    "~/.mold/models".to_string()
}

fn default_port() -> u16 {
    7680
}

fn default_output_dir() -> String {
    ".".to_string()
}

fn default_dimension() -> u32 {
    1024
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_model: default_model(),
            models_dir: default_models_dir(),
            server_port: default_port(),
            output_dir: default_output_dir(),
            default_width: default_dimension(),
            default_height: default_dimension(),
            models: HashMap::new(),
        }
    }
}

impl Config {
    pub fn load_or_default() -> Self {
        let config_path = Self::config_path();
        if config_path.exists() {
            match std::fs::read_to_string(&config_path) {
                Ok(contents) => toml::from_str(&contents).unwrap_or_default(),
                Err(_) => Config::default(),
            }
        } else {
            Config::default()
        }
    }

    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".mold")
            .join("config.toml")
    }

    pub fn data_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".mold")
    }

    pub fn resolved_models_dir(&self) -> PathBuf {
        if let Ok(env_dir) = std::env::var("MOLD_MODELS_DIR") {
            PathBuf::from(env_dir)
        } else {
            let expanded = self.models_dir.replace(
                "~",
                &dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .to_string_lossy(),
            );
            PathBuf::from(expanded)
        }
    }
}
