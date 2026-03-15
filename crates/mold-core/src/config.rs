use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::manifest::resolve_model_name;

/// Per-model file path + default settings configuration.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ModelConfig {
    // --- paths ---
    pub transformer: Option<String>,
    pub vae: Option<String>,
    pub t5_encoder: Option<String>,
    pub clip_encoder: Option<String>,
    pub t5_tokenizer: Option<String>,
    pub clip_tokenizer: Option<String>,

    // --- generation defaults ---
    /// Default inference steps (e.g. 4 for schnell, 25 for dev)
    pub default_steps: Option<u32>,
    /// Default guidance scale (0.0 for schnell, 3.5 for dev finetuned)
    pub default_guidance: Option<f64>,
    /// Default output width
    pub default_width: Option<u32>,
    /// Default output height
    pub default_height: Option<u32>,
    /// Whether this model uses the schnell (distilled) timestep schedule.
    /// If None, auto-detected from the transformer filename.
    pub is_schnell: Option<bool>,

    // --- metadata ---
    pub description: Option<String>,
    pub family: Option<String>,
}

impl ModelConfig {
    /// Effective steps: model default → global fallback → hardcoded default.
    pub fn effective_steps(&self, global_cfg: &Config) -> u32 {
        self.default_steps
            .unwrap_or_else(|| global_cfg.default_steps)
    }

    /// Effective guidance.
    pub fn effective_guidance(&self) -> f64 {
        self.default_guidance.unwrap_or(3.5)
    }

    /// Effective width.
    pub fn effective_width(&self, global_cfg: &Config) -> u32 {
        self.default_width.unwrap_or(global_cfg.default_width)
    }

    /// Effective height.
    pub fn effective_height(&self, global_cfg: &Config) -> u32 {
        self.default_height.unwrap_or(global_cfg.default_height)
    }
}

/// Resolved model file paths (all required).
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub transformer: PathBuf,
    pub vae: PathBuf,
    pub t5_encoder: PathBuf,
    pub clip_encoder: PathBuf,
    pub t5_tokenizer: PathBuf,
    pub clip_tokenizer: PathBuf,
}

impl ModelPaths {
    /// Resolve paths for a model. Checks config, then env vars.
    /// Returns None if any required path is unresolvable.
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
        let t5_tokenizer = Self::resolve_path(
            model_cfg.and_then(|m| m.t5_tokenizer.as_deref()),
            "MOLD_T5_TOKENIZER_PATH",
        )?;
        let clip_tokenizer = Self::resolve_path(
            model_cfg.and_then(|m| m.clip_tokenizer.as_deref()),
            "MOLD_CLIP_TOKENIZER_PATH",
        )?;

        Some(Self {
            transformer,
            vae,
            t5_encoder,
            clip_encoder,
            t5_tokenizer,
            clip_tokenizer,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
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

    #[serde(default = "default_steps")]
    pub default_steps: u32,

    /// Per-model configurations, keyed by model name.
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
    768
}

fn default_steps() -> u32 {
    4
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
            default_steps: default_steps(),
            models: HashMap::new(),
        }
    }
}

impl Config {
    pub fn load_or_default() -> Self {
        let config_path = Self::config_path();
        if config_path.exists() {
            match std::fs::read_to_string(&config_path) {
                Ok(contents) => match toml::from_str(&contents) {
                    Ok(cfg) => cfg,
                    Err(e) => {
                        eprintln!(
                            "warning: failed to parse config at {}: {e} — using defaults",
                            config_path.display()
                        );
                        Config::default()
                    }
                },
                Err(e) => {
                    eprintln!(
                        "warning: failed to read config at {}: {e} — using defaults",
                        config_path.display()
                    );
                    Config::default()
                }
            }
        } else {
            Config::default()
        }
    }

    /// Returns true if `~/.mold/` exists (legacy layout).
    fn legacy_dir_exists() -> bool {
        dirs::home_dir()
            .map(|h| h.join(".mold").is_dir())
            .unwrap_or(false)
    }

    pub fn config_path() -> PathBuf {
        // Legacy: ~/.mold/config.toml if ~/.mold/ exists
        if Self::legacy_dir_exists() {
            return dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".mold")
                .join("config.toml");
        }
        // XDG: ~/.config/mold/config.toml
        dirs::config_dir()
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".config")
            })
            .join("mold")
            .join("config.toml")
    }

    pub fn data_dir() -> PathBuf {
        // Legacy: ~/.mold/ if it exists
        if Self::legacy_dir_exists() {
            return dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".mold");
        }
        // XDG: ~/.local/share/mold/
        dirs::data_dir()
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".local")
                    .join("share")
            })
            .join("mold")
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

    /// Return the ModelConfig for a given model name, or an empty default.
    /// Tries the exact name first, then the canonical `name:tag` form.
    pub fn model_config(&self, name: &str) -> ModelConfig {
        if let Some(cfg) = self.models.get(name) {
            return cfg.clone();
        }
        // Try canonical name resolution (e.g. "flux-dev-q4" -> "flux-dev:q4")
        let canonical = resolve_model_name(name);
        if canonical != name {
            if let Some(cfg) = self.models.get(&canonical) {
                return cfg.clone();
            }
        }
        ModelConfig::default()
    }

    /// Insert or update a model configuration entry.
    pub fn upsert_model(&mut self, name: String, config: ModelConfig) {
        self.models.insert(name, config);
    }

    /// Write the config to disk at `config_path()`.
    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&path, contents)?;
        Ok(())
    }

    /// Whether a config file exists on disk.
    pub fn exists_on_disk() -> bool {
        Self::config_path().exists()
    }
}
