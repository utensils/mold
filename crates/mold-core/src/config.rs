use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::manifest::resolve_model_name;
use crate::types::Scheduler;

/// Per-model file path + default settings configuration.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ModelConfig {
    // --- paths ---
    pub transformer: Option<String>,
    /// Multi-shard transformer paths (Z-Image BF16); empty means use single `transformer`
    pub transformer_shards: Option<Vec<String>>,
    pub vae: Option<String>,
    pub t5_encoder: Option<String>,
    pub clip_encoder: Option<String>,
    pub t5_tokenizer: Option<String>,
    pub clip_tokenizer: Option<String>,
    /// CLIP-G / OpenCLIP encoder path (SDXL only)
    pub clip_encoder_2: Option<String>,
    /// CLIP-G / OpenCLIP tokenizer path (SDXL only)
    pub clip_tokenizer_2: Option<String>,
    /// Generic text encoder shard paths (Qwen3 for Z-Image)
    pub text_encoder_files: Option<Vec<String>>,
    /// Generic text encoder tokenizer path (Qwen3 for Z-Image)
    pub text_tokenizer: Option<String>,
    /// Stage B decoder weights path (Wuerstchen only)
    pub decoder: Option<String>,

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
    /// Whether this model uses a turbo (few-step distilled) schedule.
    /// If None, auto-detected from the model name.
    pub is_turbo: Option<bool>,
    /// Scheduler algorithm for UNet-based models (SD1.5, SDXL). Ignored by flow-matching models.
    pub scheduler: Option<Scheduler>,

    // --- metadata ---
    pub description: Option<String>,
    pub family: Option<String>,
}

impl ModelConfig {
    /// Collect all file path strings from this model config into a flat list.
    /// Used for reference counting when determining which files are shared.
    pub fn all_file_paths(&self) -> Vec<String> {
        let mut paths = Vec::new();
        let singles = [
            &self.transformer,
            &self.vae,
            &self.t5_encoder,
            &self.clip_encoder,
            &self.t5_tokenizer,
            &self.clip_tokenizer,
            &self.clip_encoder_2,
            &self.clip_tokenizer_2,
            &self.text_tokenizer,
            &self.decoder,
        ];
        for p in singles.into_iter().flatten() {
            paths.push(p.clone());
        }
        if let Some(ref shards) = self.transformer_shards {
            paths.extend(shards.iter().cloned());
        }
        if let Some(ref files) = self.text_encoder_files {
            paths.extend(files.iter().cloned());
        }
        paths
    }

    /// Effective steps: model default → global fallback → hardcoded default.
    pub fn effective_steps(&self, global_cfg: &Config) -> u32 {
        self.default_steps.unwrap_or(global_cfg.default_steps)
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

/// Resolved model file paths.
/// `transformer` and `vae` are always required.
/// Other paths are optional — each engine validates what it needs at load time.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub transformer: PathBuf,
    /// Multi-shard transformer paths (Z-Image BF16); empty means use single `transformer`
    pub transformer_shards: Vec<PathBuf>,
    pub vae: PathBuf,
    pub t5_encoder: Option<PathBuf>,
    pub clip_encoder: Option<PathBuf>,
    pub t5_tokenizer: Option<PathBuf>,
    pub clip_tokenizer: Option<PathBuf>,
    /// CLIP-G / OpenCLIP encoder (SDXL only)
    pub clip_encoder_2: Option<PathBuf>,
    /// CLIP-G / OpenCLIP tokenizer (SDXL only)
    pub clip_tokenizer_2: Option<PathBuf>,
    /// Generic text encoder shard paths (Qwen3 for Z-Image)
    pub text_encoder_files: Vec<PathBuf>,
    /// Generic text encoder tokenizer (Qwen3 for Z-Image)
    pub text_tokenizer: Option<PathBuf>,
    /// Stage B decoder weights (Wuerstchen only)
    pub decoder: Option<PathBuf>,
}

impl ModelPaths {
    /// Resolve paths for a model. Checks config, then env vars.
    /// Returns None if transformer and VAE paths can't be resolved.
    /// All other paths are optional (depend on model family).
    pub fn resolve(model_name: &str, config: &Config) -> Option<Self> {
        let model_cfg = config.models.get(model_name);

        let transformer = Self::resolve_path(
            model_cfg.and_then(|m| m.transformer.as_deref()),
            "MOLD_TRANSFORMER_PATH",
        )?;
        let transformer_shards = model_cfg
            .and_then(|m| m.transformer_shards.as_ref())
            .map(|shards| shards.iter().map(PathBuf::from).collect())
            .unwrap_or_default();
        let vae = Self::resolve_path(model_cfg.and_then(|m| m.vae.as_deref()), "MOLD_VAE_PATH")?;
        let t5_encoder = Self::resolve_path(
            model_cfg.and_then(|m| m.t5_encoder.as_deref()),
            "MOLD_T5_PATH",
        );
        let clip_encoder = Self::resolve_path(
            model_cfg.and_then(|m| m.clip_encoder.as_deref()),
            "MOLD_CLIP_PATH",
        );
        let t5_tokenizer = Self::resolve_path(
            model_cfg.and_then(|m| m.t5_tokenizer.as_deref()),
            "MOLD_T5_TOKENIZER_PATH",
        );
        let clip_tokenizer = Self::resolve_path(
            model_cfg.and_then(|m| m.clip_tokenizer.as_deref()),
            "MOLD_CLIP_TOKENIZER_PATH",
        );
        let clip_encoder_2 = Self::resolve_path(
            model_cfg.and_then(|m| m.clip_encoder_2.as_deref()),
            "MOLD_CLIP2_PATH",
        );
        let clip_tokenizer_2 = Self::resolve_path(
            model_cfg.and_then(|m| m.clip_tokenizer_2.as_deref()),
            "MOLD_CLIP2_TOKENIZER_PATH",
        );
        let text_encoder_files = model_cfg
            .and_then(|m| m.text_encoder_files.as_ref())
            .map(|files| files.iter().map(PathBuf::from).collect())
            .unwrap_or_default();
        let text_tokenizer = Self::resolve_path(
            model_cfg.and_then(|m| m.text_tokenizer.as_deref()),
            "MOLD_TEXT_TOKENIZER_PATH",
        );
        let decoder = Self::resolve_path(
            model_cfg.and_then(|m| m.decoder.as_deref()),
            "MOLD_DECODER_PATH",
        );

        Some(Self {
            transformer,
            transformer_shards,
            vae,
            t5_encoder,
            clip_encoder,
            t5_tokenizer,
            clip_tokenizer,
            clip_encoder_2,
            clip_tokenizer_2,
            text_encoder_files,
            text_tokenizer,
            decoder,
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

    #[serde(default = "default_dimension")]
    pub default_width: u32,

    #[serde(default = "default_dimension")]
    pub default_height: u32,

    #[serde(default = "default_steps")]
    pub default_steps: u32,

    /// Preferred T5 encoder variant: "fp16" (default), "q8", "q6", "q5", "q4", "q3", or "auto".
    /// "auto" selects the best variant that fits in GPU VRAM.
    /// An explicit quantized tag always uses that variant regardless of VRAM.
    #[serde(default)]
    pub t5_variant: Option<String>,

    /// Preferred Qwen3 text encoder variant: "bf16" (default), "q8", "q6", "iq4", "q3", or "auto".
    /// "auto" selects the best variant that fits in GPU VRAM (with drop-and-reload).
    #[serde(default)]
    pub qwen3_variant: Option<String>,

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
            default_width: default_dimension(),
            default_height: default_dimension(),
            default_steps: default_steps(),
            t5_variant: None,
            qwen3_variant: None,
            models: HashMap::new(),
        }
    }
}

impl Config {
    pub fn load_or_default() -> Self {
        let Some(config_path) = Self::config_path() else {
            eprintln!("warning: could not determine home directory — using default config");
            return Config::default();
        };
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

    /// The root mold directory: `~/.mold/` on all platforms.
    /// Falls back to `./.mold` (relative to CWD) if the home directory cannot
    /// be determined (e.g. containers, CI). This preserves write-ability for
    /// `mold pull` and server-side auto-pull even when `HOME` is unset.
    pub fn mold_dir() -> Option<PathBuf> {
        Some(
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".mold"),
        )
    }

    pub fn config_path() -> Option<PathBuf> {
        Self::mold_dir().map(|d| d.join("config.toml"))
    }

    pub fn data_dir() -> Option<PathBuf> {
        Self::mold_dir()
    }

    pub fn resolved_models_dir(&self) -> PathBuf {
        if let Ok(env_dir) = std::env::var("MOLD_MODELS_DIR") {
            PathBuf::from(env_dir)
        } else {
            let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            let expanded = self.models_dir.replace("~", &home.to_string_lossy());
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

    /// Remove a model entry from the config, returning it if it existed.
    pub fn remove_model(&mut self, name: &str) -> Option<ModelConfig> {
        self.models.remove(name)
    }

    /// Write the config to disk at `config_path()`.
    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::config_path()
            .ok_or_else(|| anyhow::anyhow!("cannot determine home directory for config path"))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&path, contents)?;
        Ok(())
    }

    /// Whether a config file exists on disk.
    pub fn exists_on_disk() -> bool {
        Self::config_path().is_some_and(|p| p.exists())
    }
}
