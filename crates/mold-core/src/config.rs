use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::expand::ExpandSettings;
use crate::manifest::resolve_model_name;
use crate::types::Scheduler;

static RUNTIME_MODELS_DIR_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

/// Which fallback step resolved the default model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefaultModelSource {
    /// `MOLD_DEFAULT_MODEL` environment variable
    EnvVar,
    /// Config file `default_model` with a custom `[models]` entry
    ConfigCustomEntry,
    /// Config file `default_model` (manifest model, downloaded)
    Config,
    /// Last-used model from `$MOLD_HOME/last-model`
    LastUsed,
    /// Only one model is downloaded — auto-selected
    OnlyDownloaded,
    /// Config file default (model not downloaded, will auto-pull)
    ConfigDefault,
}

/// Result of resolving the default model: the model name and how it was resolved.
#[derive(Debug, Clone)]
pub struct DefaultModelResolution {
    pub model: String,
    pub source: DefaultModelSource,
}

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
    /// Per-model default negative prompt for CFG-based models.
    pub negative_prompt: Option<String>,
    /// Default LoRA adapter path for this model.
    pub lora: Option<String>,
    /// Default LoRA scale for this model (0.0-2.0).
    pub lora_scale: Option<f64>,

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

    /// Total disk usage of all model files: `(bytes, gigabytes)`.
    ///
    /// Sums the file sizes of all paths referenced by this config entry.
    /// Missing files are silently skipped.
    pub fn disk_usage(&self) -> (u64, f64) {
        let total: u64 = self
            .all_file_paths()
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        (total, total as f64 / 1_073_741_824.0)
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

    /// Effective negative prompt: per-model override → global default → None.
    pub fn effective_negative_prompt(&self, global_cfg: &Config) -> Option<String> {
        self.negative_prompt
            .clone()
            .or_else(|| global_cfg.default_negative_prompt.clone())
    }

    /// Effective LoRA config: per-model default path and scale, or None.
    pub fn effective_lora(&self) -> Option<(String, f64)> {
        self.lora
            .as_ref()
            .map(|path| (path.clone(), self.lora_scale.unwrap_or(1.0)))
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
        if let Some(model_cfg) = config.discovered_manifest_model_config(model_name) {
            return Self::resolve_from_model_config(Some(&model_cfg));
        }

        if crate::manifest::find_manifest(model_name).is_some() && config.has_models_dir_override()
        {
            return Self::resolve_from_model_config(None);
        }

        let model_cfg = config.lookup_model_config(model_name);
        Self::resolve_from_model_config(model_cfg.as_ref())
    }

    fn resolve_from_model_config(model_cfg: Option<&ModelConfig>) -> Option<Self> {
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
        if let Ok(path) = std::env::var(env_var) {
            return Some(PathBuf::from(path));
        }
        if let Some(path) = config_val {
            return Some(PathBuf::from(path));
        }
        None
    }
}

/// Current config schema version. Increment when adding migrations.
const CURRENT_CONFIG_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Config schema version for migrations. Old configs without this field
    /// default to 0 and are migrated on first load.
    #[serde(default)]
    pub config_version: u32,

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

    #[serde(default = "default_embed_metadata")]
    pub embed_metadata: bool,

    /// Preferred T5 encoder variant: "fp16" (default), "q8", "q6", "q5", "q4", "q3", or "auto".
    /// "auto" selects the best variant that fits in GPU VRAM.
    /// An explicit quantized tag always uses that variant regardless of VRAM.
    #[serde(default)]
    pub t5_variant: Option<String>,

    /// Preferred Qwen3 text encoder variant: "bf16" (default), "q8", "q6", "iq4", "q3", or "auto".
    /// "auto" selects the best variant that fits in GPU VRAM (with drop-and-reload).
    #[serde(default)]
    pub qwen3_variant: Option<String>,

    /// Directory to persist generated images. Default: `~/.mold/output/`.
    /// Override with `MOLD_OUTPUT_DIR` env var. Set to empty string to disable
    /// (TUI gallery will not function when disabled).
    #[serde(default)]
    pub output_dir: Option<String>,

    /// Global default negative prompt for CFG-based models (SD1.5, SDXL, SD3, Wuerstchen).
    /// Overridden by per-model `negative_prompt` or CLI `--negative-prompt`.
    #[serde(default)]
    pub default_negative_prompt: Option<String>,

    /// Prompt expansion settings.
    #[serde(default)]
    pub expand: ExpandSettings,

    /// Logging configuration.
    #[serde(default)]
    pub logging: LoggingConfig,

    /// Per-model configurations, keyed by model name.
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
}

/// Logging configuration for file output and rotation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    /// Log level: trace, debug, info, warn, error. Overridden by MOLD_LOG env var.
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Enable file logging. When true, logs go to ~/.mold/logs/.
    #[serde(default)]
    pub file: bool,

    /// Custom log file directory (default: ~/.mold/logs/).
    #[serde(default)]
    pub dir: Option<String>,

    /// Number of days to retain log files. Default: 7.
    #[serde(default = "default_log_max_days")]
    pub max_days: u32,
}

fn default_log_level() -> String {
    "info".to_string()
}
fn default_log_max_days() -> u32 {
    7
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: false,
            dir: None,
            max_days: default_log_max_days(),
        }
    }
}

fn default_model() -> String {
    "flux2-klein".to_string()
}

fn default_models_dir() -> String {
    if let Ok(home) = std::env::var("MOLD_HOME") {
        format!("{home}/models")
    } else {
        "~/.mold/models".to_string()
    }
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

fn default_embed_metadata() -> bool {
    true
}

impl Default for Config {
    fn default() -> Self {
        Self {
            config_version: CURRENT_CONFIG_VERSION,
            default_model: default_model(),
            models_dir: default_models_dir(),
            server_port: default_port(),
            default_width: default_dimension(),
            default_height: default_dimension(),
            default_steps: default_steps(),
            embed_metadata: default_embed_metadata(),
            t5_variant: None,
            qwen3_variant: None,
            output_dir: None,
            default_negative_prompt: None,
            expand: ExpandSettings::default(),
            logging: LoggingConfig::default(),
            models: HashMap::new(),
        }
    }
}

impl Config {
    pub fn install_runtime_models_dir_override(models_dir: PathBuf) {
        let _ = RUNTIME_MODELS_DIR_OVERRIDE.get_or_init(|| models_dir);
    }

    pub fn load_or_default() -> Self {
        let Some(config_path) = Self::config_path() else {
            eprintln!("warning: could not determine home directory — using default config");
            return Config::default();
        };
        let mut cfg = if config_path.exists() {
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
        };

        // Run config migrations if needed
        if cfg.config_version < CURRENT_CONFIG_VERSION {
            Self::run_migrations(&mut cfg);
            cfg.config_version = CURRENT_CONFIG_VERSION;
            if let Err(e) = cfg.save() {
                eprintln!("warning: failed to save migrated config: {e}");
            }
        }

        cfg
    }

    /// Run all pending config migrations from cfg.config_version to CURRENT.
    pub(crate) fn run_migrations(cfg: &mut Config) {
        if cfg.config_version < 1 {
            Self::migrate_v0_to_v1(cfg);
        }
        // Future migrations:
        // if cfg.config_version < 2 { Self::migrate_v1_to_v2(cfg); }
    }

    /// v0 → v1: Strip stale manifest defaults from known model entries.
    ///
    /// Old `mold pull` wrote all manifest defaults (steps, guidance, dimensions,
    /// description, family, is_schnell, scheduler) into config.toml. These become
    /// stale when manifests update. This migration removes them so
    /// `resolved_model_config()` reads fresh values from the manifest at runtime.
    fn migrate_v0_to_v1(cfg: &mut Config) {
        let model_names: Vec<String> = cfg.models.keys().cloned().collect();
        for name in model_names {
            if crate::manifest::find_manifest(&name).is_some() {
                if let Some(mc) = cfg.models.get_mut(&name) {
                    mc.default_steps = None;
                    mc.default_guidance = None;
                    mc.default_width = None;
                    mc.default_height = None;
                    mc.is_schnell = None;
                    mc.is_turbo = None;
                    mc.scheduler = None;
                    mc.negative_prompt = None;
                    mc.description = None;
                    mc.family = None;
                }
            }
        }
        eprintln!("config: migrated v0 → v1 (cleared stale manifest defaults)");
    }

    /// Reload config from disk while preserving runtime-only overrides.
    pub fn reload_from_disk_preserving_runtime(&self) -> Self {
        let mut fresh = Self::load_or_default();
        fresh.models_dir = self.models_dir.clone();
        fresh
    }

    /// The root mold directory.
    /// Resolution: `MOLD_HOME` env var → `~/.mold/` → `./.mold` (if HOME unset).
    pub fn mold_dir() -> Option<PathBuf> {
        if let Ok(home) = std::env::var("MOLD_HOME") {
            return Some(PathBuf::from(home));
        }
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
        if let Some(models_dir) = RUNTIME_MODELS_DIR_OVERRIDE.get() {
            return models_dir.clone();
        }
        if let Ok(env_dir) = std::env::var("MOLD_MODELS_DIR") {
            PathBuf::from(env_dir)
        } else {
            let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            let expanded = self.models_dir.replace("~", &home.to_string_lossy());
            PathBuf::from(expanded)
        }
    }

    pub fn has_models_dir_override(&self) -> bool {
        RUNTIME_MODELS_DIR_OVERRIDE.get().is_some() || std::env::var_os("MOLD_MODELS_DIR").is_some()
    }

    /// Resolve the effective default model with idiot-proof fallback chain:
    /// 1. `MOLD_DEFAULT_MODEL` env var (if set and non-empty)
    /// 2. Config file `default_model` (if that model has a custom `[models]` entry)
    /// 3. Config file `default_model` (if that model is a known manifest model that is downloaded)
    /// 4. Last-used model from `$MOLD_HOME/last-model` (if downloaded)
    /// 5. If exactly one model is downloaded, use it automatically
    /// 6. Fall back to config value (will trigger auto-pull on use)
    pub fn resolved_default_model(&self) -> String {
        self.resolve_default_model().model
    }

    /// Like [`resolved_default_model`] but also returns which fallback step resolved it.
    pub fn resolve_default_model(&self) -> DefaultModelResolution {
        // 1. Env var override
        if let Ok(m) = std::env::var("MOLD_DEFAULT_MODEL") {
            if !m.is_empty() {
                return DefaultModelResolution {
                    model: m,
                    source: DefaultModelSource::EnvVar,
                };
            }
        }
        // 2. Explicit config entry — honor custom/manual models even when not manifest-backed.
        let configured = &self.default_model;
        if self.lookup_model_config(configured).is_some() {
            return DefaultModelResolution {
                model: configured.clone(),
                source: DefaultModelSource::ConfigCustomEntry,
            };
        }
        // 3. Configured manifest model — if downloaded
        if self.manifest_model_is_downloaded(configured) {
            return DefaultModelResolution {
                model: configured.clone(),
                source: DefaultModelSource::Config,
            };
        }
        // 4. Last-used model — if still downloaded
        if let Some(last) = Self::read_last_model() {
            if self.manifest_model_is_downloaded(&last) {
                return DefaultModelResolution {
                    model: last,
                    source: DefaultModelSource::LastUsed,
                };
            }
        }
        // 5. Single downloaded model (exclude utility models like qwen3-expand)
        let downloaded: Vec<String> = crate::manifest::known_manifests()
            .iter()
            .filter(|m| !m.is_utility() && self.manifest_model_is_downloaded(&m.name))
            .map(|m| m.name.clone())
            .collect();
        if downloaded.len() == 1 {
            return DefaultModelResolution {
                model: downloaded.into_iter().next().unwrap(),
                source: DefaultModelSource::OnlyDownloaded,
            };
        }
        // 6. Config default (will auto-pull) — resolve bare names like
        //    "flux2-klein" → "flux2-klein:q8" so the TUI/CLI show the real tag.
        DefaultModelResolution {
            model: crate::manifest::resolve_model_name(configured),
            source: DefaultModelSource::ConfigDefault,
        }
    }

    /// Path to the last-model state file: `$MOLD_HOME/last-model`
    fn last_model_path() -> Option<PathBuf> {
        Self::mold_dir().map(|d| d.join("last-model"))
    }

    /// Read the last-used model name from the state file.
    pub fn read_last_model() -> Option<String> {
        let path = Self::last_model_path()?;
        std::fs::read_to_string(path).ok().and_then(|s| {
            let trimmed = s.trim().to_string();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        })
    }

    /// Write the last-used model name to the state file (best-effort, non-fatal).
    pub fn write_last_model(model: &str) {
        if let Some(path) = Self::last_model_path() {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, model);
        }
    }

    /// Resolve the output directory for server-mode image persistence.
    /// `MOLD_OUTPUT_DIR` env var takes precedence over the config file value.
    /// Returns `None` when disabled (default).
    pub fn resolved_output_dir(&self) -> Option<PathBuf> {
        let raw = if let Ok(env_dir) = std::env::var("MOLD_OUTPUT_DIR") {
            if env_dir.is_empty() {
                None
            } else {
                Some(env_dir)
            }
        } else {
            self.output_dir.clone().filter(|s| !s.is_empty())
        };
        raw.map(|dir| {
            let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            if dir == "~" {
                home
            } else if let Some(rest) = dir.strip_prefix("~/") {
                home.join(rest)
            } else {
                PathBuf::from(dir)
            }
        })
    }

    /// Check if image output has been explicitly disabled by the user
    /// (empty `MOLD_OUTPUT_DIR` env var or empty `output_dir` config field).
    pub fn is_output_disabled(&self) -> bool {
        if let Ok(env_dir) = std::env::var("MOLD_OUTPUT_DIR") {
            return env_dir.is_empty();
        }
        matches!(self.output_dir.as_deref(), Some(""))
    }

    /// Resolved output directory with a default fallback to `~/.mold/output/`.
    /// Unlike `resolved_output_dir()`, this always returns a path.
    pub fn effective_output_dir(&self) -> PathBuf {
        self.resolved_output_dir().unwrap_or_else(|| {
            Self::mold_dir()
                .unwrap_or_else(|| PathBuf::from(".mold"))
                .join("output")
        })
    }

    /// Resolved log directory from config or default (~/.mold/logs/).
    pub fn resolved_log_dir(&self) -> PathBuf {
        if let Some(ref dir) = self.logging.dir {
            let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            if dir == "~" {
                home
            } else if let Some(rest) = dir.strip_prefix("~/") {
                home.join(rest)
            } else {
                PathBuf::from(dir)
            }
        } else {
            Self::mold_dir()
                .unwrap_or_else(|| PathBuf::from(".mold"))
                .join("logs")
        }
    }

    pub fn effective_embed_metadata(&self, override_value: Option<bool>) -> bool {
        if let Some(value) = override_value {
            return value;
        }

        match std::env::var("MOLD_EMBED_METADATA") {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => true,
                "0" | "false" | "no" | "off" => false,
                _ => {
                    eprintln!(
                        "warning: invalid MOLD_EMBED_METADATA value '{value}' — using config/default"
                    );
                    self.embed_metadata
                }
            },
            Err(_) => self.embed_metadata,
        }
    }

    pub fn discovered_manifest_paths(&self, name: &str) -> Option<ModelPaths> {
        if crate::download::has_pulling_marker(name) {
            return None;
        }
        let manifest = crate::manifest::find_manifest(name)?;
        let models_dir = self.resolved_models_dir();
        let downloads = manifest
            .files
            .iter()
            .map(|file| {
                let path = models_dir.join(crate::manifest::storage_path(manifest, file));
                path.exists().then_some((file.component, path))
            })
            .collect::<Option<Vec<_>>>()?;
        crate::manifest::paths_from_downloads(&downloads)
    }

    pub fn manifest_model_is_downloaded(&self, name: &str) -> bool {
        if crate::download::has_pulling_marker(name) {
            return false;
        }
        self.resolved_local_manifest_model_config(name).is_some()
    }

    /// Return the ModelConfig for a given model name, or an empty default.
    /// Tries the exact name first, then the canonical `name:tag` form.
    pub fn model_config(&self, name: &str) -> ModelConfig {
        let mut cfg = self.lookup_model_config(name).unwrap_or_default();

        if let Some(discovered) = self.resolved_local_manifest_model_config(name) {
            overlay_model_paths(&mut cfg, &discovered);
            if cfg.description.is_none() {
                cfg.description = discovered.description;
            }
            if cfg.family.is_none() {
                cfg.family = discovered.family;
            }
        }

        cfg
    }

    /// Return a model config merged with manifest defaults and metadata.
    pub fn resolved_model_config(&self, name: &str) -> ModelConfig {
        let mut cfg = self.model_config(name);

        if let Some(manifest) = crate::manifest::find_manifest(name) {
            // Manifest provides defaults when the config file doesn't specify them.
            // Since to_model_config() no longer writes manifest defaults to config,
            // config values are only Some when the user explicitly set them.
            // User overrides are preserved; manifest fills in the rest.
            if cfg.default_steps.is_none() {
                cfg.default_steps = Some(manifest.defaults.steps);
            }
            if cfg.default_guidance.is_none() {
                cfg.default_guidance = Some(manifest.defaults.guidance);
            }
            if cfg.default_width.is_none() {
                cfg.default_width = Some(manifest.defaults.width);
            }
            if cfg.default_height.is_none() {
                cfg.default_height = Some(manifest.defaults.height);
            }
            if cfg.is_schnell.is_none() {
                cfg.is_schnell = Some(manifest.defaults.is_schnell);
            }
            if cfg.scheduler.is_none() {
                cfg.scheduler = manifest.defaults.scheduler;
            }
            if cfg.negative_prompt.is_none() {
                cfg.negative_prompt = manifest.defaults.negative_prompt.clone();
            }
            // Description and family always come from the manifest for known models.
            // These are metadata, not user-configurable settings.
            cfg.description = Some(manifest.description.clone());
            cfg.family = Some(manifest.family.clone());
        }

        cfg
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

    /// Look up a model config entry by name (exact or canonical `name:tag` form).
    /// Public so CLI commands can check whether a model has a custom config entry.
    pub fn lookup_model_config(&self, name: &str) -> Option<ModelConfig> {
        if let Some(cfg) = self.models.get(name) {
            return Some(cfg.clone());
        }
        let canonical = resolve_model_name(name);
        if canonical != name {
            return self.models.get(&canonical).cloned();
        }
        None
    }

    fn discovered_manifest_model_config(&self, name: &str) -> Option<ModelConfig> {
        let manifest = crate::manifest::find_manifest(name)?;
        let paths = self.discovered_manifest_paths(name)?;
        Some(manifest.to_model_config(&paths))
    }

    fn resolved_local_manifest_model_config(&self, name: &str) -> Option<ModelConfig> {
        let manifest = crate::manifest::find_manifest(name)?;
        let paths = ModelPaths::resolve(name, self)?;
        Some(manifest.to_model_config(&paths))
    }
}

fn overlay_model_paths(target: &mut ModelConfig, source: &ModelConfig) {
    target.transformer = source.transformer.clone();
    target.transformer_shards = source.transformer_shards.clone();
    target.vae = source.vae.clone();

    if source.t5_encoder.is_some() {
        target.t5_encoder = source.t5_encoder.clone();
    }
    if source.clip_encoder.is_some() {
        target.clip_encoder = source.clip_encoder.clone();
    }
    if source.t5_tokenizer.is_some() {
        target.t5_tokenizer = source.t5_tokenizer.clone();
    }
    if source.clip_tokenizer.is_some() {
        target.clip_tokenizer = source.clip_tokenizer.clone();
    }
    if source.clip_encoder_2.is_some() {
        target.clip_encoder_2 = source.clip_encoder_2.clone();
    }
    if source.clip_tokenizer_2.is_some() {
        target.clip_tokenizer_2 = source.clip_tokenizer_2.clone();
    }
    if source.text_encoder_files.is_some() {
        target.text_encoder_files = source.text_encoder_files.clone();
    }
    if source.text_tokenizer.is_some() {
        target.text_tokenizer = source.text_tokenizer.clone();
    }
    if source.decoder.is_some() {
        target.decoder = source.decoder.clone();
    }
}
