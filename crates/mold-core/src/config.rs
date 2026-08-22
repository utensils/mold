use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::expand::ExpandSettings;
use crate::manifest::resolve_model_name;
use crate::types::Scheduler;

static RUNTIME_MODELS_DIR_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

/// Banner comment written at the top of a `save_bootstrap_only` output so
/// readers understand why the usual user-preference fields are missing.
const BOOTSTRAP_ONLY_BANNER: &str = "\
# mold config — bootstrap-only surface.
#
# User preferences (expand.*, scheduler.*, gallery.*, generate.default_*, per-model
# generation defaults, LoRA, and generation scheduler) live in SQLite at
# <MOLD_HOME>/mold.db.
# Edit them via:
#   mold config set <key> <value>
#   mold config get <key>
#   mold config reset <key>     # drop the DB row, fall back to this file
#
# This file retains only identifiers, paths, ports, credentials, logging,
# and per-model file-path entries. Adding generation defaults here is
# silently overridden by the DB on load.

";

/// Global + per-model keys moved to the DB by issue #265. Stripped from
/// the TOML value tree by [`Config::save_bootstrap_only_to`].
const STRIPPED_GLOBAL_KEYS: &[&str] = &[
    "default_width",
    "default_height",
    "default_steps",
    "embed_metadata",
    "default_negative_prompt",
    "t5_variant",
    "umt5_variant",
    "qwen3_variant",
    "expand",
    "scheduler",
    "gallery",
    "generate",
];

const STRIPPED_MODEL_KEYS: &[&str] = &[
    "default_steps",
    "default_guidance",
    "default_width",
    "default_height",
    "scheduler",
    "negative_prompt",
    "lora",
    "lora_scale",
    "default_frames",
    "default_fps",
];

fn strip_user_pref_fields(doc: &mut toml::Value) {
    let Some(table) = doc.as_table_mut() else {
        return;
    };
    for key in STRIPPED_GLOBAL_KEYS {
        table.remove(*key);
    }
    if let Some(toml::Value::Table(models)) = table.get_mut("models") {
        for (_, mc) in models.iter_mut() {
            if let Some(mc_table) = mc.as_table_mut() {
                for key in STRIPPED_MODEL_KEYS {
                    mc_table.remove(*key);
                }
            }
        }
    }
}

/// Hook installed by callers (typically `mold-cli` at startup) that wants
/// to overlay DB-backed user preferences onto every freshly-loaded
/// `Config`. `mold-core` itself must not depend on `mold-db`, so the hook
/// is just an opaque function pointer registered once per process.
///
/// Runs after TOML parsing + legacy v0→v1 migration and before the
/// returned value is handed to the caller. Errors should be swallowed
/// inside the hook (the DB layer logs); failing here must never break
/// `load_or_default()`.
pub type ConfigPostLoadHook = fn(&mut Config);
static POST_LOAD_HOOK: OnceLock<ConfigPostLoadHook> = OnceLock::new();

/// Register a post-load hook. First caller wins; subsequent calls are a
/// no-op so tests can't clobber each other.
pub fn install_post_load_hook(hook: ConfigPostLoadHook) {
    let _ = POST_LOAD_HOOK.set(hook);
}

/// Hook for [`Config::read_last_model`] that lets the DB layer provide
/// the value without `mold-core` depending on `mold-db`. When installed,
/// the hook's return value takes precedence over the legacy sidecar file.
pub type ReadLastModelHook = fn() -> Option<String>;
static READ_LAST_MODEL_HOOK: OnceLock<ReadLastModelHook> = OnceLock::new();

/// Register a read-last-model hook. First caller wins.
pub fn install_read_last_model_hook(hook: ReadLastModelHook) {
    let _ = READ_LAST_MODEL_HOOK.set(hook);
}

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
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct ModelConfig {
    // --- paths ---
    pub transformer: Option<String>,
    /// Multi-shard transformer paths (Z-Image BF16); empty means use single `transformer`
    pub transformer_shards: Option<Vec<String>>,
    /// Low-noise expert of a two-expert checkpoint pair (Wan 2.2 A14B); the
    /// high-noise expert is `transformer`.
    pub low_noise_transformer: Option<String>,
    pub vae: Option<String>,
    /// LTX latent upsampler / spatial upscaler weights.
    pub spatial_upscaler: Option<String>,
    /// Optional temporal upscaler weights for LTX-2/LTX-2.3.
    pub temporal_upscaler: Option<String>,
    /// Optional distilled LoRA bundled with a model manifest.
    pub distilled_lora: Option<String>,
    /// Distilled LoRA for `low_noise_transformer`.
    pub low_noise_distilled_lora: Option<String>,
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
    /// Default number of video frames for video models (e.g. 25 for ltx-video).
    pub default_frames: Option<u32>,
    /// Default video FPS for video models (e.g. 24 for ltx-video).
    pub default_fps: Option<u32>,

    // --- metadata ---
    pub description: Option<String>,
    pub family: Option<String>,

    /// Per-component device placement override. `None` preserves the
    /// engine's VRAM-aware auto-placement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<crate::types::DevicePlacement>,
}

impl ModelConfig {
    /// Collect all file path strings from this model config into a flat list.
    /// Used for reference counting when determining which files are shared.
    pub fn all_file_paths(&self) -> Vec<String> {
        let mut paths = Vec::new();
        let singles = [
            &self.transformer,
            &self.low_noise_transformer,
            &self.vae,
            &self.spatial_upscaler,
            &self.temporal_upscaler,
            &self.distilled_lora,
            &self.low_noise_distilled_lora,
            &self.t5_encoder,
            &self.clip_encoder,
            &self.t5_tokenizer,
            &self.clip_tokenizer,
            &self.clip_encoder_2,
            &self.clip_tokenizer_2,
            &self.text_tokenizer,
            &self.decoder,
            &self.lora,
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

    /// Effective video frames: per-model default, or None for image-only models.
    pub fn effective_frames(&self) -> Option<u32> {
        self.default_frames
    }

    /// Effective video FPS: per-model default, or None for image-only models.
    pub fn effective_fps(&self) -> Option<u32> {
        self.default_fps
    }
}

/// Resolved model file paths.
/// For diffusion models, `transformer` and `vae` are always required.
/// For upscaler models, only `transformer` (weights) is required; `vae` is empty.
/// For utility models, only `transformer` is required; `vae` may be empty.
/// Other paths are optional — each engine validates what it needs at load time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPaths {
    pub transformer: PathBuf,
    /// Multi-shard transformer paths (Z-Image BF16); empty means use single `transformer`
    pub transformer_shards: Vec<PathBuf>,
    /// The low-noise expert of a two-expert checkpoint pair (Wan 2.2 A14B).
    ///
    /// `Some` is the two-expert predicate. The high-noise expert lives in
    /// `transformer` because it runs first — the sampler starts at high sigma —
    /// so every consumer that only understands one transformer keeps working
    /// against the one that opens the generation.
    pub low_noise_transformer: Option<PathBuf>,
    pub vae: PathBuf,
    pub spatial_upscaler: Option<PathBuf>,
    pub temporal_upscaler: Option<PathBuf>,
    pub distilled_lora: Option<PathBuf>,
    /// The distilled adapter for `low_noise_transformer`. Each expert of an
    /// A14B pair is distilled separately; swapping them is not a degradation,
    /// it is the wrong model.
    pub low_noise_distilled_lora: Option<PathBuf>,
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
    const FROZEN_MODEL_PREFIX: &'static str = "\0mold-frozen-chain:";

    /// Every resolved local artifact that can participate in engine creation.
    /// Policy fences consume this before inspecting or loading any artifact.
    pub fn all_file_paths(&self) -> Vec<&Path> {
        let mut paths = vec![self.transformer.as_path(), self.vae.as_path()];
        paths.extend(self.transformer_shards.iter().map(PathBuf::as_path));
        paths.extend(self.low_noise_transformer.as_deref());
        paths.extend(self.spatial_upscaler.as_deref());
        paths.extend(self.temporal_upscaler.as_deref());
        paths.extend(self.distilled_lora.as_deref());
        paths.extend(self.low_noise_distilled_lora.as_deref());
        paths.extend(self.t5_encoder.as_deref());
        paths.extend(self.clip_encoder.as_deref());
        paths.extend(self.t5_tokenizer.as_deref());
        paths.extend(self.clip_tokenizer.as_deref());
        paths.extend(self.clip_encoder_2.as_deref());
        paths.extend(self.clip_tokenizer_2.as_deref());
        paths.extend(self.text_encoder_files.iter().map(PathBuf::as_path));
        paths.extend(self.text_tokenizer.as_deref());
        paths.extend(self.decoder.as_deref());
        paths
    }

    /// Resolve paths for a model. Checks config, then env vars.
    /// Returns None if transformer and VAE paths can't be resolved.
    /// All other paths are optional (depend on model family).
    pub fn resolve(model_name: &str, config: &Config) -> Option<Self> {
        if let Some(model_cfg) = config
            .models
            .get(&format!("{}{model_name}", Self::FROZEN_MODEL_PREFIX))
        {
            return Self::resolve_from_model_config_exact(model_cfg);
        }
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

    /// Resolve only the paths captured in `model_cfg`.
    ///
    /// This intentionally does not consult `MOLD_*_PATH` environment variables,
    /// manifests, sidecars, or model discovery. Durable chain jobs use it after
    /// admission so a restart cannot silently substitute different artifacts.
    pub fn resolve_from_model_config_exact(model_cfg: &ModelConfig) -> Option<Self> {
        let path = |value: Option<&str>| value.map(PathBuf::from);
        let vae = match model_cfg.vae.as_deref().filter(|path| !path.is_empty()) {
            Some(path) => PathBuf::from(path),
            None if model_cfg.family.as_deref().is_some_and(|family| {
                family == "ltx2"
                    || crate::manifest::UTILITY_FAMILIES.contains(&family)
                    || crate::manifest::UPSCALER_FAMILIES.contains(&family)
            }) =>
            {
                PathBuf::new()
            }
            None => return None,
        };
        Some(Self {
            transformer: PathBuf::from(model_cfg.transformer.as_deref()?),
            transformer_shards: model_cfg
                .transformer_shards
                .as_ref()
                .map(|paths| paths.iter().map(PathBuf::from).collect())
                .unwrap_or_default(),
            low_noise_transformer: path(model_cfg.low_noise_transformer.as_deref()),
            vae,
            spatial_upscaler: path(model_cfg.spatial_upscaler.as_deref()),
            temporal_upscaler: path(model_cfg.temporal_upscaler.as_deref()),
            distilled_lora: path(model_cfg.distilled_lora.as_deref()),
            low_noise_distilled_lora: path(model_cfg.low_noise_distilled_lora.as_deref()),
            t5_encoder: path(model_cfg.t5_encoder.as_deref()),
            clip_encoder: path(model_cfg.clip_encoder.as_deref()),
            t5_tokenizer: path(model_cfg.t5_tokenizer.as_deref()),
            clip_tokenizer: path(model_cfg.clip_tokenizer.as_deref()),
            clip_encoder_2: path(model_cfg.clip_encoder_2.as_deref()),
            clip_tokenizer_2: path(model_cfg.clip_tokenizer_2.as_deref()),
            text_encoder_files: model_cfg
                .text_encoder_files
                .as_ref()
                .map(|paths| paths.iter().map(PathBuf::from).collect())
                .unwrap_or_default(),
            text_tokenizer: path(model_cfg.text_tokenizer.as_deref()),
            decoder: path(model_cfg.decoder.as_deref()),
        })
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
        let low_noise_transformer = Self::resolve_path(
            model_cfg.and_then(|m| m.low_noise_transformer.as_deref()),
            "MOLD_LOW_NOISE_TRANSFORMER_PATH",
        );
        let vae = Self::resolve_path(model_cfg.and_then(|m| m.vae.as_deref()), "MOLD_VAE_PATH")?;
        let spatial_upscaler = Self::resolve_path(
            model_cfg.and_then(|m| m.spatial_upscaler.as_deref()),
            "MOLD_SPATIAL_UPSCALER_PATH",
        );
        let temporal_upscaler = Self::resolve_path(
            model_cfg.and_then(|m| m.temporal_upscaler.as_deref()),
            "MOLD_TEMPORAL_UPSCALER_PATH",
        );
        let distilled_lora = Self::resolve_path(
            model_cfg.and_then(|m| m.distilled_lora.as_deref()),
            "MOLD_DISTILLED_LORA_PATH",
        );
        let low_noise_distilled_lora = Self::resolve_path(
            model_cfg.and_then(|m| m.low_noise_distilled_lora.as_deref()),
            "MOLD_LOW_NOISE_DISTILLED_LORA_PATH",
        );
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
            low_noise_transformer,
            vae,
            spatial_upscaler,
            temporal_upscaler,
            distilled_lora,
            low_noise_distilled_lora,
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
    /// Quantized UMT5 encoder variant for Wan (`q8`, `q6`, `q5`, `fp16`,
    /// `auto`). Wan's encoder is 11.4 GB at FP16 and is the floor of every
    /// wan render's memory estimate.
    pub umt5_variant: Option<String>,

    /// Preferred Qwen3 text encoder variant: "bf16" (default), "q8", "q6", "iq4", "q3", or "auto".
    /// "auto" selects the best variant that fits in GPU VRAM (with drop-and-reload).
    #[serde(default)]
    pub qwen3_variant: Option<String>,

    /// Directory to persist generated images. Default: `~/.mold/output/`.
    /// Override with `MOLD_OUTPUT_DIR` env var. Set to empty string to disable
    /// (TUI gallery will not function when disabled).
    #[serde(default)]
    pub output_dir: Option<String>,

    /// Allow roots for trusted server-local media request paths.
    /// Override with `MOLD_MEDIA_ROOTS` using the platform path-list separator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_roots: Option<Vec<String>>,

    /// Global default negative prompt for CFG-based models (SD1.5, SDXL, SD3, Wuerstchen).
    /// Overridden by per-model `negative_prompt` or CLI `--negative-prompt`.
    #[serde(default)]
    pub default_negative_prompt: Option<String>,

    /// Prompt expansion settings.
    #[serde(default)]
    pub expand: ExpandSettings,

    /// Profile-scoped scheduler behavior. Persisted in `mold.db`; the
    /// serialized field exists only for one-shot import of older/manual TOML.
    #[serde(default)]
    pub scheduler: SchedulerSettings,

    /// Profile-scoped gallery behavior (trash retention). Persisted in
    /// `mold.db` like `scheduler`; the serialized field exists only for
    /// one-shot import of a hand-written `[gallery]` TOML section.
    #[serde(default)]
    pub gallery: GallerySettings,

    /// Profile-scoped generation behavior that has no flat legacy key name
    /// (currently just `generate.auto_tag_title`). Persisted in `mold.db`
    /// like `gallery`; the serialized field exists only for one-shot import
    /// of a hand-written `[generate]` TOML section.
    #[serde(default)]
    pub generate: GenerateSettings,

    /// Logging configuration.
    #[serde(default)]
    pub logging: LoggingConfig,

    /// RunPod integration settings (api key, defaults, auto-teardown behaviour).
    #[serde(default)]
    pub runpod: crate::runpod::RunPodSettings,

    /// Lambda Cloud integration settings.
    #[serde(default)]
    pub lambda: crate::lambda::LambdaSettings,

    /// GPUs to use at startup (None = all visible).
    ///
    /// Accepts legacy ordinal arrays, stable/NVIDIA UUID string arrays, and
    /// explicit `"all"` / `"none"` keywords.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpus: Option<crate::types::GpuSelection>,

    /// Max queued requests before 503 (default: 200).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_size: Option<usize>,

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

pub const SCHEDULER_TIMING_MAX_MS: u32 = 30_000;

/// Upper bound for `gallery.trash_retention_days` (ten years).
pub const GALLERY_TRASH_RETENTION_MAX_DAYS: u32 = 3650;

/// Generation behaviour that is a user preference rather than bootstrap
/// state, and that does not already have a flat legacy key name.
///
/// Deliberately separate from the older flat `default_width` /
/// `embed_metadata` globals: those keep their historical CLI spelling for
/// backwards compatibility even though they persist under `generate.*` in the
/// DB. New generation preferences are named `generate.*` on both surfaces.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct GenerateSettings {
    /// Whether a titled print is also tagged with its title slug, by the
    /// client that submits it. Default on.
    ///
    /// This is a CLIENT-side default for the CLI and TUI — the server never
    /// auto-tags, because it cannot tell a title the user typed from one a
    /// script generated, and a host silently adding tags to every print that
    /// crosses it would be surprising from any other machine. `mold run
    /// --no-auto-tag` overrides it per invocation.
    #[serde(default = "default_auto_tag_title")]
    pub auto_tag_title: bool,
}

const fn default_auto_tag_title() -> bool {
    true
}

impl Default for GenerateSettings {
    fn default() -> Self {
        Self {
            auto_tag_title: default_auto_tag_title(),
        }
    }
}

/// Gallery behaviour that is a user preference rather than bootstrap state.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct GallerySettings {
    /// Days a trashed print survives in `<output_dir>/.trash/` before the
    /// retention sweeper purges it. `0` keeps trashed prints forever.
    /// Default 30. Env override: `MOLD_GALLERY_TRASH_RETENTION_DAYS`.
    #[serde(default = "default_trash_retention_days")]
    pub trash_retention_days: u32,
}

const fn default_trash_retention_days() -> u32 {
    30
}

impl Default for GallerySettings {
    fn default() -> Self {
        Self {
            trash_retention_days: default_trash_retention_days(),
        }
    }
}

impl GallerySettings {
    /// Name of the env var that overrides `trash_retention_days`.
    pub const TRASH_RETENTION_DAYS_ENV: &'static str = "MOLD_GALLERY_TRASH_RETENTION_DAYS";

    /// Effective retention in days: the `MOLD_GALLERY_TRASH_RETENTION_DAYS`
    /// env var when it holds a valid value in `0..=3650`, else the stored
    /// setting. Invalid env values warn once per call and fall through, the
    /// same contract `Config::effective_embed_metadata` keeps.
    pub fn effective_trash_retention_days(&self) -> u32 {
        match std::env::var(Self::TRASH_RETENTION_DAYS_ENV) {
            Ok(value) => match value.trim().parse::<u32>() {
                Ok(days) if days <= GALLERY_TRASH_RETENTION_MAX_DAYS => days,
                _ => {
                    eprintln!(
                        "warning: invalid {} value '{value}' (expected 0..={}) — using config/default",
                        Self::TRASH_RETENTION_DAYS_ENV,
                        GALLERY_TRASH_RETENTION_MAX_DAYS
                    );
                    self.trash_retention_days
                }
            },
            Err(_) => self.trash_retention_days,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
pub struct SchedulerSettings {
    #[serde(default = "default_replan_debounce_ms")]
    pub replan_debounce_ms: u32,
    #[serde(default = "default_replan_max_delay_ms")]
    pub replan_max_delay_ms: u32,
    #[serde(default = "default_warm_wait_max_ms")]
    pub warm_wait_max_ms: u32,
}

impl<'de> Deserialize<'de> for SchedulerSettings {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(default = "default_replan_debounce_ms")]
            replan_debounce_ms: u32,
            #[serde(default = "default_replan_max_delay_ms")]
            replan_max_delay_ms: u32,
            #[serde(default = "default_warm_wait_max_ms")]
            warm_wait_max_ms: u32,
        }

        let wire = Wire::deserialize(deserializer)?;
        SchedulerSettings {
            replan_debounce_ms: wire.replan_debounce_ms,
            replan_max_delay_ms: wire.replan_max_delay_ms,
            warm_wait_max_ms: wire.warm_wait_max_ms,
        }
        .validate()
        .map_err(serde::de::Error::custom)
    }
}

const fn default_replan_debounce_ms() -> u32 {
    2_000
}

const fn default_replan_max_delay_ms() -> u32 {
    5_000
}

const fn default_warm_wait_max_ms() -> u32 {
    2_000
}

impl SchedulerSettings {
    pub fn validate(self) -> anyhow::Result<Self> {
        for (key, value) in [
            ("scheduler.replan_debounce_ms", self.replan_debounce_ms),
            ("scheduler.replan_max_delay_ms", self.replan_max_delay_ms),
            ("scheduler.warm_wait_max_ms", self.warm_wait_max_ms),
        ] {
            anyhow::ensure!(
                value <= SCHEDULER_TIMING_MAX_MS,
                "{key} must be between 0 and {SCHEDULER_TIMING_MAX_MS}"
            );
        }
        anyhow::ensure!(
            self.replan_max_delay_ms >= self.replan_debounce_ms,
            "scheduler.replan_max_delay_ms must be greater than or equal to \
             scheduler.replan_debounce_ms"
        );
        Ok(self)
    }
}

impl Default for SchedulerSettings {
    fn default() -> Self {
        Self {
            replan_debounce_ms: default_replan_debounce_ms(),
            replan_max_delay_ms: default_replan_max_delay_ms(),
            warm_wait_max_ms: default_warm_wait_max_ms(),
        }
    }
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
    "flux2-klein:q8".to_string()
}

fn default_models_dir() -> String {
    if let Ok(home) = std::env::var("MOLD_HOME") {
        format!("{home}/models")
    } else if let Some(home) = Config::saved_mold_dir() {
        home.join("models").to_string_lossy().into_owned()
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
            umt5_variant: None,
            qwen3_variant: None,
            output_dir: None,
            media_roots: None,
            default_negative_prompt: None,
            expand: ExpandSettings::default(),
            scheduler: SchedulerSettings::default(),
            gallery: GallerySettings::default(),
            generate: GenerateSettings::default(),
            logging: LoggingConfig::default(),
            runpod: crate::runpod::RunPodSettings::default(),
            lambda: crate::lambda::LambdaSettings::default(),
            gpus: None,
            queue_size: None,
            models: HashMap::new(),
        }
    }
}

impl Config {
    /// Install an immutable model snapshot for one in-memory execution.
    ///
    /// The private sentinel makes [`ModelPaths::resolve`] choose the exact
    /// captured paths before any mutable manifest, sidecar, or environment
    /// source. The ordinary model entry is also replaced so engine-shaping
    /// defaults (family, LoRA, scheduler, dtype flags) remain frozen while the
    /// semantic model name is preserved.
    pub fn install_frozen_model_config(&mut self, model_name: &str, model: ModelConfig) {
        self.models.insert(
            format!("{}{model_name}", ModelPaths::FROZEN_MODEL_PREFIX),
            model.clone(),
        );
        self.models.insert(model_name.to_string(), model);
    }

    pub fn has_frozen_model_config(&self, model_name: &str) -> bool {
        self.models
            .contains_key(&format!("{}{model_name}", ModelPaths::FROZEN_MODEL_PREFIX))
    }

    /// Build a `GpuSelection` from the config's `gpus` field.
    pub fn gpu_selection(&self) -> crate::types::GpuSelection {
        self.gpus.clone().unwrap_or_default()
    }

    /// Return the configured queue size or the default (200).
    pub fn queue_size(&self) -> usize {
        self.queue_size.unwrap_or(200)
    }

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

        // Post-load hook (DB-backed user-pref overlay, if installed).
        if let Some(hook) = POST_LOAD_HOOK.get() {
            hook(&mut cfg);
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
                    mc.default_frames = None;
                    mc.default_fps = None;
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

    /// The root Mold directory shared by every local surface.
    /// Resolution: `MOLD_HOME` env var → saved bootstrap selection →
    /// `~/.mold/` → `./.mold` (if HOME unset).
    pub fn mold_dir() -> Option<PathBuf> {
        if let Ok(home) = std::env::var("MOLD_HOME") {
            return Some(PathBuf::from(home));
        }
        if let Some(home) = Self::saved_mold_dir() {
            return Some(home);
        }
        Some(
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".mold"),
        )
    }

    /// Location of the tiny bootstrap pointer used to find a non-default
    /// Mold home before config.toml or mold.db can be opened.
    pub fn mold_home_pointer_path() -> Option<PathBuf> {
        if let Some(path) = std::env::var_os("MOLD_HOME_POINTER_PATH") {
            return Some(PathBuf::from(path));
        }
        dirs::config_dir().map(|dir| dir.join("mold").join("home"))
    }

    pub fn saved_mold_dir() -> Option<PathBuf> {
        Self::read_saved_mold_dir().ok().flatten()
    }

    /// Read the bootstrap pointer without treating corruption like absence.
    /// `NotFound` means the default root is still authoritative; every other
    /// malformed/unreadable state must fail closed at process startup.
    pub fn read_saved_mold_dir() -> std::io::Result<Option<PathBuf>> {
        let Some(pointer) = Self::mold_home_pointer_path() else {
            return Ok(None);
        };
        let raw = match std::fs::read_to_string(&pointer) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let value = raw.trim();
        if value.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{} is empty", pointer.display()),
            ));
        }
        let path = PathBuf::from(value);
        if !path.is_absolute() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{} does not contain an absolute path", pointer.display()),
            ));
        }
        Ok(Some(path))
    }

    /// Refuse to initialize a missing saved root. A saved selection is only
    /// written after Desktop has created or validated it, so disappearance
    /// means an external drive is unavailable rather than a new root that a
    /// CLI/server process should recreate. Explicit `MOLD_HOME` remains
    /// allowed to name a directory that the invoking process intends to make.
    pub fn ensure_saved_mold_dir_available() -> anyhow::Result<()> {
        if std::env::var_os("MOLD_HOME").is_some() {
            return Ok(());
        }
        let saved = Self::read_saved_mold_dir().with_context(|| {
            "the saved Mold home selection is unreadable or invalid; repair it in Mold Desktop Settings"
        })?;
        if let Some(path) = saved.filter(|path| !path.is_dir()) {
            anyhow::bail!(
                "the saved Mold home at {} is unavailable; reconnect its drive or change the location in Mold Desktop Settings",
                path.display()
            );
        }
        Ok(())
    }

    /// Atomically save the shared local Mold-home selection. Environment
    /// `MOLD_HOME` remains the highest-precedence, process-scoped override.
    pub fn save_mold_dir(path: &Path) -> std::io::Result<()> {
        let pointer = Self::mold_home_pointer_path().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "could not resolve the Mold home bootstrap location",
            )
        })?;
        if let Some(parent) = pointer.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temp = pointer.with_extension("tmp");
        std::fs::write(&temp, path.to_string_lossy().as_bytes())?;
        match std::fs::rename(&temp, &pointer) {
            Ok(()) => Ok(()),
            Err(_error) if pointer.exists() => {
                // Windows does not replace an existing destination with
                // rename. The temp file is already complete, so use the
                // narrow non-atomic fallback required by that platform.
                std::fs::remove_file(&pointer)?;
                std::fs::rename(temp, pointer)
            }
            Err(error) => Err(error),
        }
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
        } else if self.models_dir == "~/.mold/models" {
            Self::mold_dir()
                .unwrap_or_else(|| PathBuf::from(".mold"))
                .join("models")
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
        // 5. Single downloaded model. Only a model that can actually be the
        //    primary counts — utility, upscaler, and auxiliary bundles (a
        //    hidden LTX-2 adapter, PuLID's identity assets) are installable but
        //    never generate, so one of them on disk must not become the
        //    "only downloaded model".
        let downloaded: Vec<String> = crate::manifest::known_manifests()
            .iter()
            .filter(|m| m.is_generation_model() && self.manifest_model_is_downloaded(&m.name))
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

    /// Read the last-used model. When the DB-backed read hook is
    /// installed (production path), defers to it entirely; otherwise
    /// reads the legacy `$MOLD_HOME/last-model` sidecar. Callers doing
    /// one-shot sidecar migration should use
    /// [`Self::read_last_model_from_sidecar`] directly.
    pub fn read_last_model() -> Option<String> {
        if let Some(hook) = READ_LAST_MODEL_HOOK.get() {
            return hook();
        }
        Self::read_last_model_from_sidecar()
    }

    /// Read the legacy `$MOLD_HOME/last-model` sidecar directly. Used by
    /// the one-shot `config.toml + sidecar → DB` migration.
    pub fn read_last_model_from_sidecar() -> Option<String> {
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

    pub fn resolved_media_roots(&self) -> Vec<PathBuf> {
        if let Ok(roots) = std::env::var("MOLD_MEDIA_ROOTS") {
            return crate::parse_media_roots_env(&roots);
        }
        self.media_roots
            .as_deref()
            .map(crate::configured_media_roots)
            .unwrap_or_default()
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
        let manifest = crate::manifest::find_manifest(name)?;
        if self.incomplete_pull_blocks_manifest(manifest) {
            return None;
        }
        let models_dir = self.resolved_models_dir();
        let downloads = manifest
            .files
            .iter()
            .map(|file| {
                // Prefer a canonical clean-path hit (or a documented legacy
                // path) under the models dir.
                let local = crate::manifest::storage_path_candidates(manifest, file)
                    .into_iter()
                    .map(|path| models_dir.join(path))
                    // Same completeness rules as `manifest_files_exist`:
                    // marker present OR size matches manifest. Plain
                    // `.exists()` here let truncated downloads masquerade
                    // as installed models — the gallery race lived here.
                    .find(|path| Self::file_is_complete(path, file.size_bytes));
                if let Some(path) = local {
                    return Some((file.component, path));
                }
                // Fallback (companion manifests only): walk mold's managed
                // `<models_dir>/.hf-cache/` for the same file. A previous
                // mold install may have placed companion files under a
                // different canonical layout — e.g. Gemma TE under
                // `shared/ltx2/...` from a manifest LTX-2 model install
                // vs the catalog `ltx2-te` companion expecting
                // `shared/companion/...`. Letting the layout-agnostic
                // hf-hub cache view satisfy either keeps a single Gemma
                // download serving both. Restricted to `family ==
                // "companion"` so non-companion manifests still require
                // their files at the canonical clean-path location with
                // the proper completeness guard above — that preserves
                // the "model not downloaded" branch the tests rely on.
                if manifest.family != "companion" {
                    return None;
                }
                crate::download::cached_file_path_in_mold_cache(&file.hf_repo, &file.hf_filename)
                    .map(|path| (file.component, path))
            })
            .collect::<Option<Vec<_>>>()?;
        crate::manifest::paths_from_downloads(&downloads, &manifest.family)
    }

    pub fn manifest_model_is_downloaded(&self, name: &str) -> bool {
        let manifest = match crate::manifest::find_manifest(name) {
            Some(m) => m,
            None => return false,
        };
        if self.incomplete_pull_blocks_manifest(manifest) {
            return false;
        }
        // Upscaler, utility, and files-only bundles don't produce full
        // ModelPaths (no VAE, sometimes no transformer at all). Check file
        // existence directly instead of going through ModelPaths::resolve —
        // and read the same `is_files_only_bundle` predicate the pull path
        // does, so a bundle cannot be pulled by one rule and then reported
        // missing by another.
        if manifest.is_upscaler() || manifest.is_files_only_bundle() {
            return self.manifest_files_exist(manifest);
        }
        self.resolved_local_manifest_model_config(name).is_some()
    }

    /// Return true when a known manifest-backed model is missing any required
    /// downloadable asset and should be repaired with `mold pull`.
    pub fn manifest_model_needs_download(&self, name: &str) -> bool {
        let canonical = crate::manifest::resolve_model_name(name);
        crate::manifest::find_manifest(&canonical).is_some()
            && !self.manifest_model_is_downloaded(&canonical)
    }

    /// Check whether all files for a manifest are completely on disk.
    ///
    /// Two acceptance signals (a file is considered complete if **either** holds):
    ///
    /// 1. A `.sha256-verified` sidecar exists. Written by the pull path on a
    ///    successful download — positive proof the file finished writing and,
    ///    when the manifest declared a hash, matches it.
    /// 2. The on-disk size matches the manifest's declared `size_bytes`.
    ///    Covers two legitimate cases without forcing an upgrade-day rehash:
    ///    legacy installs created before markers were written, and HF cache
    ///    symlinks pointing at fully-downloaded blobs in `~/.cache/huggingface`.
    ///
    /// Truncated / partial files reject under both signals — they have no
    /// marker (because no successful verify ever ran) and their size does not
    /// match. That's the load-bearing change for the "downloaded model
    /// sometimes doesn't show up" gallery race.
    fn manifest_files_exist(&self, manifest: &crate::manifest::ModelManifest) -> bool {
        manifest
            .files
            .iter()
            .all(|file| self.complete_manifest_file_path(manifest, file).is_some())
    }

    /// Resolve one manifest file to a complete on-disk path, or `None` when it
    /// is absent or only partially downloaded.
    ///
    /// Exposed because files-only bundles (PuLID's identity assets) have no
    /// [`ModelPaths`] to resolve through and must ask about their files one at
    /// a time — while still using exactly the acceptance rules
    /// [`Self::manifest_files_exist`] documents, so "installed" means the same
    /// thing on both routes.
    pub fn complete_manifest_file_path(
        &self,
        manifest: &crate::manifest::ModelManifest,
        file: &crate::manifest::ModelFile,
    ) -> Option<PathBuf> {
        let models_dir = self.resolved_models_dir();
        crate::manifest::storage_path_candidates(manifest, file)
            .into_iter()
            .map(|path| models_dir.join(path))
            .find(|path| Self::file_is_complete(path, file.size_bytes))
    }

    /// True when the on-disk file at `path` should be treated as a fully
    /// downloaded artifact. See [`Self::manifest_files_exist`] for the
    /// acceptance rules.
    fn file_is_complete(path: &std::path::Path, expected_size: u64) -> bool {
        if !path.exists() {
            return false;
        }
        if crate::download::has_sha256_marker(path) {
            return true;
        }
        // Marker missing — fall back to size match against the manifest.
        // Symlinks (HF cache) follow through to the target's metadata.
        match path.metadata() {
            Ok(meta) => meta.len() == expected_size,
            Err(_) => false,
        }
    }

    /// Return true when an active `.pulling` marker should block manifest discovery.
    ///
    /// If all manifest files already exist, the marker is stale (for example a
    /// prior pull finished but crashed before marker cleanup). In that case we
    /// remove it and continue with manifest-derived paths instead of falling
    /// back to potentially stale config entries.
    fn incomplete_pull_blocks_manifest(&self, manifest: &crate::manifest::ModelManifest) -> bool {
        let marker_path =
            crate::download::pulling_marker_path_in(&self.resolved_models_dir(), &manifest.name);
        if !marker_path.exists() {
            return false;
        }

        if self.manifest_files_exist(manifest) {
            let _ = std::fs::remove_file(&marker_path);
            return false;
        }

        true
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
            if cfg.default_frames.is_none() {
                cfg.default_frames = manifest.defaults.frames;
            }
            if cfg.default_fps.is_none() {
                cfg.default_fps = manifest.defaults.fps;
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

    /// Return the effective placement for a model: config entry plus env overrides.
    ///
    /// Precedence (higher wins):
    ///   1. `MOLD_PLACE_TRANSFORMER`, `MOLD_PLACE_VAE`, `MOLD_PLACE_TEXT_ENCODERS`,
    ///      `MOLD_PLACE_T5`, `MOLD_PLACE_CLIP_L`, `MOLD_PLACE_CLIP_G`,
    ///      `MOLD_PLACE_QWEN` (env overrides per-component).
    ///   2. Config file `[models."name:tag".placement]` table.
    ///   3. `None` (use engine auto).
    ///
    /// Each env var parses:
    ///   - `"auto"`    — `DeviceRef::Auto`
    ///   - `"cpu"`     — `DeviceRef::Cpu`
    ///   - `"gpu:N"`   — `DeviceRef::Gpu { ordinal: N }`
    ///   - `"gpu"`     — `DeviceRef::Gpu { ordinal: 0 }`
    pub fn resolved_placement(&self, model_name: &str) -> Option<crate::types::DevicePlacement> {
        use crate::types::DevicePlacement;

        let mut placement = self
            .lookup_model_config(model_name)
            .and_then(|mc| mc.placement);

        let env_tier1 = parse_device_ref_env("MOLD_PLACE_TEXT_ENCODERS");
        let env_transformer = parse_device_ref_env("MOLD_PLACE_TRANSFORMER");
        let env_vae = parse_device_ref_env("MOLD_PLACE_VAE");
        let env_t5 = parse_device_ref_env("MOLD_PLACE_T5");
        let env_clip_l = parse_device_ref_env("MOLD_PLACE_CLIP_L");
        let env_clip_g = parse_device_ref_env("MOLD_PLACE_CLIP_G");
        let env_qwen = parse_device_ref_env("MOLD_PLACE_QWEN");

        let any_env = env_tier1.is_some()
            || env_transformer.is_some()
            || env_vae.is_some()
            || env_t5.is_some()
            || env_clip_l.is_some()
            || env_clip_g.is_some()
            || env_qwen.is_some();

        if !any_env {
            return placement;
        }

        let mut effective: DevicePlacement = placement.unwrap_or_default();
        if let Some(r) = env_tier1 {
            effective.text_encoders = r;
        }
        let any_advanced = env_transformer.is_some()
            || env_vae.is_some()
            || env_t5.is_some()
            || env_clip_l.is_some()
            || env_clip_g.is_some()
            || env_qwen.is_some();
        if any_advanced {
            let mut adv = effective.advanced.unwrap_or_default();
            if let Some(r) = env_transformer {
                adv.transformer = r;
            }
            if let Some(r) = env_vae {
                adv.vae = r;
            }
            if let Some(r) = env_t5 {
                adv.t5 = Some(r);
            }
            if let Some(r) = env_clip_l {
                adv.clip_l = Some(r);
            }
            if let Some(r) = env_clip_g {
                adv.clip_g = Some(r);
            }
            if let Some(r) = env_qwen {
                adv.qwen = Some(r);
            }
            effective.advanced = Some(adv);
        }
        placement = Some(effective);
        placement
    }

    /// Normalize placement at the request boundary.
    ///
    /// An explicit request placement is a complete user decision, including
    /// any `Auto` fields it contains, and therefore wins wholly over
    /// environment and persisted defaults. Without a request override,
    /// `resolved_placement` applies environment over persisted values. The
    /// final fallback is an all-`Auto` placement.
    pub fn effective_placement(
        &self,
        model_name: &str,
        request: Option<&crate::types::DevicePlacement>,
    ) -> crate::types::DevicePlacement {
        request
            .cloned()
            .or_else(|| self.resolved_placement(model_name))
            .unwrap_or_default()
    }

    /// Persist a placement for `model_name`, creating the model entry if
    /// missing. `None` clears the placement (and leaves the rest of the
    /// entry intact).
    pub fn set_model_placement(
        &mut self,
        model_name: &str,
        placement: Option<crate::types::DevicePlacement>,
    ) {
        let entry = self.models.entry(model_name.to_string()).or_default();
        entry.placement = placement;
    }

    /// Write the config to disk at `config_path()`.
    ///
    /// Safety: refuses to save if `models_dir` points to a temp/test directory,
    /// which can happen when tests race on the `MOLD_HOME` env var.
    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::config_path()
            .ok_or_else(|| anyhow::anyhow!("cannot determine home directory for config path"))?;

        // Guard: refuse to persist a config with a test-temp models_dir into a
        // non-temp config path. This catches the race condition where parallel tests
        // set MOLD_HOME to /tmp/... and a config.save() writes the corrupted
        // models_dir to the user's real config file.
        // `/tmp` is only the unix spelling of "a temporary directory": on
        // Windows a tempdir lives under `%LOCALAPPDATA%\Temp`, so both halves
        // of this guard were dead there and the race it protects against was
        // unguarded. Ask the platform where temp is instead of adding a second
        // hardcoded prefix, and keep the literal `/tmp/` checks so a unix path
        // built by a test that overrode TMPDIR still matches.
        let temp_root = std::env::temp_dir();
        let under_temp = |value: &str| {
            value.contains("/tmp/")
                || Path::new(value).starts_with(&temp_root)
                || value.contains("/mold-config-test-")
                || value.contains("\\mold-config-test-")
        };
        let path_str = path.to_string_lossy();
        let is_temp_config = under_temp(&path_str);
        let has_temp_models_dir = under_temp(&self.models_dir)
            && (self.models_dir.contains("mold-") || self.models_dir.contains("mold_"));
        if has_temp_models_dir && !is_temp_config {
            eprintln!(
                "warning: refusing to save config with test models_dir ({}) to real config ({})",
                self.models_dir,
                path.display()
            );
            return Ok(());
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&path, contents)?;
        Ok(())
    }

    /// Serialize only the bootstrap/ops slice of the config to TOML:
    /// identifiers, paths, ports, credentials, logging, runpod, per-model
    /// file-path entries. User-preference fields that now live in the DB
    /// (`expand.*`, `generate.*` globals, per-model generation defaults,
    /// lora, scheduler) are stripped.
    ///
    /// Used by the post-migration rewrite to keep `config.toml` honest
    /// after the user-preference surface has moved to SQLite.
    pub fn save_bootstrap_only_to(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut doc = toml::Value::try_from(self)?;
        strip_user_pref_fields(&mut doc);
        let body = toml::to_string_pretty(&doc)?;
        let contents = format!("{BOOTSTRAP_ONLY_BANNER}{body}");
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Save the bootstrap-only slice to the default config path.
    pub fn save_bootstrap_only(&self) -> anyhow::Result<()> {
        let path = Self::config_path()
            .ok_or_else(|| anyhow::anyhow!("cannot determine home directory for config path"))?;
        self.save_bootstrap_only_to(&path)
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
        let paths = if let Some(paths) = self.discovered_manifest_paths(name) {
            paths
        } else {
            let paths = ModelPaths::resolve(name, self)?;
            if !resolved_manifest_paths_exist(manifest, &paths) {
                return None;
            }
            paths
        };
        Some(manifest.to_model_config(&paths))
    }
}

fn overlay_model_paths(target: &mut ModelConfig, source: &ModelConfig) {
    target.transformer = source.transformer.clone();
    target.transformer_shards = source.transformer_shards.clone();
    if source.low_noise_transformer.is_some() {
        target.low_noise_transformer = source.low_noise_transformer.clone();
    }
    target.vae = source.vae.clone();
    if source.spatial_upscaler.is_some() {
        target.spatial_upscaler = source.spatial_upscaler.clone();
    }
    if source.temporal_upscaler.is_some() {
        target.temporal_upscaler = source.temporal_upscaler.clone();
    }
    if source.distilled_lora.is_some() {
        target.distilled_lora = source.distilled_lora.clone();
    }
    if source.low_noise_distilled_lora.is_some() {
        target.low_noise_distilled_lora = source.low_noise_distilled_lora.clone();
    }

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

fn resolved_manifest_paths_exist(
    manifest: &crate::manifest::ModelManifest,
    paths: &ModelPaths,
) -> bool {
    use crate::manifest::ModelComponent;

    let mut transformer_shard_idx = 0usize;
    let mut text_encoder_idx = 0usize;

    manifest.files.iter().all(|file| match file.component {
        ModelComponent::Transformer => paths.transformer.exists(),
        ModelComponent::TransformerShard => {
            let path = paths.transformer_shards.get(transformer_shard_idx);
            transformer_shard_idx += 1;
            path.is_some_and(|path| path.exists())
        }
        ModelComponent::Vae => paths.vae.exists(),
        ModelComponent::SpatialUpscaler => paths
            .spatial_upscaler
            .as_ref()
            .is_some_and(|path| path.exists()),
        ModelComponent::TemporalUpscaler => paths
            .temporal_upscaler
            .as_ref()
            .is_some_and(|path| path.exists()),
        ModelComponent::LowNoiseTransformer => paths
            .low_noise_transformer
            .as_ref()
            .is_some_and(|path| path.exists()),
        ModelComponent::DistilledLora => paths
            .distilled_lora
            .as_ref()
            .is_some_and(|path| path.exists()),
        ModelComponent::LowNoiseDistilledLora => paths
            .low_noise_distilled_lora
            .as_ref()
            .is_some_and(|path| path.exists()),
        ModelComponent::T5Encoder => paths.t5_encoder.as_ref().is_some_and(|path| path.exists()),
        ModelComponent::ClipEncoder => paths
            .clip_encoder
            .as_ref()
            .is_some_and(|path| path.exists()),
        ModelComponent::T5Tokenizer => paths
            .t5_tokenizer
            .as_ref()
            .is_some_and(|path| path.exists()),
        ModelComponent::ClipTokenizer => paths
            .clip_tokenizer
            .as_ref()
            .is_some_and(|path| path.exists()),
        ModelComponent::ClipEncoder2 => paths
            .clip_encoder_2
            .as_ref()
            .is_some_and(|path| path.exists()),
        ModelComponent::ClipTokenizer2 => paths
            .clip_tokenizer_2
            .as_ref()
            .is_some_and(|path| path.exists()),
        ModelComponent::TextEncoder => {
            let path = paths.text_encoder_files.get(text_encoder_idx);
            text_encoder_idx += 1;
            path.is_some_and(|path| path.exists())
        }
        ModelComponent::TextTokenizer => paths
            .text_tokenizer
            .as_ref()
            .is_some_and(|path| path.exists()),
        // H3 remains contract-only. ModelPaths does not yet carry these
        // components, so a manifest can never be mistaken for runnable merely
        // because the legacy transformer/VAE subset exists.
        ModelComponent::AudioVae
        | ModelComponent::Processor
        | ModelComponent::VideoScheduler
        | ModelComponent::AudioScheduler
        | ModelComponent::ModelConfig
        | ModelComponent::TaskConfig => false,
        // PuLID's identity assets are a files-only bundle: `ModelPaths` has no
        // slot for any of them, so a `ModelPaths` can never prove the bundle
        // is present. `crate::pulid_assets` answers that question instead.
        ModelComponent::IdentityAdapter
        | ModelComponent::IdentityVisionEncoder
        | ModelComponent::FaceDetector
        | ModelComponent::FaceRecognizer
        | ModelComponent::FaceParser => false,
        ModelComponent::Decoder => paths.decoder.as_ref().is_some_and(|path| path.exists()),
        ModelComponent::Upscaler => paths.transformer.exists(),
    })
}

/// Parse a device-placement string into a `DeviceRef`.
///
/// Keywords and the legacy `gpu:N` prefix are case-insensitive. Durable IDs
/// preserve their spelling and accept `device:<id>` plus direct `cuda:` and
/// `metal:` forms. Raw NVIDIA `GPU-`/`MIG-` selectors belong to startup GPU
/// selection; component placement uses the exact opaque ID advertised by
/// `/api/devices`.
pub fn parse_device_ref_str(raw: &str) -> Result<crate::types::DeviceRef, String> {
    use crate::types::DeviceRef;
    let raw = raw.trim();
    let normalized = raw.to_ascii_lowercase();
    if normalized == "auto" {
        Ok(DeviceRef::Auto)
    } else if normalized == "cpu" {
        Ok(DeviceRef::Cpu)
    } else if normalized == "gpu" {
        Ok(DeviceRef::gpu(0))
    } else if let Some(rest) = normalized.strip_prefix("gpu:") {
        rest.parse::<usize>()
            .map(DeviceRef::gpu)
            .map_err(|_| invalid_device_ref(raw))
    } else if normalized.starts_with("device:") {
        parse_stable_device_id(&raw["device:".len()..])
    } else if is_stable_device_id(raw) {
        Ok(DeviceRef::device(raw))
    } else {
        Err(invalid_device_ref(raw))
    }
}

fn parse_stable_device_id(raw: &str) -> Result<crate::types::DeviceRef, String> {
    if is_stable_device_id(raw) {
        Ok(crate::types::DeviceRef::device(raw))
    } else {
        Err(invalid_device_ref(raw))
    }
}

fn is_stable_device_id(raw: &str) -> bool {
    let lower = raw.to_ascii_lowercase();
    (lower.starts_with("cuda:") && raw.len() > "cuda:".len())
        || (lower.starts_with("metal:") && raw.len() > "metal:".len())
}

fn invalid_device_ref(raw: &str) -> String {
    format!(
        "invalid device '{raw}' (expected auto|cpu|gpu[:N]|device:<stable-id>|cuda:<id>|metal:<id>)"
    )
}

fn parse_device_ref_env(key: &str) -> Option<crate::types::DeviceRef> {
    let raw = std::env::var(key).ok()?;
    match parse_device_ref_str(&raw) {
        Ok(dr) => Some(dr),
        Err(msg) => {
            eprintln!("mold: ignoring {key}={raw}: {msg}");
            None
        }
    }
}

#[cfg(test)]
mod scheduler_settings_tests {
    use super::{overlay_model_paths, ModelConfig, SchedulerSettings};

    #[test]
    fn manifest_overlay_preserves_paired_expert_paths() {
        let mut target = ModelConfig::default();
        let source = ModelConfig {
            low_noise_transformer: Some("/models/low-noise.gguf".to_string()),
            low_noise_distilled_lora: Some("/models/low-noise-lora.safetensors".to_string()),
            ..ModelConfig::default()
        };

        overlay_model_paths(&mut target, &source);

        assert_eq!(
            target.low_noise_transformer.as_deref(),
            Some("/models/low-noise.gguf")
        );
        assert_eq!(
            target.low_noise_distilled_lora.as_deref(),
            Some("/models/low-noise-lora.safetensors")
        );
    }

    #[test]
    fn scheduler_settings_defaults_and_rejects_invalid_toml() {
        let defaults: SchedulerSettings = toml::from_str("").unwrap();
        assert_eq!(defaults, SchedulerSettings::default());

        let error = toml::from_str::<SchedulerSettings>(
            "replan_debounce_ms = 5000\nreplan_max_delay_ms = 4999",
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("replan_max_delay_ms must be greater than or equal"));
        assert!(toml::from_str::<SchedulerSettings>("warm_wait_max_ms = 30001").is_err());
    }
}
