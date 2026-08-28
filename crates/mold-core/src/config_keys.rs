//! The `mold config` key registry and typed get/set machinery.
//!
//! Moved out of `mold-cli` so the server's `/api/config` surface shares the
//! exact same key registry, parsing/validation, env-override detection, and
//! DB-vs-TOML surface classification as the CLI verbs. Persistence itself is
//! split by dependency direction: TOML writes go through
//! [`crate::Config::save_bootstrap_only`], DB writes through the helpers in
//! `mold_db::config_sync` (mold-core cannot depend on mold-db).

use anyhow::{anyhow, bail, Result};

use crate::config::Config;
use crate::Scheduler;

// ── Key registry ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValueType {
    String,
    OptionalString,
    Bool,
    U16,
    U32,
    F64,
}

/// Days a trashed print stays in `<output_dir>/.trash/` before the sweeper
/// purges it. `0` keeps trashed prints forever. Profile-scoped DB key.
pub const GALLERY_TRASH_RETENTION_DAYS_KEY: &str = "gallery.trash_retention_days";
pub const GALLERY_TRASH_RETENTION_DAYS_ENV: &str = "MOLD_GALLERY_TRASH_RETENTION_DAYS";
/// Days a held durable queue row is kept before the sweeper purges it, and
/// days a fully settled batch summary is kept after its last child settled.
/// `0` keeps both forever. Profile-scoped DB key.
pub const QUEUE_HELD_RETENTION_DAYS_KEY: &str = "queue.held_retention_days";
pub const QUEUE_HELD_RETENTION_DAYS_ENV: &str = "MOLD_QUEUE_HELD_RETENTION_DAYS";

/// Whether a titled print is also tagged with its title slug, by the CLIENT
/// that submits it. Profile-scoped DB key, default true.
///
/// Deliberately has no env override: it shapes what the CLI puts in a
/// request, not how the server behaves, so a host-level environment variable
/// would be the wrong lever.
pub const GENERATE_AUTO_TAG_TITLE_KEY: &str = "generate.auto_tag_title";

pub struct ConfigKeyInfo {
    pub key: &'static str,
    pub value_type: ValueType,
    pub env_var: Option<&'static str>,
    pub section: &'static str,
}

pub const ALL_KEYS: &[ConfigKeyInfo] = &[
    // General
    ConfigKeyInfo {
        key: "default_model",
        value_type: ValueType::String,
        env_var: Some("MOLD_DEFAULT_MODEL"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "models_dir",
        value_type: ValueType::String,
        env_var: Some("MOLD_MODELS_DIR"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "output_dir",
        value_type: ValueType::OptionalString,
        env_var: Some("MOLD_OUTPUT_DIR"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "server_port",
        value_type: ValueType::U16,
        env_var: None,
        section: "General",
    },
    ConfigKeyInfo {
        key: "default_width",
        value_type: ValueType::U32,
        env_var: None,
        section: "General",
    },
    ConfigKeyInfo {
        key: "default_height",
        value_type: ValueType::U32,
        env_var: None,
        section: "General",
    },
    ConfigKeyInfo {
        key: "default_steps",
        value_type: ValueType::U32,
        env_var: None,
        section: "General",
    },
    ConfigKeyInfo {
        key: "embed_metadata",
        value_type: ValueType::Bool,
        env_var: Some("MOLD_EMBED_METADATA"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "t5_variant",
        value_type: ValueType::OptionalString,
        env_var: Some("MOLD_T5_VARIANT"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "umt5_variant",
        value_type: ValueType::OptionalString,
        env_var: Some("MOLD_UMT5_VARIANT"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "qwen3_variant",
        value_type: ValueType::OptionalString,
        env_var: Some("MOLD_QWEN3_VARIANT"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "default_negative_prompt",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "General",
    },
    // Expand
    ConfigKeyInfo {
        key: "expand.enabled",
        value_type: ValueType::Bool,
        env_var: Some("MOLD_EXPAND"),
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.backend",
        value_type: ValueType::String,
        env_var: Some("MOLD_EXPAND_BACKEND"),
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.model",
        value_type: ValueType::String,
        env_var: Some("MOLD_EXPAND_MODEL"),
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.api_model",
        value_type: ValueType::String,
        env_var: Some("MOLD_EXPAND_MODEL"),
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.temperature",
        value_type: ValueType::F64,
        env_var: Some("MOLD_EXPAND_TEMPERATURE"),
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.top_p",
        value_type: ValueType::F64,
        env_var: None,
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.max_tokens",
        value_type: ValueType::U32,
        env_var: None,
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.thinking",
        value_type: ValueType::Bool,
        env_var: Some("MOLD_EXPAND_THINKING"),
        section: "Expand",
    },
    // Scheduler
    ConfigKeyInfo {
        key: "scheduler.replan_debounce_ms",
        value_type: ValueType::U32,
        env_var: None,
        section: "Scheduler",
    },
    ConfigKeyInfo {
        key: "scheduler.replan_max_delay_ms",
        value_type: ValueType::U32,
        env_var: None,
        section: "Scheduler",
    },
    ConfigKeyInfo {
        key: "scheduler.warm_wait_max_ms",
        value_type: ValueType::U32,
        env_var: None,
        section: "Scheduler",
    },
    // Gallery
    ConfigKeyInfo {
        key: GALLERY_TRASH_RETENTION_DAYS_KEY,
        value_type: ValueType::U32,
        env_var: Some(GALLERY_TRASH_RETENTION_DAYS_ENV),
        section: "Gallery",
    },
    // Queue
    ConfigKeyInfo {
        key: QUEUE_HELD_RETENTION_DAYS_KEY,
        value_type: ValueType::U32,
        env_var: Some(QUEUE_HELD_RETENTION_DAYS_ENV),
        section: "Queue",
    },
    // Generate
    ConfigKeyInfo {
        key: GENERATE_AUTO_TAG_TITLE_KEY,
        value_type: ValueType::Bool,
        env_var: None,
        section: "Generate",
    },
    // Logging
    ConfigKeyInfo {
        key: "logging.level",
        value_type: ValueType::String,
        env_var: None,
        section: "Logging",
    },
    ConfigKeyInfo {
        key: "logging.file",
        value_type: ValueType::Bool,
        env_var: None,
        section: "Logging",
    },
    ConfigKeyInfo {
        key: "logging.dir",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "Logging",
    },
    ConfigKeyInfo {
        key: "logging.max_days",
        value_type: ValueType::U32,
        env_var: None,
        section: "Logging",
    },
    // RunPod
    ConfigKeyInfo {
        key: "runpod.api_key",
        value_type: ValueType::OptionalString,
        env_var: Some("RUNPOD_API_KEY"),
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.default_gpu",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.default_datacenter",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.default_network_volume_id",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.auto_teardown",
        value_type: ValueType::Bool,
        env_var: None,
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.auto_teardown_idle_mins",
        value_type: ValueType::U32,
        env_var: None,
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.cost_alert_usd",
        value_type: ValueType::F64,
        env_var: None,
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.endpoint",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "RunPod",
    },
    // Lambda
    ConfigKeyInfo {
        key: "lambda.api_key",
        value_type: ValueType::OptionalString,
        env_var: Some("LAMBDA_API_KEY"),
        section: "Lambda",
    },
    ConfigKeyInfo {
        key: "lambda.endpoint",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "Lambda",
    },
    ConfigKeyInfo {
        key: "lambda.image_repository",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "Lambda",
    },
    ConfigKeyInfo {
        key: "lambda.ssh_key_name",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "Lambda",
    },
    ConfigKeyInfo {
        key: "lambda.ssh_private_key_path",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "Lambda",
    },
    ConfigKeyInfo {
        key: "lambda.filesystem_prefix",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "Lambda",
    },
    ConfigKeyInfo {
        key: "lambda.filesystem_mount_path",
        value_type: ValueType::String,
        env_var: None,
        section: "Lambda",
    },
    ConfigKeyInfo {
        key: "lambda.confirm_hourly_usd",
        value_type: ValueType::F64,
        env_var: None,
        section: "Lambda",
    },
    ConfigKeyInfo {
        key: "lambda.local_port",
        value_type: ValueType::U16,
        env_var: None,
        section: "Lambda",
    },
];

/// Per-model field names and their value types.
pub const MODEL_FIELDS: &[(&str, ValueType)] = &[
    ("default_steps", ValueType::U32),
    ("default_guidance", ValueType::F64),
    ("default_width", ValueType::U32),
    ("default_height", ValueType::U32),
    ("scheduler", ValueType::OptionalString),
    ("negative_prompt", ValueType::OptionalString),
    ("lora", ValueType::OptionalString),
    ("lora_scale", ValueType::F64),
];

// ── Value representation ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ConfigValue {
    String(String),
    U16(u16),
    U32(u32),
    F64(f64),
    Bool(bool),
    None,
}

impl ConfigValue {
    pub fn display(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::U16(v) => v.to_string(),
            Self::U32(v) => v.to_string(),
            Self::F64(v) => format!("{v:.}"),
            Self::Bool(v) => if *v { "true" } else { "false" }.to_string(),
            Self::None => "(not set)".to_string(),
        }
    }

    pub fn raw(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::U16(v) => v.to_string(),
            Self::U32(v) => v.to_string(),
            Self::F64(v) => format!("{v:.}"),
            Self::Bool(v) => if *v { "true" } else { "false" }.to_string(),
            Self::None => String::new(),
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::String(s) => serde_json::Value::String(s.clone()),
            Self::U16(v) => serde_json::json!(*v),
            Self::U32(v) => serde_json::json!(*v),
            Self::F64(v) => serde_json::json!(*v),
            Self::Bool(v) => serde_json::Value::Bool(*v),
            Self::None => serde_json::Value::Null,
        }
    }
}

// ── Key lookup helpers ──────────────────────────────────────────────

pub fn find_static_key(key: &str) -> Option<&'static ConfigKeyInfo> {
    ALL_KEYS.iter().find(|k| k.key == key)
}

/// True when `key` names a static config key or a well-formed
/// `models.<name>.<field>` key.
pub fn is_known_key(key: &str) -> bool {
    find_static_key(key).is_some() || (key.starts_with("models.") && parse_model_key(key).is_ok())
}

/// Parse a `models.<name>.<field>` key, returning (model_name, field_name, value_type).
pub fn parse_model_key(key: &str) -> Result<(&str, &str, ValueType)> {
    let rest = key
        .strip_prefix("models.")
        .ok_or_else(|| anyhow!("not a model key: {key}"))?;
    // Find the last dot — everything before it is the model name (may contain dots/colons),
    // everything after is the field name.
    let last_dot = rest
        .rfind('.')
        .ok_or_else(|| anyhow!("invalid model key (expected models.<name>.<field>): {key}"))?;
    let model_name = &rest[..last_dot];
    let field_name = &rest[last_dot + 1..];
    if model_name.is_empty() || field_name.is_empty() {
        bail!("invalid model key: {key}");
    }
    let vt = MODEL_FIELDS
        .iter()
        .find(|(f, _)| *f == field_name)
        .map(|(_, vt)| *vt)
        .ok_or_else(|| {
            let valid: Vec<&str> = MODEL_FIELDS.iter().map(|(f, _)| *f).collect();
            anyhow!(
                "unknown model field '{}'. Valid fields: {}",
                field_name,
                valid.join(", ")
            )
        })?;
    Ok((model_name, field_name, vt))
}

// ── Read a config value ─────────────────────────────────────────────

pub fn get_value(config: &Config, key: &str) -> Result<ConfigValue> {
    // Try static keys first
    if find_static_key(key).is_some() {
        return get_static_value(config, key);
    }
    // Try model keys
    if key.starts_with("models.") {
        return get_model_value(config, key);
    }
    // Unknown key
    Err(unknown_key_error(key))
}

pub fn get_static_value(config: &Config, key: &str) -> Result<ConfigValue> {
    Ok(match key {
        // General
        "default_model" => ConfigValue::String(config.default_model.clone()),
        "models_dir" => ConfigValue::String(config.models_dir.clone()),
        "output_dir" => match &config.output_dir {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "server_port" => ConfigValue::U16(config.server_port),
        "default_width" => ConfigValue::U32(config.default_width),
        "default_height" => ConfigValue::U32(config.default_height),
        "default_steps" => ConfigValue::U32(config.default_steps),
        "embed_metadata" => ConfigValue::Bool(config.embed_metadata),
        "t5_variant" => match &config.t5_variant {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::String("auto".into()),
        },
        "qwen3_variant" => match &config.qwen3_variant {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::String("auto".into()),
        },
        "default_negative_prompt" => match &config.default_negative_prompt {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        // Expand
        "expand.enabled" => ConfigValue::Bool(config.expand.enabled),
        "expand.backend" => ConfigValue::String(config.expand.backend.clone()),
        "expand.model" => ConfigValue::String(config.expand.model.clone()),
        "expand.api_model" => ConfigValue::String(config.expand.api_model.clone()),
        "expand.temperature" => ConfigValue::F64(config.expand.temperature),
        "expand.top_p" => ConfigValue::F64(config.expand.top_p),
        "expand.max_tokens" => ConfigValue::U32(config.expand.max_tokens),
        "expand.thinking" => ConfigValue::Bool(config.expand.thinking),
        // Scheduler
        "scheduler.replan_debounce_ms" => ConfigValue::U32(config.scheduler.replan_debounce_ms),
        "scheduler.replan_max_delay_ms" => ConfigValue::U32(config.scheduler.replan_max_delay_ms),
        "scheduler.warm_wait_max_ms" => ConfigValue::U32(config.scheduler.warm_wait_max_ms),
        // Gallery
        GALLERY_TRASH_RETENTION_DAYS_KEY => ConfigValue::U32(config.gallery.trash_retention_days),
        // Queue
        QUEUE_HELD_RETENTION_DAYS_KEY => ConfigValue::U32(config.queue.held_retention_days),
        // Generate
        GENERATE_AUTO_TAG_TITLE_KEY => ConfigValue::Bool(config.generate.auto_tag_title),
        // Logging
        "logging.level" => ConfigValue::String(config.logging.level.clone()),
        "logging.file" => ConfigValue::Bool(config.logging.file),
        "logging.dir" => match &config.logging.dir {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "logging.max_days" => ConfigValue::U32(config.logging.max_days),
        // RunPod
        "runpod.api_key" => match &config.runpod.api_key {
            Some(_) => ConfigValue::String("<set>".to_string()),
            None => ConfigValue::None,
        },
        "runpod.default_gpu" => match &config.runpod.default_gpu {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "runpod.default_datacenter" => match &config.runpod.default_datacenter {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "runpod.default_network_volume_id" => match &config.runpod.default_network_volume_id {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "runpod.auto_teardown" => ConfigValue::Bool(config.runpod.auto_teardown),
        "runpod.auto_teardown_idle_mins" => ConfigValue::U32(config.runpod.auto_teardown_idle_mins),
        "runpod.cost_alert_usd" => ConfigValue::F64(config.runpod.cost_alert_usd),
        "runpod.endpoint" => match &config.runpod.endpoint {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        // Lambda
        "lambda.api_key" => match &config.lambda.api_key {
            Some(_) => ConfigValue::String("<set>".to_string()),
            None => ConfigValue::None,
        },
        "lambda.endpoint" => match &config.lambda.endpoint {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "lambda.image_repository" => match &config.lambda.image_repository {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "lambda.ssh_key_name" => match &config.lambda.ssh_key_name {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "lambda.ssh_private_key_path" => match &config.lambda.ssh_private_key_path {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "lambda.filesystem_prefix" => match &config.lambda.filesystem_prefix {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "lambda.filesystem_mount_path" => {
            ConfigValue::String(config.lambda.filesystem_mount_path.clone())
        }
        "lambda.confirm_hourly_usd" => ConfigValue::F64(config.lambda.confirm_hourly_usd),
        "lambda.local_port" => ConfigValue::U16(config.lambda.local_port),
        _ => return Err(unknown_key_error(key)),
    })
}

pub fn get_model_value(config: &Config, key: &str) -> Result<ConfigValue> {
    let (model_name, field_name, _) = parse_model_key(key)?;
    let mc = config.models.get(model_name).ok_or_else(|| {
        anyhow!(
            "no model '{}' in config. Run mold config list to see configured models.",
            model_name,
        )
    })?;
    Ok(match field_name {
        "default_steps" => mc
            .default_steps
            .map(ConfigValue::U32)
            .unwrap_or(ConfigValue::None),
        "default_guidance" => mc
            .default_guidance
            .map(ConfigValue::F64)
            .unwrap_or(ConfigValue::None),
        "default_width" => mc
            .default_width
            .map(ConfigValue::U32)
            .unwrap_or(ConfigValue::None),
        "default_height" => mc
            .default_height
            .map(ConfigValue::U32)
            .unwrap_or(ConfigValue::None),
        "scheduler" => mc
            .scheduler
            .map(|s| ConfigValue::String(s.to_string()))
            .unwrap_or(ConfigValue::None),
        "negative_prompt" => mc
            .negative_prompt
            .as_ref()
            .map(|s| ConfigValue::String(s.clone()))
            .unwrap_or(ConfigValue::None),
        "lora" => mc
            .lora
            .as_ref()
            .map(|s| ConfigValue::String(s.clone()))
            .unwrap_or(ConfigValue::None),
        "lora_scale" => mc
            .lora_scale
            .map(ConfigValue::F64)
            .unwrap_or(ConfigValue::None),
        _ => return Err(unknown_key_error(key)),
    })
}

// ── Write a config value ────────────────────────────────────────────

pub fn set_value(config: &mut Config, key: &str, raw: &str) -> Result<()> {
    if find_static_key(key).is_some() {
        return set_static_value(config, key, raw);
    }
    if key.starts_with("models.") {
        return set_model_value(config, key, raw);
    }
    Err(unknown_key_error(key))
}

fn set_static_value(config: &mut Config, key: &str, raw: &str) -> Result<()> {
    match key {
        // General
        "default_model" => config.default_model = parse_string(raw)?,
        "models_dir" => config.models_dir = parse_string(raw)?,
        "output_dir" => config.output_dir = parse_optional_string(raw),
        "server_port" => config.server_port = parse_u16(raw, 1, 65535, key)?,
        "default_width" => config.default_width = parse_u32(raw, 64, 8192, key)?,
        "default_height" => config.default_height = parse_u32(raw, 64, 8192, key)?,
        "default_steps" => config.default_steps = parse_u32(raw, 1, 1000, key)?,
        "embed_metadata" => config.embed_metadata = parse_bool(raw, key)?,
        "t5_variant" => {
            let val = parse_optional_string(raw);
            if let Some(ref v) = val {
                validate_enum(v, &["auto", "fp16", "q8", "q6", "q5", "q4", "q3"], key)?;
                config.t5_variant = if v == "auto" { None } else { val };
            } else {
                config.t5_variant = None;
            }
        }
        "qwen3_variant" => {
            let val = parse_optional_string(raw);
            if let Some(ref v) = val {
                validate_enum(v, &["auto", "bf16", "q8", "q6", "iq4", "q3"], key)?;
                config.qwen3_variant = if v == "auto" { None } else { val };
            } else {
                config.qwen3_variant = None;
            }
        }
        "default_negative_prompt" => config.default_negative_prompt = parse_optional_string(raw),
        // Expand
        "expand.enabled" => config.expand.enabled = parse_bool(raw, key)?,
        "expand.backend" => config.expand.backend = parse_string(raw)?,
        "expand.model" => config.expand.model = parse_string(raw)?,
        "expand.api_model" => config.expand.api_model = parse_string(raw)?,
        "expand.temperature" => config.expand.temperature = parse_f64(raw, 0.0, 2.0, key)?,
        "expand.top_p" => config.expand.top_p = parse_f64(raw, 0.0, 1.0, key)?,
        "expand.max_tokens" => config.expand.max_tokens = parse_u32(raw, 1, 65535, key)?,
        "expand.thinking" => config.expand.thinking = parse_bool(raw, key)?,
        // Scheduler
        "scheduler.replan_debounce_ms"
        | "scheduler.replan_max_delay_ms"
        | "scheduler.warm_wait_max_ms" => {
            let mut scheduler = config.scheduler;
            let value = parse_u32(raw, 0, crate::config::SCHEDULER_TIMING_MAX_MS, key)?;
            match key {
                "scheduler.replan_debounce_ms" => scheduler.replan_debounce_ms = value,
                "scheduler.replan_max_delay_ms" => scheduler.replan_max_delay_ms = value,
                "scheduler.warm_wait_max_ms" => scheduler.warm_wait_max_ms = value,
                _ => unreachable!("scheduler key match is exhaustive"),
            }
            config.scheduler = scheduler.validate()?;
        }
        // Gallery
        GALLERY_TRASH_RETENTION_DAYS_KEY => {
            config.gallery.trash_retention_days =
                parse_u32(raw, 0, crate::config::GALLERY_TRASH_RETENTION_MAX_DAYS, key)
                    .map_err(|error| anyhow!("{error} Use 0 to keep trashed prints forever."))?;
        }
        // Queue
        QUEUE_HELD_RETENTION_DAYS_KEY => {
            config.queue.held_retention_days =
                parse_u32(raw, 0, crate::config::GALLERY_TRASH_RETENTION_MAX_DAYS, key)
                    .map_err(|error| anyhow!("{error} Use 0 to keep held rows forever."))?;
        }
        // Generate
        GENERATE_AUTO_TAG_TITLE_KEY => {
            config.generate.auto_tag_title = parse_bool(raw, key)?;
        }
        // Logging
        "logging.level" => {
            validate_enum(raw, &["trace", "debug", "info", "warn", "error"], key)?;
            config.logging.level = raw.to_string();
        }
        "logging.file" => config.logging.file = parse_bool(raw, key)?,
        "logging.dir" => config.logging.dir = parse_optional_string(raw),
        "logging.max_days" => config.logging.max_days = parse_u32(raw, 1, 3650, key)?,
        // RunPod
        "runpod.api_key" => config.runpod.api_key = parse_optional_string(raw),
        "runpod.default_gpu" => config.runpod.default_gpu = parse_optional_string(raw),
        "runpod.default_datacenter" => {
            config.runpod.default_datacenter = parse_optional_string(raw)
        }
        "runpod.default_network_volume_id" => {
            config.runpod.default_network_volume_id = parse_optional_string(raw)
        }
        "runpod.auto_teardown" => config.runpod.auto_teardown = parse_bool(raw, key)?,
        "runpod.auto_teardown_idle_mins" => {
            config.runpod.auto_teardown_idle_mins = parse_u32(raw, 0, 10_080, key)?
        }
        "runpod.cost_alert_usd" => config.runpod.cost_alert_usd = parse_f64(raw, 0.0, 1000.0, key)?,
        "runpod.endpoint" => config.runpod.endpoint = parse_optional_string(raw),
        // Lambda
        "lambda.api_key" => config.lambda.api_key = parse_optional_string(raw),
        "lambda.endpoint" => config.lambda.endpoint = parse_optional_string(raw),
        "lambda.image_repository" => config.lambda.image_repository = parse_optional_string(raw),
        "lambda.ssh_key_name" => config.lambda.ssh_key_name = parse_optional_string(raw),
        "lambda.ssh_private_key_path" => {
            config.lambda.ssh_private_key_path = parse_optional_string(raw)
        }
        "lambda.filesystem_prefix" => config.lambda.filesystem_prefix = parse_optional_string(raw),
        "lambda.filesystem_mount_path" => config.lambda.filesystem_mount_path = parse_string(raw)?,
        "lambda.confirm_hourly_usd" => {
            config.lambda.confirm_hourly_usd = parse_f64(raw, 0.0, 1000.0, key)?
        }
        "lambda.local_port" => config.lambda.local_port = parse_u16(raw, 1, 65535, key)?,
        _ => return Err(unknown_key_error(key)),
    }
    Ok(())
}

fn set_model_value(config: &mut Config, key: &str, raw: &str) -> Result<()> {
    let (model_name, field_name, _) = parse_model_key(key)?;
    // Create model entry if it doesn't exist
    let mc = config.models.entry(model_name.to_string()).or_default();
    match field_name {
        "default_steps" => mc.default_steps = parse_optional_u32(raw, 1, 1000, key)?,
        "default_guidance" => mc.default_guidance = parse_optional_f64(raw, 0.0, 100.0, key)?,
        "default_width" => mc.default_width = parse_optional_u32(raw, 64, 8192, key)?,
        "default_height" => mc.default_height = parse_optional_u32(raw, 64, 8192, key)?,
        "scheduler" => {
            let val = parse_optional_string(raw);
            mc.scheduler = match val.as_deref() {
                Some("ddim") => Some(Scheduler::Ddim),
                Some("euler-ancestral") => Some(Scheduler::EulerAncestral),
                Some("uni-pc") => Some(Scheduler::UniPc),
                Some(v) => {
                    bail!(
                        "invalid value for {key}: '{}'. Valid: none, ddim, euler-ancestral, uni-pc",
                        v
                    );
                }
                None => None,
            };
        }
        "negative_prompt" => mc.negative_prompt = parse_optional_string(raw),
        "lora" => mc.lora = parse_optional_string(raw),
        "lora_scale" => mc.lora_scale = parse_optional_f64(raw, 0.0, 2.0, key)?,
        _ => return Err(unknown_key_error(key)),
    }
    Ok(())
}

// ── Env override detection ──────────────────────────────────────────

/// Returns (var_name, current_value) if an env var overrides the given key.
pub fn env_override_for(key: &str) -> Option<(&'static str, String)> {
    let info = find_static_key(key)?;
    let var = info.env_var?;
    std::env::var(var).ok().map(|v| (var, v))
}

// ── Parsing helpers ─────────────────────────────────────────────────

pub fn parse_string(raw: &str) -> Result<String> {
    if raw.is_empty() {
        bail!("value cannot be empty");
    }
    Ok(raw.to_string())
}

pub fn parse_optional_string(raw: &str) -> Option<String> {
    match raw.to_lowercase().as_str() {
        "none" | "" => None,
        _ => Some(raw.to_string()),
    }
}

pub fn parse_bool(raw: &str, key: &str) -> Result<bool> {
    match raw.to_lowercase().as_str() {
        "true" | "on" | "1" | "yes" => Ok(true),
        "false" | "off" | "0" | "no" => Ok(false),
        _ => bail!(
            "invalid value for {key}: '{}'. Use true/false, on/off, or 1/0.",
            raw
        ),
    }
}

pub fn parse_u16(raw: &str, min: u16, max: u16, key: &str) -> Result<u16> {
    let val: u16 = raw
        .parse()
        .map_err(|_| anyhow!("invalid value for {key}: '{}'. Must be a number.", raw))?;
    if val < min || val > max {
        bail!("invalid value for {key}: {val}. Must be between {min} and {max}.");
    }
    Ok(val)
}

pub fn parse_u32(raw: &str, min: u32, max: u32, key: &str) -> Result<u32> {
    let val: u32 = raw
        .parse()
        .map_err(|_| anyhow!("invalid value for {key}: '{}'. Must be a number.", raw))?;
    if val < min || val > max {
        bail!("invalid value for {key}: {val}. Must be between {min} and {max}.");
    }
    Ok(val)
}

pub fn parse_f64(raw: &str, min: f64, max: f64, key: &str) -> Result<f64> {
    let val: f64 = raw
        .parse()
        .map_err(|_| anyhow!("invalid value for {key}: '{}'. Must be a number.", raw))?;
    if val < min || val > max {
        bail!("invalid value for {key}: {val}. Must be between {min} and {max}.");
    }
    Ok(val)
}

pub fn parse_optional_u32(raw: &str, min: u32, max: u32, key: &str) -> Result<Option<u32>> {
    if raw.eq_ignore_ascii_case("none") || raw.is_empty() {
        return Ok(None);
    }
    parse_u32(raw, min, max, key).map(Some)
}

pub fn parse_optional_f64(raw: &str, min: f64, max: f64, key: &str) -> Result<Option<f64>> {
    if raw.eq_ignore_ascii_case("none") || raw.is_empty() {
        return Ok(None);
    }
    parse_f64(raw, min, max, key).map(Some)
}

pub fn validate_enum(raw: &str, valid: &[&str], key: &str) -> Result<()> {
    if !valid.contains(&raw) {
        bail!(
            "invalid value for {key}: '{}'. Valid: {}",
            raw,
            valid.join(", ")
        );
    }
    Ok(())
}

pub fn unknown_key_error(key: &str) -> anyhow::Error {
    anyhow!(
        "unknown config key: '{}'. Run {} to see all keys.",
        key,
        "mold config list"
    )
}

// ── Surface classification (DB vs TOML) ─────────────────────────────

/// Where a given config key is persisted after issue #265.
///
/// Reads always go through `Config::load_or_default()` which overlays DB
/// values on top of TOML, so callers of `get` don't care. Writes do —
/// `Db` keys must round-trip through `mold-db` instead of rewriting
/// `config.toml`, and `where` prints this so operators can tell which
/// surface owns a given key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// Persisted in `config.toml` (paths, bootstrap, credentials).
    File,
    /// Persisted in the SQLite `settings` / `model_prefs` tables.
    Db,
}

impl Surface {
    pub fn as_str(self) -> &'static str {
        match self {
            Surface::File => "file",
            Surface::Db => "db",
        }
    }
}

/// Classify a key into its storage surface. Unknown keys return `File`
/// as a conservative default (matches pre-issue-#265 behavior).
pub fn surface_for_key(key: &str) -> Surface {
    // Prefix-based: user-preference slices that moved to DB.
    if key.starts_with("tui.")
        || key.starts_with("expand.")
        || key.starts_with("generate.")
        || key.starts_with("scheduler.")
        || key.starts_with("gallery.")
        || key.starts_with("queue.")
        || key.starts_with("model_prefs.")
    {
        return Surface::Db;
    }
    // Global generation defaults that live on Config but now persist in
    // the DB. Keep the old flat names for backwards-compatible CLI UX.
    matches!(
        key,
        "default_width"
            | "default_height"
            | "default_steps"
            | "embed_metadata"
            | "default_negative_prompt"
            | "t5_variant"
            | "qwen3_variant"
    )
    .then_some(Surface::Db)
    .unwrap_or(Surface::File)
}

/// For `models.<name>.<field>` keys: user-preference fields (generation
/// defaults, LoRA) live in the DB `model_prefs` table; path fields stay
/// in TOML. Returns `None` for non-`models.*` keys.
pub fn model_field_surface(key: &str) -> Option<Surface> {
    let rest = key.strip_prefix("models.")?;
    let field = rest.rsplit('.').next()?;
    let db_fields = [
        "default_steps",
        "default_guidance",
        "default_width",
        "default_height",
        "scheduler",
        "negative_prompt",
        "lora",
        "lora_scale",
    ];
    Some(if db_fields.contains(&field) {
        Surface::Db
    } else {
        Surface::File
    })
}

/// The effective surface for any key — per-model split when applicable,
/// otherwise the static classification.
pub fn effective_surface(key: &str) -> Surface {
    model_field_surface(key).unwrap_or_else(|| surface_for_key(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_surface_splits_model_fields_and_globals() {
        assert_eq!(effective_surface("models.flux-dev:q4.lora"), Surface::Db);
        assert_eq!(
            effective_surface("models.flux-dev:q4.transformer"),
            Surface::File
        );
        assert_eq!(effective_surface("expand.enabled"), Surface::Db);
        assert_eq!(effective_surface("models_dir"), Surface::File);
    }

    #[test]
    fn is_known_key_accepts_static_and_model_keys() {
        assert!(is_known_key("server_port"));
        assert!(is_known_key("expand.enabled"));
        assert!(is_known_key("scheduler.replan_debounce_ms"));
        assert!(is_known_key("models.flux-dev:q4.default_steps"));
        assert!(!is_known_key("definitely.not.a.key"));
        assert!(!is_known_key("models.flux-dev:q4.bogus_field"));
    }

    #[test]
    fn gallery_trash_retention_is_a_registered_db_surface_key() {
        let info = find_static_key(GALLERY_TRASH_RETENTION_DAYS_KEY).expect("registered");
        assert_eq!(info.value_type, ValueType::U32);
        assert_eq!(info.env_var, Some(GALLERY_TRASH_RETENTION_DAYS_ENV));
        assert_eq!(info.section, "Gallery");
        assert!(is_known_key(GALLERY_TRASH_RETENTION_DAYS_KEY));
        assert_eq!(
            surface_for_key(GALLERY_TRASH_RETENTION_DAYS_KEY),
            Surface::Db
        );
        assert_eq!(
            effective_surface(GALLERY_TRASH_RETENTION_DAYS_KEY),
            Surface::Db
        );
        // Unknown siblings under the prefix still route to the DB so a
        // future gallery.* key can never split-brain against the sweeper.
        assert_eq!(surface_for_key("gallery.something_else"), Surface::Db);
    }

    /// `generate.auto_tag_title` is a DB-surface bool that defaults on, is
    /// reachable through the same registry as every other key, and has no
    /// env override — it shapes what a CLIENT puts in a request, so a
    /// host-level environment variable would be the wrong lever.
    #[test]
    fn generate_auto_tag_title_round_trips_on_the_db_surface() {
        assert!(is_known_key(GENERATE_AUTO_TAG_TITLE_KEY));
        assert_eq!(effective_surface(GENERATE_AUTO_TAG_TITLE_KEY), Surface::Db);
        assert_eq!(env_override_for(GENERATE_AUTO_TAG_TITLE_KEY), None);

        let mut config = Config::default();
        assert!(matches!(
            get_value(&config, GENERATE_AUTO_TAG_TITLE_KEY).unwrap(),
            ConfigValue::Bool(true)
        ));

        set_value(&mut config, GENERATE_AUTO_TAG_TITLE_KEY, "false").unwrap();
        assert!(!config.generate.auto_tag_title);
        assert!(matches!(
            get_value(&config, GENERATE_AUTO_TAG_TITLE_KEY).unwrap(),
            ConfigValue::Bool(false)
        ));

        set_value(&mut config, GENERATE_AUTO_TAG_TITLE_KEY, "true").unwrap();
        assert!(config.generate.auto_tag_title);
        assert!(set_value(&mut config, GENERATE_AUTO_TAG_TITLE_KEY, "maybe").is_err());
    }

    #[test]
    fn gallery_trash_retention_round_trips_and_enforces_bounds() {
        let mut config = Config::default();
        assert!(matches!(
            get_value(&config, GALLERY_TRASH_RETENTION_DAYS_KEY).unwrap(),
            ConfigValue::U32(30)
        ));

        set_value(&mut config, GALLERY_TRASH_RETENTION_DAYS_KEY, "0").unwrap();
        assert_eq!(config.gallery.trash_retention_days, 0);
        assert!(matches!(
            get_value(&config, GALLERY_TRASH_RETENTION_DAYS_KEY).unwrap(),
            ConfigValue::U32(0)
        ));

        set_value(&mut config, GALLERY_TRASH_RETENTION_DAYS_KEY, "3650").unwrap();
        assert_eq!(config.gallery.trash_retention_days, 3650);

        let error = set_value(&mut config, GALLERY_TRASH_RETENTION_DAYS_KEY, "3651").unwrap_err();
        assert!(error.to_string().contains("between 0 and 3650"), "{error}");
        assert!(error.to_string().contains("forever"), "{error}");
        assert_eq!(config.gallery.trash_retention_days, 3650);

        let error = set_value(&mut config, GALLERY_TRASH_RETENTION_DAYS_KEY, "-1").unwrap_err();
        assert!(error.to_string().contains("Must be a number"), "{error}");
        let error = set_value(&mut config, GALLERY_TRASH_RETENTION_DAYS_KEY, "soon").unwrap_err();
        assert!(error.to_string().contains("Must be a number"), "{error}");
        assert_eq!(config.gallery.trash_retention_days, 3650);
    }

    #[test]
    fn every_registered_key_reads_from_a_default_config() {
        // Guards against a registry entry whose get/set arm was forgotten:
        // `get_static_value` would return the unknown-key error.
        //
        // `umt5_variant` is a known pre-existing gap (#778 registered the key
        // without get/set arms or a DB persistence slot); it is skipped here
        // rather than silently widened by this guard.
        let config = Config::default();
        for info in ALL_KEYS.iter().filter(|info| info.key != "umt5_variant") {
            get_static_value(&config, info.key)
                .unwrap_or_else(|error| panic!("{} has no getter: {error}", info.key));
        }
    }

    #[test]
    fn scheduler_timings_validate_range_and_cross_field_order_atomically() {
        let mut config = Config::default();
        set_value(&mut config, "scheduler.replan_debounce_ms", "3000").unwrap();
        assert_eq!(config.scheduler.replan_debounce_ms, 3_000);

        let before = config.scheduler;
        let error = set_value(&mut config, "scheduler.replan_max_delay_ms", "2999").unwrap_err();
        assert!(error
            .to_string()
            .contains("replan_max_delay_ms must be greater than or equal"));
        assert_eq!(config.scheduler, before);

        let error = set_value(&mut config, "scheduler.warm_wait_max_ms", "30001").unwrap_err();
        assert!(error.to_string().contains("between 0 and 30000"));
        assert_eq!(config.scheduler, before);
    }
}
